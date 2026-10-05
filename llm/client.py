"""LLM 客户端：一问一答、带工具调用的一轮对话，以及临时错误的有限重试。

只负责「发消息、收消息」，不关心工具怎么执行——那是 Agent 的事；也不认
HTTP / SSE 那一套——那是 api 的事。留给外面的名字是 ask / chat /
chat_with_tools / call_with_stats，Agent 则统一走文件末尾的 call_llm()。

**重试（13.5）**：只重试「明确是临时故障」的错——限流（429）、服务端 5xx、
连接不上 / 超时。参数错、认证失败、模型不存在这些，重试多少次结果都一样，
一次就抛。最多重试 max_retries 次（默认 2，= 最多 3 次尝试），退避是指数
的：backoff、2×backoff……不无限重试，也不无限等。

**统计（13.5）**：一次调用（重试算同一次调用的多次 attempt）产出一份
LLMCallStats：耗时、试了几次、成没成、model、usage。**不记 prompt 正文、
不记完整回复、更不记 API key**——统计里只有数字、名字和错误摘要。

**取消（把 13.4 的开关接进来）**：调用可以带一个取消开关（鸭子类型：
有 raise_if_cancelled() 就行——本模块不 import agent，免得绕出导入环）。
退避等待切成小片睡，每片醒来先问一句「取消了吗」：一次重试的等待绝不会
让取消长时间失效，整轮 timeout 也不会被重试拖长。取消不吞——RunCancelled
不是 OpenAIError，不会被这里的重试逻辑接住，原样穿给调用方。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from config import LLMConfig, load_llm_config

if TYPE_CHECKING:
    # 只为类型检查：运行时不 import，llm 和 agent 之间没有运行期依赖
    from agent.cancel import CancellationContext
    from agent.trace import Trace

# 退避等待的切片：睡这么久就醒来问一次「取消了吗」。它只影响「取消到生效」
# 的延迟——0.1 秒的粒度人感觉不出来，也不会睡得太碎
CANCEL_POLL_SECONDS = 0.1

# 明确可重试的 SDK 异常：连接 / 超时（APITimeoutError 是它的子类，跟着一起
# 中）、服务端 5xx、限流 429。别的（参数错、认证失败、模型不存在）重试没有
# 意义，一次就抛
_RETRYABLE_ERRORS = (APIConnectionError, InternalServerError, RateLimitError)

# 兜底：带状态码但不在上面那几族里的错（比如自定义 transport），也按状态码认
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """调用 LLM 失败（重试完还是失败时，带的是最后一次的明确错误）。"""


@dataclass(frozen=True)
class LLMUsage:
    """接口返回的 token 用量；接口没返回就整个是 None（不编数据）。

    字段名跟着接口走（prompt / completion / total）；某一项接口没给就是
    None，同样不替它算。
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMCallStats:
    """一次 LLM 调用（重试算同一次调用）的统计。

    duration_seconds 是**这一跳**（attempt）的耗时；total_duration_seconds
    是从第一次尝试算起、连退避等待一起的总耗时——重试过的话两者差不少，
    看「模型慢不慢」用前者，看「这一问问了多久」用后者。

    attempt_count 是到这一跳为止累计试了几次：最后一次事件的它就是总次数
    （成功时 = 终于成功的第几次，失败时 = 一共试了几次）。

    这里没有 prompt、没有回复正文、没有 API key——只有数字、名字、错误摘要。
    """

    model: str
    success: bool
    duration_seconds: float
    total_duration_seconds: float
    attempt_count: int
    max_attempts: int
    usage: LLMUsage | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转成可以直接 json.dumps 的普通字典（键固定，缺值给 None / 空串）。"""
        return {
            "model": self.model,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "error": self.error,
        }


def _message_to_dict(message: Any) -> dict[str, Any]:
    """把 SDK 的 message 对象转成普通字典，好存进消息历史再原样发回去。"""
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]

    return result


def _usage_from(response: Any) -> LLMUsage | None:
    """接口这一跳的 token 用量；没带 usage 就是 None（老接口、假客户端都常见）。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return LLMUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _is_retryable(exc: OpenAIError) -> bool:
    """这个错值不值得再试一次：只认明确的临时故障（见模块开头）。"""
    if isinstance(exc, _RETRYABLE_ERRORS):
        return True
    return getattr(exc, "status_code", None) in _RETRYABLE_STATUS


class LLMClient:
    """对 OpenAI 兼容接口的封装：调用、有限重试、统计。

    重试次数和基础退避来自 config（LLM_MAX_RETRIES / LLM_RETRY_BACKOFF）。
    sleep / clock 可以注入：测试里换成假的，退避就不用真等。

    **重试只有这一处**：构造 SDK 客户端时把它的内建重试关掉了
    （`max_retries=0`）。所以 `LLM_MAX_RETRIES=n` 就是「最多 n 次重试」，
    一次尝试 = 一次 HTTP 请求，可数、可测。
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or load_llm_config()
        # max_retries=0 是**必须**的，不是可调项：openai SDK 自己默认带
        # max_retries=2（_constants.DEFAULT_MAX_RETRIES），对 429/5xx/连接
        # 错误自己重试。不关掉的话，两次重试会叠在本模块的重试循环外面：
        #   LLM_MAX_RETRIES=0（README 写「0 = 一次就抛」）实际发 3 次请求，
        #   默认 2 时最坏 3×3=9 次。
        # 更麻烦的是 SDK 的重试发生在一次 create() 调用**内部**，本模块的
        # attempt 计数、Trace 的 llm_start/llm_end 事件和取消检查点全都看
        # 不见它——取消最坏要等 SDK 那几次退避跑完才生效。
        # 重试策略只有这一处：次数、退避、可重试分类都在下面。
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=0,
        )
        self._sleep = sleep
        self._clock = clock

    def ask(self, question: str) -> str:
        """把 question 发给模型，返回模型回复的纯文本（单轮）。"""
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        return self.chat([{"role": "user", "content": question}])

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """把完整消息列表发给模型，返回模型回复的纯文本。"""
        return self.call_with_stats(messages)[0]

    def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """带上工具清单发一轮，返回 assistant 消息字典。

        返回值不是纯文本，因为这一轮模型可能不回答，而是要求调用工具；
        这些要求放在返回字典的 "tool_calls" 里。
        """
        if not tools:
            raise ValueError("tools 不能为空")

        return self.call_with_stats(messages, tools=tools)[0]

    def call_with_stats(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        observer: "Trace | None" = None,
        cancellation: "CancellationContext | None" = None,
    ) -> tuple[Any, LLMCallStats]:
        """调一次模型：带重试、带统计。返回 (老接口会返回的东西, 统计)。

        tools 没给 → 第一个是纯文本；给了 → 是 assistant 消息字典（和
        chat / chat_with_tools 的返回值一模一样）。

        observer（可选）：观察这一跳的对象——有 start_llm_call(model,
        attempt, max_attempts) 和 record_llm_call(stats) 两个方法就行
        （agent/trace.py 的 Trace 就是）。它抛错不影响调用。

        cancellation（可选）：取消开关。退避等待期间也响应它，取消了就抛
        RunCancelled，一次都不会多试。
        """
        if not messages:
            raise ValueError("messages 不能为空")

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if tools:
            kwargs["tools"] = tools

        max_attempts = self.config.max_retries + 1
        total_started = self._clock()
        attempt = 0

        while True:
            # 试之前先看一眼：已经取消了就一次都不发（重试排队的下一次也在
            # 这儿被拦下——上一次的退避等待里也查，见 _wait_before_retry）
            self._check(cancellation)
            attempt += 1
            self._notify(
                observer, "start_llm_call", self.config.model, attempt, max_attempts
            )
            started = self._clock()

            try:
                response = self._client.chat.completions.create(**kwargs)
            except OpenAIError as exc:
                self._notify(
                    observer,
                    "record_llm_call",
                    self._stats(False, started, total_started, attempt, max_attempts,
                                error=str(exc)),
                )
                if attempt > self.config.max_retries or not _is_retryable(exc):
                    # 用完了次数，或者这个错本来就不该重试：抛最后一次的
                    # 明确错误（原文带上，别让它烂在这一层）
                    raise LLMError(self._failure_message(exc, attempt)) from exc
                self._wait_before_retry(attempt, cancellation)
                continue

            if not response.choices:
                self._notify(
                    observer,
                    "record_llm_call",
                    self._stats(False, started, total_started, attempt, max_attempts,
                                error="LLM 返回了空结果（choices 为空）"),
                )
                raise LLMError("LLM 返回了空结果（choices 为空）")

            stats = self._stats(
                True, started, total_started, attempt, max_attempts,
                usage=_usage_from(response),
            )
            self._notify(observer, "record_llm_call", stats)
            message = response.choices[0].message
            if tools:
                return _message_to_dict(message), stats
            return message.content or "", stats

    def _stats(
        self,
        success: bool,
        started: float,
        total_started: float,
        attempt: int,
        max_attempts: int,
        *,
        usage: LLMUsage | None = None,
        error: str = "",
    ) -> LLMCallStats:
        """凑一份统计：耗时从两个起点现算，不让调用方传时间。"""
        now = self._clock()
        return LLMCallStats(
            model=self.config.model,
            success=success,
            duration_seconds=now - started,
            total_duration_seconds=now - total_started,
            attempt_count=attempt,
            max_attempts=max_attempts,
            usage=usage,
            error=error,
        )

    def _failure_message(self, exc: OpenAIError, attempt: int) -> str:
        """最终失败时的说明：重试过就把「试了几次」说上，错误原文不动。"""
        if attempt > 1:
            return f"调用 LLM 失败：{exc}（重试 {attempt - 1} 次后仍失败）"
        return f"调用 LLM 失败：{exc}"

    def _wait_before_retry(
        self, attempt: int, cancellation: "CancellationContext | None"
    ) -> None:
        """退避等待：backoff × 2**(尝试次数-1)，随时能被取消打断。

        整段切成 0.1 秒的小片，每片醒来问一句「取消了吗」——一次重试的
        等待绝不会让取消长时间失效。backoff 配成 0 就一跳都不睡。
        """
        remaining = self.config.retry_backoff * (2 ** (attempt - 1))
        while remaining > 0:
            self._check(cancellation)
            step = min(CANCEL_POLL_SECONDS, remaining)
            self._sleep(step)
            remaining -= step
        # 最后一跳之后再看一眼：正好在这期间被取消，就别开始下一跳
        self._check(cancellation)

    @staticmethod
    def _check(cancellation: "CancellationContext | None") -> None:
        """取消开关（鸭子类型）：有 raise_if_cancelled() 就问一句。"""
        if cancellation is not None:
            cancellation.raise_if_cancelled()

    @staticmethod
    def _notify(observer: "Trace | None", name: str, *args: Any) -> None:
        """叫一声观察者；它炸了就当没发生过。

        观察者是外人（网络连接、日志之类），它的错不归这一层管，更不能把
        一次已经发出去的调用变成失败——和 agent/trace.py 一个分寸。
        """
        if observer is None:
            return
        try:
            getattr(observer, name)(*args)
        except Exception:
            pass


def call_llm(
    client: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    observer: "Trace | None" = None,
    cancellation: "CancellationContext | None" = None,
) -> Any:
    """Agent 调模型的唯一入口：能带统计就带，不能就退回老接口。

    真 LLMClient 有 call_with_stats()，走它（重试 + 统计 + 观察者）；测试
    替身 / 自定义客户端只有 chat / chat_with_tools，就走老接口——它们本来
    也没有重试和 usage，这样谁都不用为这一层改签名。
    """
    entry = getattr(client, "call_with_stats", None)
    if entry is None:
        if tools:
            return client.chat_with_tools(messages, tools)
        return client.chat(messages)

    return entry(
        messages, tools, observer=observer, cancellation=cancellation
    )[0]
