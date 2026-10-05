"""Agent：维护会话消息历史，并在需要时调用工具，直到拿到最终回答。

run() 一次调用就是一个 tool-calling loop：
    用户输入 -> LLM -> 要求调用工具就执行 -> 结果回灌 -> LLM -> ... -> 最终回答

11.2 起一轮任务前后还各问一次 Memory（memory/context.py）：开始时按输入
检索历史、把「仅供参考」的那段提示夹进请求，结束时把这次的结果记一条。
检索 / 写入都是只降级不抛错的，Agent 这边只有两个调用点。

13.4 起 run() 还可以带一个取消开关（agent/cancel.py）：给了就在每轮开始 /
模型调用前后 / 工具执行前后 / 验证与修复步骤之间看一眼，取消了就抛
RunCancelled 正常停下（这一轮的消息一起回滚）。不给就一个检查点都不走，
行为和以前一字不差。

13.5 起模型调用统一走 llm.call_llm()：客户端自己负责重试、退避、统计
（llm/client.py），这一层只把取消开关和观察者递过去——HTTP 细节、重试
策略都不在这条推理循环里。模型调用的观察（llm_start / llm_end）和工具
一样挂在 Trace 上，取消开关也一并在退避期间生效。
"""

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent.cancel import CancellationContext, RunCancelled
from agent.context import ContextManager
from agent.history import RepairHistory
from agent.prompt import DEFAULT_SYSTEM_PROMPT
from agent.repair import REPORT, NUDGE, DEFAULT_MAX_REPAIR_ROUNDS, RepairLoop
from agent.trace import Trace, TraceEvent
from agent.verify import Verifier
from llm import LLMClient, LLMError, call_llm
from memory.context import MemoryContext
from tools import ToolExecutor, ToolRegistry

if TYPE_CHECKING:
    # 只为类型检查：运行时不 import（llm 已经在上面 import 了，这里只是名字）
    from llm import LLMCallStats

# 一轮对话里最多允许「模型要求调用工具」几轮，防止模型停不下来。
#
# 别往下调。实测：单是问「这个项目里的工具是怎么注册进去的」，模型就要跑
# 4~6 轮工具调用（看它一次批几条），5 是完全没余量的——同一个问题跑两次，
# 一次正好答完，一次撞墙。真实一点的任务只会更长。
DEFAULT_MAX_STEPS = 15


class AgentError(RuntimeError):
    """Agent 运行失败，例如工具调用轮数超出上限。"""


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    """构造一条回灌给模型的工具结果消息。"""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _check_cancelled(cancellation: CancellationContext | None) -> None:
    """检查点：给了取消开关就问一句，取消了抛 RunCancelled。

    没给（默认）什么都不做——不传的调用方行为和以前一字不差。
    """
    if cancellation is not None:
        cancellation.raise_if_cancelled()


class Agent:
    """最小可用的 Agent。

    - messages：本次会话的完整历史，每次请求都整份发给 LLM。
    - tools：可用工具。模型看到它们的 schema 后，通过 tool_calls 决定用哪个。
    - executor：工具执行层。Agent 不直接执行工具，一律走
      Agent → ToolExecutor → ToolRegistry → Tool，执行细节不混进推理循环。
    - verifier：改完代码验一次的那套东西。走同一个执行层，所以验证本身
      在 trace 里也看得到，跟模型自己发起的调用一样。
    - repair：验证没过时接着修的那层（最多 max_repair_rounds 轮）。它和
      max_steps 是两回事：max_steps 管一次 run() 里最多问几次模型，
      max_repair_rounds 管「最多修几轮」。修过的话，最终回答前面会多一句
      修复摘要（几轮、卡在哪一关、成没成）——那是 repair 给的，不是模型编的。
    - history：把每次 run() 的修复过程写一份 JSON 留档（agent/history.py）。
      不装就是不记——**写历史只是附带的事**，它失败不影响这一轮的结果。
    - memory：跨任务的历史记忆（memory/context.py）。装了就在任务开始前
      检索一段「仅供参考」的提示、结束后存一条记录；不装（None）就一个
      调用点都不走——**它同样是附带的事**，检索 / 写入失败都不影响任务。
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: ToolRegistry | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        context: ContextManager | None = None,
        max_repair_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS,
        history: RepairHistory | None = None,
        memory: MemoryContext | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps 至少为 1")

        self.client = client or LLMClient()
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else ToolRegistry()
        self.executor = ToolExecutor(self.tools)
        self.verifier = Verifier(self.tools, self.executor)
        self.repair = RepairLoop(self.verifier, max_repair_rounds)
        self.history = history  # 没装就是 None：不记历史，别的都一样
        self.memory = memory  # 同上：没装就一行 Memory 代码都不走
        # messages 是完整历史，context 决定每次请求实际发多少出去
        self.context = context if context is not None else ContextManager()
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._start_trace()

    def _start_trace(
        self,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[TraceEvent], None] | None = None,
        on_llm_start: Callable[[str, int, int], None] | None = None,
        on_llm_end: Callable[["LLMCallStats"], None] | None = None,
    ) -> None:
        """换一个全新的 Trace 给这一轮用，上一轮的记录原样留在原对象里。

        用「换新对象」而不是「清空旧对象」：外部如果还拿着上一轮拿到的列表，
        不会被这一轮悄悄改掉。

        观察者（13.2 的工具、13.5 的模型调用）跟着这一轮的 Trace 走：下一轮
        不带就是 None——上一轮的回调绝不会留在执行层继续发通知。
        """
        self.trace = Trace(
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
        )
        self.executor.trace = self.trace

    def get_trace(self) -> list[TraceEvent]:
        """返回当前这一轮的工具调用记录（和 messages 一样，拿到的是本体）。"""
        return self.trace.events

    def run(
        self,
        user_input: str,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[TraceEvent], None] | None = None,
        cancellation: CancellationContext | None = None,
        on_llm_start: Callable[[str, int, int], None] | None = None,
        on_llm_end: Callable[["LLMCallStats"], None] | None = None,
    ) -> str:
        """处理一轮用户输入，返回最终回答。

        on_tool_start / on_tool_end（13.2，都可以不给）：这一轮的工具调用
        观察者，见 agent/trace.py。流式接口（api/sse.py）靠它把进度实时推给
        前端；不给就一条通知都不发，行为和以前一字不差。

        on_llm_start / on_llm_end（13.5，都可以不给）：这一轮的模型调用
        观察者，签名见 agent/trace.py。重试过就有多对通知——一跳一对；
        客户端（llm/client.py）负责在哪一刻调它们，Agent 只管把回调交出去。

        cancellation（13.4，可以不给）：这一轮的取消开关（agent/cancel.py）。
        给了就在几个边界上检查（每轮开始 / 模型调用前后 / 工具执行前后 /
        验证与修复之间），取消了抛 RunCancelled 正常停下——不吞这个信号，
        也不硬杀工具；不给就一个检查点都不走。**一次 Run 一个 context**，
        别把上一轮的传给下一轮。重试的退避等待也认它：模型调用那一层同样
        收着这个开关（13.5）。

        中途失败就把这一轮产生的所有消息（含工具调用）一起回滚，
        保证 messages 里始终是完整的对话，可以直接重试；被取消的那一轮
        同样回滚（RunCancelled 走的是同一条异常路径）。

        **一轮一个修复循环**：进门先把 repair / verifier 的状态清空
        （见 RepairLoop.reset）。上一轮停在哪儿、欠没欠验证，都是上一个
        任务的事，不该跟着进这一轮——一只 Agent 连续问好几个问题是常态
        （CLI 一个进程一只、API 一个会话一只），状态串了就会「之后的每
        一问都返回上一轮的旧报告」。

        正常跑完的话，装了 history 就顺手记一条（agent/history.py）；
        装了 memory 就写一条最终记录（成没成都写，失败的原因写在摘要里）——
        回滚掉的那一轮也写，它同样是「这个任务没做成」的历史。

        Memory 和 history 一样自己把失败咽下去，这里不包 try：附带的事
        绝不能把这一轮的回答 / 异常变成别的东西。
        """
        # 这一轮从零开始记，上一轮的记录不混进来（观察者同理，只属于这一轮）
        self._start_trace(
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_llm_start=on_llm_start,
            on_llm_end=on_llm_end,
        )
        # 修复循环的状态同样是「一次 run() 之内」的：上一轮停下过（stopped）
        # 或还欠着验证（pending），都是**上一个任务**的事。不清掉的话，
        # review() 会在这一轮一进门就早返回上一轮那份 REPORT——模型白调一次、
        # 回答被丢掉，用户拿到旧报告；Memory 也会把之后每个成功任务记成失败。
        self.repair.reset()

        user_input = user_input.strip()
        if not user_input:
            raise ValueError("用户输入不能为空")

        # 一进门就看一眼：受理时就已经被取消（排队期间客户端按了停）的话，
        # Memory、模型、工具一个都别碰
        _check_cancelled(cancellation)

        # 开始前先看历史：拿到提示就夹进请求，拿不到（没装 / 没匹配 /
        # 检索出错）就是 None，这一轮的请求和没有 Memory 时一模一样
        note = self.memory.before(user_input) if self.memory is not None else None

        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            answer = self._run_loop(note, cancellation)
        except (LLMError, AgentError, RunCancelled) as exc:
            del self.messages[start:]
            self._remember(user_input, str(exc), ok=False)
            raise
        except BaseException:
            # 兜底：回滚这件事不能只对「认识的异常」生效。中途抛出去的是
            # 什么类型（畸形响应带来的 TypeError、Ctrl-C 的
            # KeyboardInterrupt……）不重要——要紧的是**别把半截消息留在
            # 历史里**：assistant 要求了工具调用、却没有配对回复，上游会
            # 直接拒掉之后每一次请求，整个会话就废了。
            #
            # 不写 Memory：这几种失败说不清「任务做成没有」，维持原来的
            # 语义（只有上面那三种会留一条失败记录）。
            del self.messages[start:]
            raise

        # 「成没成」看的不是模型说自己成了，是修复循环的状态：停过、
        # 或者还欠着一次没验过的修改，就是没成（和 _run_loop 收尾时
        # 用的是同一条判断）
        ok = self.repair.stopped is None and not self.repair.pending
        self._remember(user_input, answer, ok=ok)

        # 记一笔修复过程。record() 自己把失败咽下去（原因在 last_error 里），
        # 所以这里不用 try——写历史绝不能把这一轮的回答变成异常
        if self.history is not None:
            self.history.record(user_input, self.repair)
        return answer

    def _remember(self, task: str, summary: str, *, ok: bool) -> None:
        """一个 user task 最多写一条 Memory（成功一条、失败也是一条）。"""
        if self.memory is not None:
            self.memory.after(task, summary, ok=ok)

    def _request(self, memory_note: dict[str, Any] | None) -> list[dict[str, Any]]:
        """这一次请求实际发出去的消息：裁剪过的历史 +（可选）Memory 提示。

        Memory 提示**不进 self.messages**：它按当前输入现检索，只属于这
        一次请求；对话历史里存的还是真实发生过的那几条消息。没提示时
        原样返回 context.build 的结果——没有历史记录时请求和以前一字不差。

        插在系统提示后面（build 结果的第 0 条一定是它：system 组永远
        不会被裁掉）；这段文本只有 top_k 行、每行都截过，撑不大。
        """
        built = self.context.build(self.messages)
        if memory_note is None:
            return built
        return [built[0], dict(memory_note), *built[1:]]

    def _run_loop(
        self,
        memory_note: dict[str, Any] | None,
        cancellation: CancellationContext | None = None,
    ) -> str:
        """反复「问模型 -> 执行工具」，直到模型给出不需要工具的回答。

        验证没过的那些轮里，模型想收尾也会被按回去接着修（见 RepairLoop）：
        decision 就是它给的说法——接着问、催一句、还是就此停下报状态。

        memory_note 是这一轮每次请求都要夹上的历史提示（或 None）。
        cancellation 是这一轮的取消开关（或 None）：检查点只放在边界上，
        模型调用和工具执行本身不被打断——取消请求从不让「正在跑的这一步」
        半途而废，只保证它一回来就没人再往下走。
        """
        schemas = [tool.to_schema() for tool in self.tools.list()]

        if not schemas:
            # 没有注册任何工具，退化成普通一问一答。
            _check_cancelled(cancellation)  # 模型调用前
            answer = call_llm(
                self.client,
                self._request(memory_note),
                observer=self.trace,
                cancellation=cancellation,
            )
            _check_cancelled(cancellation)  # 模型调用后
            self.messages.append({"role": "assistant", "content": answer})
            return answer

        for _ in range(self.max_steps):
            # 每轮开始先看一眼：上一轮的工具 / 修复跑完到这儿，取消立刻生效
            _check_cancelled(cancellation)
            # self.messages 是完整历史；发给模型的这份是裁剪过的，发完就丢。
            # 重试 / 退避 / 统计都在 llm/client.py 里，这里只递开关和观察者
            message = call_llm(
                self.client,
                self._request(memory_note),
                tools=schemas,
                observer=self.trace,
                cancellation=cancellation,
            )
            # 模型一回来就查：取消后不再执行它要求的任何工具
            _check_cancelled(cancellation)
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                _check_cancelled(cancellation)  # 进 RepairLoop 之前
                decision = self.repair.review(message.get("content") or "")
                if decision.action == NUDGE:
                    # 还欠着一次没修好的，把模型按回去接着修
                    self.messages.append({"role": "user", "content": decision.text})
                    continue
                if decision.action == REPORT:
                    return self._conclude(decision.text)
                return self._finish(message)

            for call in tool_calls:
                self.messages.append(self._run_tool_call(call, cancellation))

            # 这批工具（含验证）都跑完了，再问修复循环之前先查一次：
            # 取消之后不再调新模型、也不接着修
            _check_cancelled(cancellation)
            # 修复循环在刚才那几次调用里喊停了（到上限、诊断原地打转……）：
            # 不再问模型，直接把状态报出去
            decision = self.repair.review()
            if decision.action == REPORT:
                return self._conclude(decision.text)

        # 步骤用完了。修复还欠着的时候不能抛 AgentError：那会把这一轮
        # （连同已经改好的文件）整个回滚掉，而这里最该做的是把状态说清楚
        if self.repair.stopped is not None or self.repair.pending:
            self.repair.halt(f"工具调用已经用满 {self.max_steps} 轮，修复还没结束")
            return self._conclude(self.repair.report())

        raise AgentError(
            f"连续 {self.max_steps} 轮都在调用工具，仍未给出最终回答；"
            f"可以调大 max_steps，或检查工具是否总在报错。"
        )

    def _conclude(self, report: str) -> str:
        """修复停下时的收尾：报告也进历史，下一轮模型看得到当时停在哪。"""
        self.messages.append({"role": "assistant", "content": report})
        return report

    def _finish(self, message: dict[str, Any]) -> str:
        """修复过的话，最终回答前面加一句 Repair Summary。

        就一句话：修了几轮、每轮卡在哪一关、最后成没成。一轮都没修过
        （没触发自动修复的那种流程）时原样返回，一个字都不加。
        """
        answer = message.get("content") or ""
        summary = self.repair.summary()
        if not summary:
            return answer

        text = f"{summary}\n\n{answer}"
        # 历史里留的也是加上摘要的这一份：下一轮模型看到的，和用户当时看到的一致
        message["content"] = text
        return text

    def _run_tool_call(
        self,
        call: dict[str, Any],
        cancellation: CancellationContext | None = None,
    ) -> dict[str, Any]:
        """执行模型要求的一次工具调用，返回要回灌的 tool 消息。

        工具不存在、参数不合法、工具自己报错，都不往外抛，而是把错误文本
        当成工具结果回灌给模型，让它自己决定下一步（换参数重试或直接回答）。

        取消检查只放在这一步的边界上（执行前 / 执行后、验证前后）：**已经
        开跑的工具让它跑完**（单工具自己的 timeout 说了算），一返回就发现
        取消、后面的验证和修复一步都不再走——不硬杀子进程。
        """
        # 这一段是**不可信输入的边界**：tool_calls 是模型 / 网关给的，
        # 形状不完全由我们说了算。分两层处理：
        #
        # - 形状对不上「一条工具调用」的（不是对象、没有 function 对象）：
        #   连 call_id 都取不出来，造不出配得上 assistant 那条 tool_calls
        #   的 tool 回复，只能抛 AgentError 让整轮回滚——历史里留下
        #   「要了工具却没有回复」的结构，上游会拒掉之后**每一次**请求。
        # - 形态没问题、只有参数值不对的（JSON 坏了、解析出来不是对象）：
        #   照旧回一条错误结果让模型自己改，这一轮还能接着走。
        if not isinstance(call, dict):
            raise AgentError(
                f"模型返回的 tool_call 不是对象，收到 {type(call).__name__}，"
                f"这一轮已经回滚"
            )

        call_id = call.get("id", "")
        function = call.get("function")
        if not isinstance(function, dict):
            raise AgentError(
                f"tool_call 的 function 不是对象，收到 {type(function).__name__}，"
                f"这一轮已经回滚"
            )
        name = function.get("name", "")

        raw = function.get("arguments")
        if isinstance(raw, dict):
            # 规范要求 arguments 是 JSON 字符串，但有些 OpenAI 兼容网关
            # 直接把对象给过来。值本身是对的，收下就是——不收的话
            # json.loads(dict) 会抛 TypeError，而那是个**穿透异常**：
            # 它不在 run() 认得的那几种里，回滚不会发生。
            arguments: Any = raw
        else:
            try:
                arguments = json.loads(raw or "{}")
            except (TypeError, ValueError) as exc:
                # ValueError 含 JSONDecodeError；TypeError 是「根本不是
                # 字符串 / 字节」那一类（给了数字、列表……）
                return _tool_message(call_id, f"错误：参数不是合法 JSON（{exc}）")

        if not isinstance(arguments, dict):
            return _tool_message(
                call_id, f"错误：参数必须是 JSON 对象，收到 {type(arguments).__name__}"
            )

        # 交给执行层：它负责捕获异常和记录耗时，这里只关心结果文本
        _check_cancelled(cancellation)  # 工具执行前
        result = self.executor.execute(name, **arguments)

        # 工具一返回就查：取消后验证（它可能再跑编译 / 测试）一步都不走，
        # 结果也不再回灌给模型
        _check_cancelled(cancellation)

        # 改完代码接着验一次（该不该验、验完怎么说都在 Verifier 里，
        # 推理循环不认识 edit_file / compile 这些名字）；模型自己跑的
        # 编译 / 构建 / 测试则被记下来，当下一次比较的「之前」
        note = self.verifier.after_tool_call(name, arguments, result)

        # 验证和修复步骤之间再看一眼：验证刚跑过的编译 / 测试算完成了，
        # 但取消之后不该接着让修复循环决定「再修一轮」
        _check_cancelled(cancellation)

        # 再问修复循环一句这轮修得怎么样。它看的就是上面那次验证的结果，
        # 不另外跑工具；要不要接着修、要不要停，同样不在推理循环里判断
        repair = self.repair.after_tool_call(name, arguments, result)

        content = result.as_observation()
        if note:
            content = f"{content}\n\n{note}"
        if repair:
            content = f"{content}\n\n{repair}"

        return _tool_message(call_id, content)
