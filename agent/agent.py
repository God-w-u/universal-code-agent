"""Agent：维护会话消息历史，并在需要时调用工具，直到拿到最终回答。

run() 一次调用就是一个 tool-calling loop：
    用户输入 -> LLM -> 要求调用工具就执行 -> 结果回灌 -> LLM -> ... -> 最终回答
"""

import json
from typing import Any

from agent.context import ContextManager
from agent.history import RepairHistory
from agent.prompt import DEFAULT_SYSTEM_PROMPT
from agent.repair import REPORT, NUDGE, DEFAULT_MAX_REPAIR_ROUNDS, RepairLoop
from agent.trace import Trace, TraceEvent
from agent.verify import Verifier
from llm import LLMClient, LLMError
from tools import ToolExecutor, ToolRegistry

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
        # messages 是完整历史，context 决定每次请求实际发多少出去
        self.context = context if context is not None else ContextManager()
        self.max_steps = max_steps
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        self._start_trace()

    def _start_trace(self) -> None:
        """换一个全新的 Trace 给这一轮用，上一轮的记录原样留在原对象里。

        用「换新对象」而不是「清空旧对象」：外部如果还拿着上一轮拿到的列表，
        不会被这一轮悄悄改掉。
        """
        self.trace = Trace()
        self.executor.trace = self.trace

    def get_trace(self) -> list[TraceEvent]:
        """返回当前这一轮的工具调用记录（和 messages 一样，拿到的是本体）。"""
        return self.trace.events

    def run(self, user_input: str) -> str:
        """处理一轮用户输入，返回最终回答。

        中途失败就把这一轮产生的所有消息（含工具调用）一起回滚，
        保证 messages 里始终是完整的对话，可以直接重试。

        正常跑完的话，装了 history 就顺手记一条（agent/history.py）。
        回滚掉的那一轮没有「结果」可记，不记。
        """
        # 这一轮从零开始记，上一轮的记录不混进来
        self._start_trace()

        user_input = user_input.strip()
        if not user_input:
            raise ValueError("用户输入不能为空")

        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            answer = self._run_loop()
        except (LLMError, AgentError):
            del self.messages[start:]
            raise

        # 记一笔修复过程。record() 自己把失败咽下去（原因在 last_error 里），
        # 所以这里不用 try——写历史绝不能把这一轮的回答变成异常
        if self.history is not None:
            self.history.record(user_input, self.repair)
        return answer

    def _run_loop(self) -> str:
        """反复「问模型 -> 执行工具」，直到模型给出不需要工具的回答。

        验证没过的那些轮里，模型想收尾也会被按回去接着修（见 RepairLoop）：
        decision 就是它给的说法——接着问、催一句、还是就此停下报状态。
        """
        schemas = [tool.to_schema() for tool in self.tools.list()]

        if not schemas:
            # 没有注册任何工具，退化成普通一问一答。
            answer = self.client.chat(self.context.build(self.messages))
            self.messages.append({"role": "assistant", "content": answer})
            return answer

        for _ in range(self.max_steps):
            # self.messages 是完整历史；发给模型的这份是裁剪过的，发完就丢
            message = self.client.chat_with_tools(
                self.context.build(self.messages), schemas
            )
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                decision = self.repair.review(message.get("content") or "")
                if decision.action == NUDGE:
                    # 还欠着一次没修好的，把模型按回去接着修
                    self.messages.append({"role": "user", "content": decision.text})
                    continue
                if decision.action == REPORT:
                    return self._conclude(decision.text)
                return self._finish(message)

            for call in tool_calls:
                self.messages.append(self._run_tool_call(call))

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

    def _run_tool_call(self, call: dict[str, Any]) -> dict[str, Any]:
        """执行模型要求的一次工具调用，返回要回灌的 tool 消息。

        工具不存在、参数不合法、工具自己报错，都不往外抛，而是把错误文本
        当成工具结果回灌给模型，让它自己决定下一步（换参数重试或直接回答）。
        """
        call_id = call.get("id", "")
        function = call.get("function") or {}
        name = function.get("name", "")

        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            return _tool_message(call_id, f"错误：参数不是合法 JSON（{exc}）")

        if not isinstance(arguments, dict):
            return _tool_message(
                call_id, f"错误：参数必须是 JSON 对象，收到 {type(arguments).__name__}"
            )

        # 交给执行层：它负责捕获异常和记录耗时，这里只关心结果文本
        result = self.executor.execute(name, **arguments)

        # 改完代码接着验一次（该不该验、验完怎么说都在 Verifier 里，
        # 推理循环不认识 edit_file / compile 这些名字）；模型自己跑的
        # 编译 / 构建 / 测试则被记下来，当下一次比较的「之前」
        note = self.verifier.after_tool_call(name, arguments, result)

        # 再问修复循环一句这轮修得怎么样。它看的就是上面那次验证的结果，
        # 不另外跑工具；要不要接着修、要不要停，同样不在推理循环里判断
        repair = self.repair.after_tool_call(name, arguments, result)

        content = result.as_observation()
        if note:
            content = f"{content}\n\n{note}"
        if repair:
            content = f"{content}\n\n{repair}"

        return _tool_message(call_id, content)
