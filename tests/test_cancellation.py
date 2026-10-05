"""取消与整轮超时（13.4）：开关本身、Agent 的检查点、回滚、不跨轮泄漏。

全是 Agent 层，不起 HTTP：CancellationContext 的线程安全与幂等、Agent.run
在各个检查点上停得下来、取消之后不再调模型 / 不再跑工具、下一轮不继承
取消状态、单工具超时不等于整轮取消。HTTP 那层（Run 登记表、取消接口、
SSE 事件）在 test_api_runs.py。

时间用假时钟（test_sessions.FakeClock）：超时不真等，拨一下就到点。
"""

import threading
import unittest

from agent import Agent, CancellationContext, RunCancelled
from agent.cancel import CANCELLED, TIMEOUT
from tests.test_agent import FakeLLMClient, tool_call_reply
from tests.test_sessions import FakeClock
from tools import Tool, ToolRegistry


class ProbeTool(Tool):
    """探针工具：跑一次记一笔；可以在执行期间按掉取消开关 / 拨快时钟。"""

    name = "probe"
    description = "探针"
    parameters = {}

    def __init__(self, cancellation=None, clock=None, advance: float = 0.0) -> None:
        self.cancellation = cancellation
        self.clock = clock
        self.advance = advance
        self.calls = 0

    def execute(self) -> str:
        self.calls += 1
        if self.cancellation is not None:
            self.cancellation.cancel()
        if self.clock is not None:
            self.clock.advance(self.advance)
        return "探针结果"


class TimeoutTool(Tool):
    """模拟单工具自己超时：返回 run_command 那种「结果：超时，已强制终止」。"""

    name = "slow_command"
    description = "慢命令"
    parameters = {}

    def execute(self) -> str:
        return "命令：慢命令\n结果：超时，已强制终止"


class CancellingClient(FakeLLMClient):
    """模型一答复就把取消开关按掉：用来测「LLM 调用后」这个检查点。"""

    def __init__(self, replies, cancellation) -> None:
        super().__init__(replies)
        self.cancellation = cancellation

    def chat_with_tools(self, messages, tools):
        reply = super().chat_with_tools(messages, tools)
        self.cancellation.cancel()
        return reply


# ---------------------------------------------------------------------------
# 开关本身
# ---------------------------------------------------------------------------


class CancellationTokenTest(unittest.TestCase):
    def test_a_fresh_token_is_not_cancelled(self):
        token = CancellationContext()

        self.assertFalse(token.is_cancelled())
        self.assertIsNone(token.reason)
        token.raise_if_cancelled()  # 没取消：什么都不做，不抛

    def test_cancel_takes_effect_once_and_keeps_the_first_reason(self):
        token = CancellationContext()

        self.assertTrue(token.cancel())
        self.assertFalse(token.cancel())  # 再来一次：幂等
        self.assertFalse(token.cancel(TIMEOUT))  # 原因也不许改

        self.assertTrue(token.is_cancelled())
        self.assertEqual(token.reason, CANCELLED)

    def test_raise_if_cancelled_raises_with_the_reason(self):
        token = CancellationContext()
        token.cancel()

        with self.assertRaises(RunCancelled) as caught:
            token.raise_if_cancelled()

        self.assertEqual(caught.exception.reason, CANCELLED)
        self.assertIn("取消", str(caught.exception))

    def test_the_timeout_comes_from_the_clock(self):
        clock = FakeClock(0.0)
        token = CancellationContext(timeout=10.0, clock=clock)

        self.assertEqual(token.deadline, 10.0)
        clock.advance(9.9)
        self.assertFalse(token.is_cancelled())

        clock.advance(0.1)  # 正好到点：算超时
        self.assertTrue(token.is_cancelled())
        self.assertEqual(token.reason, TIMEOUT)
        with self.assertRaises(RunCancelled) as caught:
            token.raise_if_cancelled()
        self.assertEqual(caught.exception.reason, TIMEOUT)

    def test_a_manual_cancel_beats_the_timeout(self):
        clock = FakeClock()
        token = CancellationContext(timeout=5.0, clock=clock)
        clock.advance(6.0)

        token.cancel()  # 到点之后才按的停

        self.assertEqual(token.reason, CANCELLED)  # 先到的原因算数

    def test_no_timeout_means_it_never_expires(self):
        clock = FakeClock()
        token = CancellationContext(clock=clock)
        clock.advance(10**9)

        self.assertIsNone(token.deadline)
        self.assertFalse(token.is_cancelled())

    def test_a_non_positive_timeout_is_rejected(self):
        for timeout in (0, 0.0, -1.0):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    CancellationContext(timeout=timeout)

    def test_a_blank_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            CancellationContext().cancel("")

    def test_only_one_thread_wins_the_first_cancel(self):
        token = CancellationContext()
        results: list[bool] = []
        bookkeeping = threading.Lock()
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait(5)
            won = token.cancel()
            with bookkeeping:
                results.append(won)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        self.assertEqual(results.count(True), 1)  # 只有一枪是真的
        self.assertTrue(token.is_cancelled())


# ---------------------------------------------------------------------------
# Agent 的检查点
# ---------------------------------------------------------------------------


class AgentCancellationTest(unittest.TestCase):
    """真 Agent + 脚本化客户端 + 假工具：检查点真的能把这一轮停下来。"""

    def test_a_token_cancelled_before_the_run_touches_nothing(self):
        client = FakeLLMClient(["这一句不该被用到"])
        agent = Agent(client=client)
        token = CancellationContext()
        token.cancel()

        with self.assertRaises(RunCancelled):
            agent.run("任务甲", cancellation=token)

        self.assertEqual(client.calls, [])  # 模型没被叫
        self.assertEqual(len(agent.messages), 1)  # 历史没动过

    def test_a_tool_can_stop_the_run_and_the_llm_is_not_called_again(self):
        token = CancellationContext()
        tool = ProbeTool(cancellation=token)
        client = FakeLLMClient([tool_call_reply(("probe", "{}")), "这一句不该被用到"])
        agent = Agent(client=client, tools=ToolRegistry([tool]))

        with self.assertRaises(RunCancelled) as caught:
            agent.run("任务甲", cancellation=token)

        self.assertEqual(caught.exception.reason, CANCELLED)
        self.assertEqual(tool.calls, 1)  # 已经开跑的工具让它跑完
        self.assertEqual(len(client.calls), 1)  # 之后一次模型都没再叫
        self.assertEqual(len(agent.messages), 1)  # 这一轮的消息整体回滚

    def test_a_cancel_after_the_llm_stops_before_any_tool_runs(self):
        token = CancellationContext()
        tool = ProbeTool()
        client = CancellingClient([tool_call_reply(("probe", "{}"))], token)
        agent = Agent(client=client, tools=ToolRegistry([tool]))

        with self.assertRaises(RunCancelled):
            agent.run("任务甲", cancellation=token)

        self.assertEqual(tool.calls, 0)  # 模型要求的工具一个都没执行
        self.assertEqual(len(client.calls), 1)

    def test_a_run_timeout_stops_at_the_next_checkpoint(self):
        clock = FakeClock()
        token = CancellationContext(timeout=10.0, clock=clock)
        tool = ProbeTool(clock=clock, advance=11.0)  # 工具把整轮预算用完了
        client = FakeLLMClient([tool_call_reply(("probe", "{}")), "来不及用的回答"])
        agent = Agent(client=client, tools=ToolRegistry([tool]))

        with self.assertRaises(RunCancelled) as caught:
            agent.run("任务甲", cancellation=token)

        self.assertEqual(caught.exception.reason, TIMEOUT)
        self.assertEqual(tool.calls, 1)
        self.assertEqual(len(client.calls), 1)

    def test_the_next_run_does_not_inherit_the_cancellation(self):
        token = CancellationContext()
        tool = ProbeTool(cancellation=token)
        client = FakeLLMClient([tool_call_reply(("probe", "{}")), "第二次跑完了。"])
        agent = Agent(client=client, tools=ToolRegistry([tool]))

        with self.assertRaises(RunCancelled):
            agent.run("任务甲", cancellation=token)

        answer = agent.run("任务乙")  # 不传取消开关：旧行为，一个检查点都不走

        self.assertEqual(answer, "第二次跑完了。")
        self.assertEqual(agent.messages[1]["content"], "任务乙")
        self.assertEqual(agent.messages[2]["content"], "第二次跑完了。")
        self.assertEqual(tool.calls, 1)  # 第二次没再碰工具

    def test_no_cancellation_means_the_old_behavior(self):
        tool = ProbeTool()
        client = FakeLLMClient([tool_call_reply(("probe", "{}")), "跑完了。"])
        agent = Agent(client=client, tools=ToolRegistry([tool]))

        answer = agent.run("任务甲")

        self.assertEqual(answer, "跑完了。")
        self.assertEqual(tool.calls, 1)

    def test_a_tool_timeout_is_not_a_run_cancellation(self):
        # 单工具自己超时只是工具结果的一部分，这一轮照常跑完、照常回答
        client = FakeLLMClient([tool_call_reply(("slow_command", "{}")), "换个做法。"])
        agent = Agent(client=client, tools=ToolRegistry([TimeoutTool()]))

        answer = agent.run("任务甲")  # 不抛任何异常

        self.assertEqual(answer, "换个做法。")
        observations = [m["content"] for m in agent.messages if m["role"] == "tool"]
        self.assertTrue(any("超时" in text for text in observations))
