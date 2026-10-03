"""main.py 命令行入口测试：不管成功还是失败，Trace 都要打出来。

用假 Agent / 假客户端，不联网。
"""

import io
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from unittest.mock import patch

import main
from agent import Agent
from llm import LLMError
from tools import Tool, ToolError, ToolRegistry


class FailingTool(Tool):
    """永远失败的工具，用来制造「工具连续报错」的场景。"""

    name = "stub"
    description = "总是失败"
    parameters = {}

    def execute(self) -> str:
        raise ToolError("我不干")


class OkTool(Tool):
    """总是成功的工具。"""

    name = "stub"
    description = "总是成功"
    parameters = {}

    def execute(self) -> str:
        return "工具结果"


def tool_call_reply(call_id: str = "c1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "stub", "arguments": "{}"},
            }
        ],
    }


def final_reply(content: str = "好的") -> dict:
    return {"role": "assistant", "content": content}


class StubClient:
    """按剧本回话的假客户端；剧本里放异常，表示这一步要抛出来。"""

    class config:
        model = "stub-model"

    def __init__(self, script):
        self.script = list(script)

    def _next(self):
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def chat(self, messages, tools=None):
        return self._next()

    def chat_with_tools(self, messages, tools):
        return self._next()


class MainTestBase(unittest.TestCase):
    def setUp(self):
        self.out = io.StringIO()
        self.err = io.StringIO()

    def make_agent(self, script, tools, max_steps=2):
        return Agent(
            client=StubClient(script),
            tools=ToolRegistry(tools),
            max_steps=max_steps,
        )

    def enter(self, agent) -> ExitStack:
        """进入 main() 的运行环境：拦住 build_agent 和 input，把输出接管过来。"""
        stack = ExitStack()
        stack.enter_context(patch.object(main, "build_agent", return_value=agent))
        stack.enter_context(patch("builtins.input", side_effect=["试试", "exit"]))
        stack.enter_context(redirect_stdout(self.out))
        stack.enter_context(redirect_stderr(self.err))
        return stack

    def run_main(self, agent) -> int:
        with self.enter(agent):
            return main.main()


class MainTraceOnFailureTest(MainTestBase):
    """这几条是本阶段要修的核心：失败时反而最需要看 Trace。"""

    def test_trace_is_printed_when_the_agent_hits_max_steps(self):
        agent = self.make_agent(
            [tool_call_reply()] * 2, [FailingTool()], max_steps=2
        )

        self.assertEqual(self.run_main(agent), 0)

        self.assertIn("[Agent 失败]", self.err.getvalue())
        self.assertIn("[Tool] stub", self.out.getvalue())

    def test_the_failed_call_is_shown_as_failed(self):
        agent = self.make_agent(
            [tool_call_reply()] * 2, [FailingTool()], max_steps=2
        )

        self.run_main(agent)

        self.assertIn("success=false", self.out.getvalue())
        self.assertIn("我不干", self.out.getvalue())

    def test_every_failed_attempt_is_shown(self):
        # 连续两轮都失败 → 两条记录。只留最后一条就看不出「一直在重试」
        agent = self.make_agent(
            [tool_call_reply()] * 2, [FailingTool()], max_steps=2
        )

        self.run_main(agent)

        self.assertEqual(self.out.getvalue().count("[Tool] stub"), 2)

    def test_trace_is_printed_when_the_llm_call_fails(self):
        agent = self.make_agent(
            [tool_call_reply(), LLMError("接口超时")], [OkTool()], max_steps=2
        )

        self.run_main(agent)

        self.assertIn("[调用失败]", self.err.getvalue())
        # 报错之前那次成功的工具调用仍然留在记录里
        self.assertIn("[Tool] stub", self.out.getvalue())

    def test_trace_is_printed_on_an_unexpected_error(self):
        agent = self.make_agent(
            [tool_call_reply(), ValueError("说不清")], [OkTool()], max_steps=2
        )

        with self.enter(agent), self.assertRaises(ValueError):
            main.main()

        self.assertIn("[Tool] stub", self.out.getvalue())


class MainTraceOnSuccessTest(MainTestBase):
    def test_trace_is_printed(self):
        agent = self.make_agent(
            [tool_call_reply(), final_reply("好的")], [OkTool()], max_steps=2
        )

        self.assertEqual(self.run_main(agent), 0)

        self.assertIn("[Tool] stub", self.out.getvalue())
        self.assertIn("最终回答", self.out.getvalue())

    def test_trace_still_comes_before_the_answer(self):
        agent = self.make_agent(
            [tool_call_reply(), final_reply("好的")], [OkTool()], max_steps=2
        )

        self.run_main(agent)

        out = self.out.getvalue()
        self.assertLess(out.index("[Tool] stub"), out.index("最终回答"))

    def test_no_tool_call_is_reported_too(self):
        agent = self.make_agent([final_reply("你好")], [], max_steps=2)

        self.run_main(agent)

        self.assertIn("本轮没有调用工具", self.out.getvalue())

    def test_exit_stops_the_loop(self):
        agent = self.make_agent([final_reply("你好")], [], max_steps=2)

        self.assertEqual(self.run_main(agent), 0)


if __name__ == "__main__":
    unittest.main()
