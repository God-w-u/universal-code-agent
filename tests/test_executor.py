"""ToolExecutor 测试：成功执行、ToolError、意外异常、耗时、Agent 调用链。

全部离线跑，不联网。
"""

import dataclasses
import time
import unittest

from agent import Agent, Trace
from tools import (
    CalculatorTool,
    Tool,
    ToolError,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)


class EchoTool(Tool):
    """正常工具：原样回显。"""

    name = "echo"
    description = "回显输入"
    parameters = {
        "text": {"type": "string", "description": "要回显的文本", "required": True},
    }

    def execute(self, text: str) -> str:
        return f"echo: {text}"


class SlowTool(Tool):
    """故意慢一点的工具，用来验证耗时确实被测到了。"""

    name = "slow"
    description = "睡一会儿"
    parameters = {}

    def execute(self) -> str:
        time.sleep(0.02)
        return "done"


class AngryTool(Tool):
    """按预期报错：主动抛 ToolError。"""

    name = "angry"
    description = "总是拒绝执行"
    parameters = {}

    def execute(self) -> str:
        raise ToolError("我不干")


class BoomError(Exception):
    """工具实现里自定义的异常。"""


class BrokenTool(Tool):
    """有 bug 的工具：抛意外异常，而不是 ToolError。"""

    name = "broken"
    description = "内部有除零 bug"
    parameters = {}

    def execute(self) -> str:
        return 1 / 0


class Text(str):
    """str 的子类，用来确认返回类型检查用的是 isinstance 而不是 type() is。"""


class ForgetfulTool(Tool):
    """忘了写 return 的工具：execute() 返回 None。"""

    name = "forgetful"
    description = "没有返回值"
    parameters = {}

    def execute(self):
        """故意不 return。"""


class WrongTypeTool(Tool):
    """返回数字的工具。"""

    name = "wrongtype"
    description = "返回数字"
    parameters = {}

    def execute(self) -> int:
        return 42


class SubclassTool(Tool):
    """返回 str 子类的工具，属于合法返回值。"""

    name = "subclass"
    description = "返回 str 子类"
    parameters = {}

    def execute(self) -> str:
        return Text("子类文本")


class ExplodingTool(Tool):
    """抛自定义异常的工具。"""

    name = "exploding"
    description = "抛自定义异常"
    parameters = {}

    def execute(self) -> str:
        raise BoomError("炸了")


class FakeLLMClient:
    """按剧本回复的假客户端：轮到哪里就返回哪一条。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


def tool_call_reply(name: str, arguments: str = "{}", call_id: str = "call_1"):
    """造一条「模型要求调用工具」的回复。"""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def final_reply(content: str = "好的"):
    return {"role": "assistant", "content": content}


def build_agent(replies, tools):
    """装配一个用假客户端的 Agent，并跑一轮，返回 (agent, 回答)。"""
    agent = Agent(client=FakeLLMClient(replies), tools=ToolRegistry(tools))
    answer = agent.run("试试")
    return agent, answer


class ToolExecutorTestBase(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(
            [EchoTool(), SlowTool(), AngryTool(), BrokenTool(), ExplodingTool()]
        )
        self.executor = ToolExecutor(self.registry)


class ToolExecutorSuccessTest(ToolExecutorTestBase):
    def test_successful_call_returns_content(self):
        result = self.executor.execute("echo", text="hi")

        self.assertTrue(result.success)
        self.assertEqual(result.content, "echo: hi")
        self.assertEqual(result.error, "")

    def test_arguments_are_passed_through(self):
        self.assertEqual(
            self.executor.execute("echo", text="a b").content, "echo: a b"
        )

    def test_metadata_carries_tool_name(self):
        result = self.executor.execute("echo", text="hi")

        self.assertEqual(result.metadata["tool"], "echo")

    def test_observation_is_the_raw_content_on_success(self):
        result = self.executor.execute("echo", text="hi")

        self.assertEqual(result.as_observation(), "echo: hi")

    def test_result_is_frozen(self):
        result = self.executor.execute("echo", text="hi")

        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False

    def test_default_result_is_empty_and_successful(self):
        result = ToolResult(success=True)

        self.assertEqual(result.content, "")
        self.assertEqual(result.error, "")
        self.assertEqual(result.metadata, {})


class ToolExecutorToolErrorTest(ToolExecutorTestBase):
    """预期内的失败：都该变成 success=False，而不是抛出去。"""

    def test_unknown_tool_name(self):
        result = self.executor.execute("nope")

        self.assertFalse(result.success)
        self.assertIn("未注册的工具", result.error)
        self.assertEqual(result.content, "")

    def test_tool_raising_tool_error(self):
        result = self.executor.execute("angry")

        self.assertFalse(result.success)
        self.assertEqual(result.error, "我不干")

    def test_missing_argument(self):
        result = self.executor.execute("echo")

        self.assertFalse(result.success)
        self.assertIn("text", result.error)

    def test_wrong_argument_type(self):
        self.assertFalse(self.executor.execute("echo", text=123).success)

    def test_unexpected_argument(self):
        self.assertFalse(self.executor.execute("echo", text="hi", extra=1).success)

    def test_observation_prefixes_the_error(self):
        result = self.executor.execute("angry")

        self.assertEqual(result.as_observation(), "错误：我不干")

    def test_metadata_still_carries_the_tool_name(self):
        result = self.executor.execute("angry")

        self.assertEqual(result.metadata["tool"], "angry")

    def test_nothing_is_raised(self):
        for name in ("nope", "angry"):
            with self.subTest(name=name):
                self.executor.execute(name)


class ToolExecutorUnexpectedErrorTest(ToolExecutorTestBase):
    """工具自己有 bug 时也必须兜住，否则一个坏工具拖垮整个 Agent。"""

    def test_broken_tool_does_not_propagate(self):
        result = self.executor.execute("broken")

        self.assertFalse(result.success)

    def test_error_names_the_exception_type(self):
        result = self.executor.execute("broken")

        self.assertIn("ZeroDivisionError", result.error)

    def test_custom_exception_is_also_caught(self):
        result = self.executor.execute("exploding")

        self.assertFalse(result.success)
        self.assertIn("BoomError", result.error)
        self.assertIn("炸了", result.error)

    def test_unexpected_failure_looks_like_any_other_failure(self):
        result = self.executor.execute("broken")

        self.assertTrue(result.as_observation().startswith("错误："))

    def test_metadata_is_recorded_for_unexpected_failures_too(self):
        result = self.executor.execute("broken")

        self.assertEqual(result.metadata["tool"], "broken")
        self.assertIn("duration_seconds", result.metadata)


class ToolExecutorDurationTest(ToolExecutorTestBase):
    def test_duration_is_recorded_on_success(self):
        result = self.executor.execute("echo", text="hi")

        duration = result.metadata["duration_seconds"]
        self.assertIsInstance(duration, float)
        self.assertGreaterEqual(duration, 0.0)

    def test_duration_measures_actual_work(self):
        result = self.executor.execute("slow")

        self.assertGreaterEqual(result.metadata["duration_seconds"], 0.02)

    def test_duration_is_recorded_on_tool_error(self):
        result = self.executor.execute("angry")

        self.assertGreaterEqual(result.metadata["duration_seconds"], 0.0)

    def test_every_result_carries_both_metadata_keys(self):
        calls = [("echo", {"text": "hi"}), ("angry", {}), ("nope", {})]

        for name, arguments in calls:
            with self.subTest(name=name):
                metadata = self.executor.execute(name, **arguments).metadata
                self.assertEqual(metadata["tool"], name)
                self.assertIn("duration_seconds", metadata)


class ToolExecutorRegistryTest(ToolExecutorTestBase):
    def test_exposes_the_registry_it_wraps(self):
        self.assertIs(self.executor.registry, self.registry)

    def test_empty_registry_reports_failure_instead_of_raising(self):
        result = ToolExecutor(ToolRegistry()).execute("anything")

        self.assertFalse(result.success)

    def test_it_runs_the_real_calculator(self):
        executor = ToolExecutor(ToolRegistry([CalculatorTool()]))

        result = executor.execute("calculator", expression="3*7")

        self.assertTrue(result.success)
        self.assertIn("21", result.content)


class AgentUsesExecutorTest(unittest.TestCase):
    """Agent 必须走执行层，而不是自己 try/except 去调 ToolRegistry。"""

    def test_agent_has_an_executor_bound_to_its_registry(self):
        agent = Agent(client=FakeLLMClient(), tools=ToolRegistry([EchoTool()]))

        self.assertIsInstance(agent.executor, ToolExecutor)
        self.assertIs(agent.executor.registry, agent.tools)

    def test_default_registry_gets_an_executor_too(self):
        agent = Agent(client=FakeLLMClient())

        self.assertIs(agent.executor.registry, agent.tools)

    def test_successful_tool_result_reaches_the_model(self):
        agent, answer = build_agent(
            [tool_call_reply("echo", '{"text": "hi"}'), final_reply()],
            [EchoTool()],
        )

        self.assertEqual(agent.messages[3]["content"], "echo: hi")
        self.assertEqual(answer, "好的")

    def test_tool_error_feedback_text_is_unchanged(self):
        agent, _ = build_agent(
            [tool_call_reply("angry"), final_reply()], [AngryTool()]
        )

        self.assertEqual(agent.messages[3]["content"], "错误：我不干")

    def test_unexpected_exception_no_longer_crashes_the_turn(self):
        # 走执行层之前，这里会直接把 ZeroDivisionError 抛给调用方
        agent, answer = build_agent(
            [tool_call_reply("broken"), final_reply()], [BrokenTool()]
        )

        self.assertEqual(answer, "好的")
        self.assertIn("ZeroDivisionError", agent.messages[3]["content"])

    def test_unknown_tool_is_still_fed_back(self):
        agent, answer = build_agent(
            [tool_call_reply("nope"), final_reply()], [EchoTool()]
        )

        self.assertIn("未注册的工具", agent.messages[3]["content"])
        self.assertEqual(answer, "好的")

    def test_tool_message_keeps_the_call_id(self):
        agent, _ = build_agent(
            [tool_call_reply("echo", '{"text": "hi"}', call_id="call_9"), final_reply()],
            [EchoTool()],
        )

        self.assertEqual(agent.messages[3]["tool_call_id"], "call_9")


class ToolExecutorReturnTypeTest(unittest.TestCase):
    """工具没返回 str 时要拦成失败结果，而不是让 None 一路发到接口那边。"""

    def setUp(self):
        self.registry = ToolRegistry(
            [EchoTool(), ForgetfulTool(), WrongTypeTool(), SubclassTool()]
        )
        self.executor = ToolExecutor(self.registry)

    def test_none_return_is_a_failure(self):
        result = self.executor.execute("forgetful")

        self.assertFalse(result.success)
        self.assertIn("必须返回 str", result.error)
        self.assertIn("NoneType", result.error)

    def test_number_return_is_a_failure(self):
        result = self.executor.execute("wrongtype")

        self.assertFalse(result.success)
        self.assertIn("int", result.error)

    def test_bad_content_is_not_kept(self):
        # content 必须留空：带着 None 回灌给模型，请求体里就会冒出个 null
        result = self.executor.execute("forgetful")

        self.assertEqual(result.content, "")

    def test_observation_is_error_text(self):
        result = self.executor.execute("wrongtype")

        self.assertTrue(result.as_observation().startswith("错误："))

    def test_it_does_not_raise(self):
        # 明确钉住：是「拦住」不是「炸掉」
        self.assertFalse(self.executor.execute("forgetful").success)
        self.assertFalse(self.executor.execute("wrongtype").success)

    def test_error_names_the_tool(self):
        self.assertIn("wrongtype", self.executor.execute("wrongtype").error)

    def test_metadata_is_still_recorded(self):
        result = self.executor.execute("wrongtype")

        self.assertEqual(result.metadata["tool"], "wrongtype")
        self.assertIn("duration_seconds", result.metadata)

    def test_str_subclass_is_accepted(self):
        result = self.executor.execute("subclass")

        self.assertTrue(result.success)
        self.assertEqual(result.content, "子类文本")

    def test_a_normal_tool_is_unaffected(self):
        self.assertTrue(self.executor.execute("echo", text="hi").success)

    def test_the_failure_is_recorded_in_the_trace(self):
        trace = Trace()
        executor = ToolExecutor(self.registry, trace=trace)

        executor.execute("forgetful")

        self.assertEqual(len(trace), 1)
        self.assertFalse(trace.events[0].success)
        self.assertIn("必须返回 str", trace.events[0].error)

    def test_the_model_gets_an_error_instead_of_crashing(self):
        agent, answer = build_agent(
            [tool_call_reply("wrongtype"), final_reply()], [WrongTypeTool()]
        )

        self.assertEqual(answer, "好的")
        self.assertIn("必须返回 str", agent.messages[3]["content"])


if __name__ == "__main__":
    unittest.main()
