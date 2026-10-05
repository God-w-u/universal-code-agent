"""Trace 测试：事件结构、执行层记录、Agent 获取接口、多轮隔离。

全部离线跑，不联网。
"""

import dataclasses
import json
import time
import unittest

from agent import Agent, AgentError, Trace, TraceEvent
from llm import LLMCallStats, LLMUsage
from tools import Tool, ToolError, ToolExecutor, ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "回显输入"
    parameters = {
        "text": {"type": "string", "description": "要回显的文本", "required": True},
    }

    def execute(self, text: str) -> str:
        return f"echo: {text}"


class SlowTool(Tool):
    name = "slow"
    description = "睡一会儿"
    parameters = {}

    def execute(self) -> str:
        time.sleep(0.02)
        return "done"


class AngryTool(Tool):
    name = "angry"
    description = "总是拒绝执行"
    parameters = {}

    def execute(self) -> str:
        raise ToolError("我不干")


class BrokenTool(Tool):
    name = "broken"
    description = "有 bug：除零"
    parameters = {}

    def execute(self) -> str:
        return 1 / 0


class FakeLLMClient:
    """按剧本回复的假客户端。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])

    def chat(self, messages, tools=None):
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


def tool_call_reply(name: str, arguments: str = "{}", call_id: str = "call_1"):
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


def two_calls_reply(first: tuple, second: tuple):
    """一条消息里要求调用两个工具。"""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for index, (name, arguments) in enumerate((first, second))
        ],
    }


def final_reply(content: str = "好的"):
    return {"role": "assistant", "content": content}


def build_agent(replies, tools):
    agent = Agent(client=FakeLLMClient(replies), tools=ToolRegistry(tools))
    return agent, agent.run("试试")


class TraceEventTest(unittest.TestCase):
    def make(self, **overrides) -> TraceEvent:
        fields = {
            "type": "tool_call",
            "tool": "read_file",
            "arguments": {"path": "main.py"},
            "success": True,
            "error": "",
            "duration_seconds": 0.12,
        }
        return TraceEvent(**{**fields, **overrides})

    def test_fields(self):
        event = self.make()

        self.assertEqual(event.type, "tool_call")
        self.assertEqual(event.tool, "read_file")
        self.assertEqual(event.arguments, {"path": "main.py"})
        self.assertTrue(event.success)
        self.assertEqual(event.error, "")
        self.assertEqual(event.duration_seconds, 0.12)

    def test_to_dict_shape(self):
        self.assertEqual(
            self.make().to_dict(),
            {
                "type": "tool_call",
                "tool": "read_file",
                "arguments": {"path": "main.py"},
                "success": True,
                "error": "",
                "duration_seconds": 0.12,
            },
        )

    def test_to_dict_always_carries_the_same_keys(self):
        # 成功时 error 也在，读取方不用先判断键在不在
        self.assertEqual(set(self.make().to_dict()), set(self.make(success=False).to_dict()))

    def test_to_dict_is_json_serialisable(self):
        json.dumps(self.make().to_dict())

    def test_defaults(self):
        event = TraceEvent(type="tool_call", tool="x")

        self.assertEqual(event.arguments, {})
        self.assertEqual(event.error, "")
        self.assertEqual(event.duration_seconds, 0.0)
        self.assertFalse(event.success)

    def test_arguments_default_is_not_shared_between_instances(self):
        first = TraceEvent(type="tool_call", tool="a")
        second = TraceEvent(type="tool_call", tool="b")

        first.arguments["x"] = 1

        self.assertEqual(second.arguments, {})

    def test_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.make().success = False


class TraceTest(unittest.TestCase):
    def test_starts_empty(self):
        trace = Trace()

        self.assertEqual(len(trace), 0)
        self.assertEqual(trace.events, [])

    def test_record_appends_and_returns_the_event(self):
        trace = Trace()

        event = trace.record_tool_call(
            tool="echo", arguments={"text": "hi"}, success=True, duration_seconds=0.5
        )

        self.assertEqual(len(trace), 1)
        self.assertIs(trace.events[0], event)
        self.assertEqual(event.type, "tool_call")

    def test_iteration_yields_events_in_order(self):
        trace = Trace()
        trace.record_tool_call(tool="a", arguments={}, success=True)
        trace.record_tool_call(tool="b", arguments={}, success=False)

        self.assertEqual([event.tool for event in trace], ["a", "b"])

    def test_to_dicts(self):
        trace = Trace()
        trace.record_tool_call(tool="a", arguments={"n": 1}, success=True)

        self.assertEqual(
            trace.to_dicts(),
            [
                {
                    "type": "tool_call",
                    "tool": "a",
                    "arguments": {"n": 1},
                    "success": True,
                    "error": "",
                    "duration_seconds": 0.0,
                }
            ],
        )

    def test_arguments_are_snapshotted(self):
        # 记的是调用当时的样子，事后改原字典不该影响记录
        original = {"text": "hi"}
        trace = Trace()
        trace.record_tool_call(tool="echo", arguments=original, success=True)

        original["text"] = "改掉了"

        self.assertEqual(trace.events[0].arguments, {"text": "hi"})


class ExecutorTraceTest(unittest.TestCase):
    def make_executor(self, trace=None, *tools):
        return ToolExecutor(ToolRegistry(list(tools)), trace=trace)

    def test_no_trace_means_nothing_is_recorded(self):
        executor = self.make_executor(None, EchoTool())

        executor.execute("echo", text="hi")

        self.assertIsNone(executor.trace)

    def test_successful_call_is_recorded(self):
        trace = Trace()
        executor = self.make_executor(trace, EchoTool())

        executor.execute("echo", text="hi")

        self.assertEqual(len(trace), 1)
        event = trace.events[0]
        self.assertEqual(event.type, "tool_call")
        self.assertEqual(event.tool, "echo")
        self.assertEqual(event.arguments, {"text": "hi"})
        self.assertTrue(event.success)
        self.assertEqual(event.error, "")

    def test_tool_error_is_recorded(self):
        trace = Trace()
        executor = self.make_executor(trace, AngryTool())

        executor.execute("angry")

        event = trace.events[0]
        self.assertFalse(event.success)
        self.assertEqual(event.error, "我不干")

    def test_unexpected_exception_is_recorded(self):
        trace = Trace()
        executor = self.make_executor(trace, BrokenTool())

        executor.execute("broken")

        event = trace.events[0]
        self.assertFalse(event.success)
        self.assertIn("ZeroDivisionError", event.error)

    def test_unknown_tool_is_recorded(self):
        trace = Trace()
        executor = self.make_executor(trace, EchoTool())

        executor.execute("nope")

        self.assertEqual(trace.events[0].tool, "nope")
        self.assertFalse(trace.events[0].success)

    def test_bad_arguments_are_recorded(self):
        trace = Trace()
        executor = self.make_executor(trace, EchoTool())

        executor.execute("echo", text="hi", extra=1)

        event = trace.events[0]
        self.assertFalse(event.success)
        self.assertEqual(event.arguments, {"text": "hi", "extra": 1})

    def test_duration_is_recorded(self):
        trace = Trace()
        executor = self.make_executor(trace, SlowTool())

        executor.execute("slow")

        self.assertGreaterEqual(trace.events[0].duration_seconds, 0.02)

    def test_duration_matches_the_result_metadata(self):
        trace = Trace()
        executor = self.make_executor(trace, SlowTool())

        result = executor.execute("slow")

        self.assertEqual(
            trace.events[0].duration_seconds, result.metadata["duration_seconds"]
        )

    def test_duration_is_recorded_for_failures_too(self):
        trace = Trace()
        executor = self.make_executor(trace, AngryTool())

        executor.execute("angry")

        self.assertGreaterEqual(trace.events[0].duration_seconds, 0.0)

    def test_every_call_adds_exactly_one_event(self):
        trace = Trace()
        executor = self.make_executor(trace, EchoTool(), AngryTool())

        executor.execute("echo", text="a")
        executor.execute("angry")
        executor.execute("nope")

        self.assertEqual(len(trace), 3)

    def test_tracing_does_not_change_the_result(self):
        plain = self.make_executor(None, EchoTool())
        traced = self.make_executor(Trace(), EchoTool())

        without = plain.execute("echo", text="hi")
        withtrace = traced.execute("echo", text="hi")

        self.assertEqual(without.success, withtrace.success)
        self.assertEqual(without.content, withtrace.content)
        self.assertEqual(without.error, withtrace.error)
        self.assertEqual(set(without.metadata), set(withtrace.metadata))

    def test_executor_accepts_a_trace_after_construction(self):
        executor = self.make_executor(None, EchoTool())
        executor.trace = Trace()

        executor.execute("echo", text="hi")

        self.assertEqual(len(executor.trace), 1)


class TraceObserverTest(unittest.TestCase):
    """13.2 的观察者：开始 / 结束各通知一次，都不影响执行和记录。"""

    def make(self, tools=(EchoTool(),), **observers):
        trace = Trace(**observers)
        return trace, ToolExecutor(ToolRegistry(list(tools)), trace=trace)

    def test_start_comes_before_end(self):
        order = []
        _, executor = self.make(
            on_tool_start=lambda tool, arguments: order.append(
                ("start", tool, arguments)
            ),
            on_tool_end=lambda event: order.append(("end", event.tool)),
        )

        executor.execute("echo", text="hi")

        self.assertEqual(order, [("start", "echo", {"text": "hi"}), ("end", "echo")])

    def test_the_start_notice_is_not_an_event(self):
        trace, executor = self.make(on_tool_start=lambda tool, arguments: None)

        executor.execute("echo", text="hi")

        # 开始通知不往 events 里塞半成品：那里只有「跑完了」的记录
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace.events[0].tool, "echo")

    def test_a_failed_tool_notifies_both_ends_too(self):
        order = []
        _, executor = self.make(
            tools=(AngryTool(),),
            on_tool_start=lambda tool, arguments: order.append("start"),
            on_tool_end=lambda event: order.append((event.success, event.error)),
        )

        executor.execute("angry")

        self.assertEqual(order, ["start", (False, "我不干")])

    def test_an_exploding_observer_does_not_break_the_call(self):
        def boom(*args):
            raise RuntimeError("观察者炸了")

        trace, executor = self.make(on_tool_start=boom, on_tool_end=boom)

        result = executor.execute("echo", text="hi")

        # 工具照跑、记录照记：观察者是外人，它的错不归执行层管
        self.assertTrue(result.success)
        self.assertEqual(len(trace), 1)

    def test_no_observer_means_no_notification_but_the_same_records(self):
        trace, executor = self.make()

        executor.execute("echo", text="hi")

        self.assertEqual(len(trace), 1)
        self.assertEqual(trace.events[0].arguments, {"text": "hi"})

    def test_observers_do_not_leak_into_the_next_run(self):
        seen = []
        agent = Agent(
            client=FakeLLMClient(
                [
                    tool_call_reply("echo", '{"text": "hi"}'),
                    final_reply("第一次"),
                    final_reply("第二次"),
                ]
            ),
            tools=ToolRegistry([EchoTool()]),
        )

        agent.run("任务甲", on_tool_start=lambda tool, arguments: seen.append(tool))
        agent.run("任务乙")

        # 第二轮没带观察者：上一轮的回调不该留在执行层继续发通知
        self.assertEqual(seen, ["echo"])
        self.assertEqual(len(agent.get_trace()), 0)


class LlmObserverTest(unittest.TestCase):
    """13.5 的观察者：模型调用的开始 / 结束各通知一次，都不进 events。

    这一层只认 LLMCallStats 的形状，统计对象直接手搓——真客户端怎么攒出
    它来在 test_llm_retry.py，这里钉的是 Trace 这一侧的收发。
    """

    def make_stats(self, **overrides):
        fields = {
            "model": "假模型",
            "success": True,
            "duration_seconds": 0.25,
            "total_duration_seconds": 0.25,
            "attempt_count": 1,
            "max_attempts": 3,
            "usage": LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            "error": "",
        }
        fields.update(overrides)
        return LLMCallStats(**fields)

    def test_start_comes_before_end(self):
        order = []
        trace = Trace(
            on_llm_start=lambda model, attempt, max_attempts: order.append(
                ("start", model, attempt, max_attempts)
            ),
            on_llm_end=lambda stats: order.append(("end", stats.attempt_count)),
        )

        trace.start_llm_call("假模型", 1, 3)
        trace.record_llm_call(self.make_stats())

        self.assertEqual(order, [("start", "假模型", 1, 3), ("end", 1)])

    def test_the_llm_notices_are_not_events(self):
        trace = Trace(on_llm_start=lambda *args: None)

        trace.start_llm_call("假模型", 1, 3)
        trace.record_llm_call(self.make_stats())

        # events 里始终只有「跑完的工具调用」：模型调用一条也不往里塞
        self.assertEqual(trace.events, [])
        self.assertEqual(len(trace), 0)
        self.assertEqual(trace.to_dicts(), [])

    def test_the_stats_are_kept_in_llm_calls(self):
        stats = self.make_stats()
        trace = Trace()

        trace.record_llm_call(stats)

        self.assertEqual(trace.llm_calls, [stats])
        # 没有观察者也有留档：回调只是「顺便通知一下」
        self.assertEqual(trace.llm_calls[0].usage.total_tokens, 15)

    def test_every_hop_of_a_retry_is_kept(self):
        first = self.make_stats(success=False, attempt_count=1, usage=None, error="限流了")
        second = self.make_stats(attempt_count=2)
        trace = Trace()

        trace.record_llm_call(first)
        trace.record_llm_call(second)

        self.assertEqual([stats.attempt_count for stats in trace.llm_calls], [1, 2])
        self.assertFalse(trace.llm_calls[0].success)
        self.assertTrue(trace.llm_calls[1].success)

    def test_an_exploding_llm_observer_does_not_break_the_records(self):
        def boom(*args):
            raise RuntimeError("观察者炸了")

        trace = Trace(on_llm_start=boom, on_llm_end=boom)

        trace.start_llm_call("假模型", 1, 3)
        trace.record_llm_call(self.make_stats())

        self.assertEqual(len(trace.llm_calls), 1)  # 记录照收

    def test_no_llm_observer_means_no_notification_but_the_records_stay(self):
        trace = Trace()

        trace.start_llm_call("假模型", 1, 3)
        trace.record_llm_call(self.make_stats())

        self.assertEqual(len(trace.llm_calls), 1)
        self.assertEqual(trace.events, [])


class AgentTraceTest(unittest.TestCase):
    def test_trace_exists_before_any_run(self):
        agent = Agent(client=FakeLLMClient())

        self.assertIsInstance(agent.trace, Trace)
        self.assertEqual(agent.get_trace(), [])

    def test_get_trace_returns_the_live_event_list(self):
        agent = Agent(client=FakeLLMClient())

        self.assertIs(agent.get_trace(), agent.trace.events)

    def test_executor_shares_the_agents_trace(self):
        agent = Agent(client=FakeLLMClient())

        self.assertIs(agent.executor.trace, agent.trace)

    def test_tool_call_shows_up_in_the_trace(self):
        agent, _ = build_agent(
            [tool_call_reply("echo", '{"text": "hi"}'), final_reply()], [EchoTool()]
        )

        events = agent.get_trace()

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tool, "echo")
        self.assertEqual(events[0].arguments, {"text": "hi"})
        self.assertTrue(events[0].success)

    def test_failed_tool_call_is_recorded(self):
        agent, _ = build_agent(
            [tool_call_reply("angry"), final_reply()], [AngryTool()]
        )

        event = agent.get_trace()[0]

        self.assertFalse(event.success)
        self.assertEqual(event.error, "我不干")

    def test_plain_chat_leaves_the_trace_empty(self):
        agent = Agent(client=FakeLLMClient([final_reply("你好")]))

        agent.run("你好")

        self.assertEqual(agent.get_trace(), [])

    def test_multiple_tool_calls_in_one_round_are_recorded_in_order(self):
        agent, _ = build_agent(
            [
                two_calls_reply(("echo", '{"text": "a"}'), ("slow", "{}")),
                final_reply(),
            ],
            [EchoTool(), SlowTool()],
        )

        self.assertEqual([event.tool for event in agent.get_trace()], ["echo", "slow"])

    def test_trace_is_serialisable_end_to_end(self):
        agent, _ = build_agent(
            [tool_call_reply("echo", '{"text": "hi"}'), final_reply()], [EchoTool()]
        )

        json.dumps(agent.trace.to_dicts())


class AgentTraceIsolationTest(unittest.TestCase):
    """不同 run() 的记录必须互不干扰。"""

    def make_two_round_agent(self):
        return Agent(
            client=FakeLLMClient(
                [
                    tool_call_reply("echo", '{"text": "one"}'),
                    final_reply("第一轮结束"),
                    tool_call_reply("echo", '{"text": "two"}'),
                    final_reply("第二轮结束"),
                ]
            ),
            tools=ToolRegistry([EchoTool()]),
        )

    def test_each_run_gets_a_fresh_trace_object(self):
        agent = self.make_two_round_agent()
        agent.run("第一轮")
        first_trace = agent.trace

        agent.run("第二轮")

        self.assertIsNot(first_trace, agent.trace)

    def test_second_run_only_records_its_own_calls(self):
        agent = self.make_two_round_agent()

        agent.run("第一轮")
        agent.run("第二轮")

        events = agent.get_trace()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].arguments, {"text": "two"})

    def test_previous_run_list_is_not_mutated(self):
        # 这是「换新对象」而不是「清空旧对象」的关键保证
        agent = self.make_two_round_agent()
        agent.run("第一轮")
        first = agent.get_trace()

        agent.run("第二轮")

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].arguments, {"text": "one"})

    def test_trace_is_cleared_when_a_new_run_starts(self):
        agent = Agent(
            client=FakeLLMClient(
                [
                    tool_call_reply("echo", '{"text": "a"}'),
                    final_reply(),
                    final_reply("直接回答"),
                ]
            ),
            tools=ToolRegistry([EchoTool()]),
        )
        agent.run("第一轮")
        self.assertEqual(len(agent.get_trace()), 1)

        agent.run("第二轮")  # 这轮不调工具

        self.assertEqual(len(agent.get_trace()), 0)

    def test_empty_input_also_resets_the_trace(self):
        # run() 第一件事就是换新 trace，空输入也不例外
        agent = self.make_two_round_agent()
        agent.run("第一轮")
        self.assertEqual(len(agent.get_trace()), 1)

        with self.assertRaises(ValueError):
            agent.run("   ")

        self.assertEqual(len(agent.get_trace()), 0)

    def test_failed_run_keeps_its_own_record(self):
        # 回滚的是消息历史；已经真实发生过的调用应该留着，方便排查
        agent = Agent(
            client=FakeLLMClient([tool_call_reply("echo", '{"text": "a"}')] * 3),
            tools=ToolRegistry([EchoTool()]),
            max_steps=3,
        )

        with self.assertRaises(AgentError):
            agent.run("试试")

        self.assertEqual(len(agent.get_trace()), 3)
        self.assertEqual(len(agent.messages), 1)  # 只剩 system，历史已回滚


class TraceDoesNotChangeBehaviourTest(unittest.TestCase):
    def test_messages_contain_no_trace_data(self):
        agent, _ = build_agent(
            [tool_call_reply("echo", '{"text": "hi"}'), final_reply("完成")],
            [EchoTool()],
        )

        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(agent.messages[3]["content"], "echo: hi")
        self.assertEqual(agent.messages[4]["content"], "完成")

    def test_answer_is_unchanged(self):
        _, answer = build_agent(
            [tool_call_reply("echo", '{"text": "hi"}'), final_reply("完成")],
            [EchoTool()],
        )

        self.assertEqual(answer, "完成")

    def test_max_steps_validation_is_unchanged(self):
        with self.assertRaises(ValueError):
            Agent(client=FakeLLMClient(), max_steps=0)

    def test_rollback_still_works(self):
        agent = Agent(
            client=FakeLLMClient([tool_call_reply("echo", '{"text": "a"}')] * 2),
            tools=ToolRegistry([EchoTool()]),
            max_steps=2,
        )

        with self.assertRaises(AgentError):
            agent.run("试试")

        self.assertEqual(len(agent.messages), 1)

    def test_history_is_still_usable_after_a_failed_run(self):
        agent = Agent(
            client=FakeLLMClient(
                [
                    tool_call_reply("echo", '{"text": "a"}'),
                    final_reply("第二轮结束"),
                ]
            ),
            tools=ToolRegistry([EchoTool()]),
            max_steps=1,
        )
        with self.assertRaises(AgentError):
            agent.run("第一轮")

        self.assertEqual(agent.run("第二轮"), "第二轮结束")
        self.assertEqual(len(agent.get_trace()), 0)


if __name__ == "__main__":
    unittest.main()
