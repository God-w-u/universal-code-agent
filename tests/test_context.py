"""ContextManager 测试：不裁剪、截断、按组删除、结构完整性、Agent 接入。

全部离线跑，不联网。
"""

import copy
import unittest

from agent import Agent, AgentError, ContextManager
from tools import ReadFileTool, Tool, ToolRegistry


def system(content: str = "你是助手") -> dict:
    return {"role": "system", "content": content}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def calling(*call_ids: str) -> dict:
    """assistant 要求调用工具的那条消息。"""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "big", "arguments": "{}"},
            }
            for call_id in call_ids
        ],
    }


def tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def conversation(rounds, answer: str = "完成") -> list[dict]:
    """拼一段完整对话。rounds 是 [(call_id, 工具返回内容), ...]。"""
    messages = [system(), user("帮我看看")]
    for call_id, content in rounds:
        messages.append(calling(call_id))
        messages.append(tool_result(call_id, content))
    messages.append(assistant(answer))
    return messages


def requested(messages) -> set[str]:
    """所有被 assistant 要求过的 tool_call_id。"""
    return {
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }


def answered(messages) -> set[str]:
    """所有收到了回复的 tool_call_id。"""
    return {
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    }


class ContextManagerBasicTest(unittest.TestCase):
    def test_short_history_passes_through_unchanged(self):
        manager = ContextManager()
        messages = [system(), user("你好"), assistant("你好呀")]

        self.assertEqual(manager.build(messages), messages)

    def test_empty_history(self):
        self.assertEqual(ContextManager().build([]), [])

    def test_build_returns_a_new_list(self):
        messages = [system(), user("你好")]

        self.assertIsNot(ContextManager().build(messages), messages)

    def test_build_returns_new_dicts(self):
        messages = [user("你好")]

        built = ContextManager().build(messages)

        self.assertIsNot(built[0], messages[0])
        self.assertEqual(built[0], messages[0])

    def test_original_messages_are_never_modified(self):
        manager = ContextManager(max_total_chars=200, max_tool_chars=20)
        messages = conversation([("c1", "x" * 200), ("c2", "y" * 200)])
        snapshot = copy.deepcopy(messages)

        manager.build(messages)

        self.assertEqual(messages, snapshot)

    def test_default_total_limit(self):
        self.assertEqual(ContextManager().max_total_chars, 100_000)

    def test_default_tool_limit_leaves_room_for_a_full_read_file(self):
        # 单条上限必须比 read_file 自己的返回上限宽松，
        # 否则 read_file 写在末尾的「只读了开头」会被再砍掉一次
        self.assertGreater(
            ContextManager().max_tool_chars, ReadFileTool.max_return_chars
        )

    def test_rejects_bad_limits(self):
        for kwargs in ({"max_total_chars": 0}, {"max_tool_chars": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ContextManager(**kwargs)


class ContextManagerTruncationTest(unittest.TestCase):
    def build_tool_content(self, content: str, max_tool_chars: int = 100) -> str:
        manager = ContextManager(max_total_chars=10_000, max_tool_chars=max_tool_chars)
        messages = [
            system(),
            user("读文件"),
            calling("c1"),
            tool_result("c1", content),
            assistant("好了"),
        ]
        return manager.build(messages)[3]["content"]

    def test_oversized_tool_result_is_truncated(self):
        content = self.build_tool_content("x" * 500)

        self.assertLess(len(content), 500)
        self.assertTrue(content.startswith("x" * 100))

    def test_truncation_says_it_is_not_the_full_result(self):
        content = self.build_tool_content("x" * 500)

        self.assertIn("已截断", content)
        self.assertIn("不是完整结果", content)

    def test_truncation_reports_the_original_length(self):
        self.assertIn("500", self.build_tool_content("x" * 500))

    def test_truncated_part_is_actually_kept(self):
        # 说明是附在保留内容后面的，不是把内容整个换掉
        content = self.build_tool_content("x" * 500)

        self.assertEqual(content.count("x"), 100)

    def test_small_tool_result_is_untouched(self):
        self.assertEqual(self.build_tool_content("短内容"), "短内容")

    def test_length_exactly_at_the_limit_is_untouched(self):
        self.assertEqual(self.build_tool_content("x" * 100), "x" * 100)

    def test_non_tool_messages_are_never_truncated(self):
        manager = ContextManager(max_total_chars=10_000, max_tool_chars=10)
        messages = [user("x" * 500), assistant("y" * 500)]

        built = manager.build(messages)

        self.assertEqual(built[0]["content"], "x" * 500)
        self.assertEqual(built[1]["content"], "y" * 500)

    def test_truncation_does_not_touch_the_original(self):
        manager = ContextManager(max_total_chars=10_000, max_tool_chars=100)
        messages = [tool_result("c1", "x" * 500)]

        manager.build(messages)

        self.assertEqual(messages[0]["content"], "x" * 500)


class ContextManagerEvictionTest(unittest.TestCase):
    """1000 字符预算 + 每轮 400 字符，三轮就会超，四轮必定删。"""

    def make(self, rounds: int, tool_chars: int = 400):
        manager = ContextManager(max_total_chars=1000, max_tool_chars=1000)
        messages = conversation(
            [(f"c{index}", "x" * tool_chars) for index in range(rounds)]
        )
        return manager, messages

    def test_under_the_limit_drops_nothing(self):
        manager, messages = self.make(rounds=2)

        self.assertEqual(manager.build(messages), messages)

    def test_over_the_limit_drops_something(self):
        manager, messages = self.make(rounds=3)

        self.assertLess(len(manager.build(messages)), len(messages))

    def test_system_message_survives(self):
        manager, messages = self.make(rounds=5)

        self.assertEqual(manager.build(messages)[0], system())

    def test_last_user_message_survives(self):
        manager, messages = self.make(rounds=5)

        self.assertIn(user("帮我看看"), manager.build(messages))

    def test_last_round_survives(self):
        manager, messages = self.make(rounds=5)

        self.assertIn("c4", requested(manager.build(messages)))
        self.assertIn("c4", answered(manager.build(messages)))

    def test_the_oldest_round_is_the_first_to_go(self):
        manager, messages = self.make(rounds=3)

        built = manager.build(messages)

        self.assertNotIn("c0", requested(built))
        self.assertIn("c1", requested(built))
        self.assertIn("c2", requested(built))

    def test_result_is_never_dropped_without_its_call(self):
        manager, messages = self.make(rounds=5)

        built = manager.build(messages)

        self.assertNotIn("c0", answered(built))
        self.assertNotIn("c0", requested(built))

    def test_it_keeps_dropping_until_it_fits(self):
        manager = ContextManager(max_total_chars=900, max_tool_chars=1000)
        messages = conversation([(f"c{i}", "x" * 400) for i in range(6)])

        built = manager.build(messages)

        total = sum(len(message.get("content") or "") for message in built)
        self.assertLessEqual(total, 900)

    def test_nothing_droppable_still_returns_a_usable_conversation(self):
        # system + 最后一条用户消息 + 最后一组都不能删，那就宁可超一点
        manager = ContextManager(max_total_chars=1, max_tool_chars=1000)
        messages = [system(), user("只有一个问题")]

        built = manager.build(messages)

        self.assertEqual(built, messages)

    def test_a_huge_user_message_is_not_silently_dropped(self):
        manager = ContextManager(max_total_chars=100, max_tool_chars=1000)
        messages = [system(), user("x" * 5000)]

        built = manager.build(messages)

        self.assertIn(user("x" * 5000), built)


class ContextManagerIntegrityTest(unittest.TestCase):
    """删完之后必须是结构完整的对话，否则接口会直接拒掉整个请求。"""

    def assert_complete(self, messages):
        self.assertEqual(requested(messages), answered(messages))

    def test_no_call_is_left_without_its_result(self):
        manager = ContextManager(max_total_chars=1000, max_tool_chars=1000)

        for rounds in (1, 2, 3, 5):
            with self.subTest(rounds=rounds):
                messages = conversation(
                    [(f"c{index}", "x" * 400) for index in range(rounds)]
                )
                self.assert_complete(manager.build(messages))

    def test_very_small_budget_still_leaves_a_valid_conversation(self):
        manager = ContextManager(max_total_chars=10, max_tool_chars=1000)
        messages = conversation([(f"c{i}", "x" * 400) for i in range(6)])

        built = manager.build(messages)

        self.assert_complete(built)
        self.assertTrue(built)

    def test_tool_messages_still_follow_their_call(self):
        manager = ContextManager(max_total_chars=1000, max_tool_chars=1000)
        messages = conversation([(f"c{i}", "x" * 400) for i in range(5)])

        built = manager.build(messages)

        for index, message in enumerate(built):
            if message.get("role") != "tool":
                continue
            with self.subTest(index=index):
                calls = set()
                for earlier in reversed(built[:index]):
                    if earlier.get("role") == "user":
                        break
                    if earlier.get("role") == "assistant" and earlier.get(
                        "tool_calls"
                    ):
                        calls = {call["id"] for call in earlier["tool_calls"]}
                        break
                self.assertIn(message["tool_call_id"], calls)

    def test_several_tool_results_for_one_call_go_together(self):
        manager = ContextManager(max_total_chars=1000, max_tool_chars=1000)
        messages = [system(), user("查两个文件")]
        for index in range(3):
            messages.append(calling(f"a{index}", f"b{index}"))
            messages.append(tool_result(f"a{index}", "x" * 300))
            messages.append(tool_result(f"b{index}", "y" * 300))
        messages.append(assistant("完成"))

        built = manager.build(messages)

        self.assert_complete(built)


class RecordingClient:
    """记下每次实际收到的消息。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.received: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.received.append(list(messages))
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


class SpyContext(ContextManager):
    """数一下 build() 被调用了几次。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.builds = 0

    def build(self, messages):
        self.builds += 1
        return super().build(messages)


class BigTool(Tool):
    name = "big"
    description = "返回一大段内容"
    parameters = {}

    def execute(self) -> str:
        return "y" * 400


def tool_call_reply(call_id: str):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "big", "arguments": "{}"},
            }
        ],
    }


def final_reply(content: str = "完成"):
    return {"role": "assistant", "content": content}


class AgentContextTest(unittest.TestCase):
    def make_agent(self, replies, context=None, max_steps=4):
        client = RecordingClient(replies)
        agent = Agent(
            client=client,
            tools=ToolRegistry([BigTool()]),
            max_steps=max_steps,
            context=context,
        )
        return agent, client

    def test_agent_has_a_context_manager_by_default(self):
        agent = Agent(client=RecordingClient())

        self.assertIsInstance(agent.context, ContextManager)

    def test_a_given_context_manager_is_used(self):
        manager = ContextManager()
        agent = Agent(client=RecordingClient(), context=manager)

        self.assertIs(agent.context, manager)

    def test_every_llm_request_goes_through_the_context_manager(self):
        spy = SpyContext()
        agent, _ = self.make_agent(
            [tool_call_reply("c1"), final_reply()], context=spy
        )

        agent.run("试试")

        self.assertEqual(spy.builds, 2)  # 两次请求模型 → 裁剪两次

    def test_plain_chat_also_goes_through_the_context_manager(self):
        spy = SpyContext()
        agent = Agent(client=RecordingClient([final_reply("你好")]), context=spy)

        agent.run("你好")

        self.assertEqual(spy.builds, 1)

    def test_llm_receives_less_than_the_full_history(self):
        agent, client = self.make_agent(
            [
                tool_call_reply("c1"),
                tool_call_reply("c2"),
                tool_call_reply("c3"),
                final_reply(),
            ],
            context=ContextManager(max_total_chars=1000, max_tool_chars=1000),
        )

        agent.run("试试")

        self.assertLess(len(client.received[-1]), len(agent.messages))
        self.assertNotIn("c1", requested(client.received[-1]))

    def test_full_history_is_still_kept(self):
        agent, client = self.make_agent(
            [
                tool_call_reply("c1"),
                tool_call_reply("c2"),
                tool_call_reply("c3"),
                final_reply(),
            ],
            context=ContextManager(max_total_chars=1000, max_tool_chars=1000),
        )

        agent.run("试试")

        self.assertEqual(len(agent.messages), 9)  # system + user + 3 轮 + 回答
        self.assertEqual(
            sum(1 for message in agent.messages if message["role"] == "tool"), 3
        )

    def test_tool_results_are_kept_in_full_in_the_history(self):
        agent, _ = self.make_agent(
            [tool_call_reply("c1"), final_reply()],
            context=ContextManager(max_total_chars=100, max_tool_chars=100),
        )

        agent.run("试试")

        tool_message = next(m for m in agent.messages if m["role"] == "tool")
        self.assertEqual(tool_message["content"], "y" * 400)

    def test_tool_calling_still_works(self):
        agent, client = self.make_agent(
            [tool_call_reply("c1"), final_reply("完成")],
            context=ContextManager(max_total_chars=1000, max_tool_chars=1000),
        )

        self.assertEqual(agent.run("试试"), "完成")
        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )

    def test_max_steps_is_unaffected(self):
        agent, _ = self.make_agent(
            [tool_call_reply("c1")] * 3,
            context=ContextManager(max_total_chars=1000, max_tool_chars=1000),
            max_steps=2,
        )

        with self.assertRaises(AgentError):
            agent.run("试试")

    def test_rollback_is_unaffected(self):
        agent, _ = self.make_agent(
            [tool_call_reply("c1")] * 3,
            context=ContextManager(max_total_chars=1000, max_tool_chars=1000),
            max_steps=2,
        )

        with self.assertRaises(AgentError):
            agent.run("试试")

        self.assertEqual(len(agent.messages), 1)  # 只剩 system

    def test_trace_is_unaffected(self):
        agent, _ = self.make_agent(
            [tool_call_reply("c1"), final_reply()],
            context=ContextManager(max_total_chars=1000, max_tool_chars=1000),
        )

        agent.run("试试")

        self.assertEqual(len(agent.get_trace()), 1)
        self.assertTrue(agent.get_trace()[0].success)


if __name__ == "__main__":
    unittest.main()
