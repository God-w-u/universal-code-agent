"""Agent 测试：单轮、多轮、异常，以及 Phase 4 的 tool-calling 循环。

用测试替身替代真实 LLM，不产生网络请求。
"""

import unittest

from agent import DEFAULT_MAX_STEPS, Agent, AgentError
from llm import LLMError
from tools import CalculatorTool, ToolRegistry


class FakeLLMClient:
    """替代 LLMClient 的测试替身。

    replies 是脚本化的回答序列，每一项可以是：
    - 字符串：模型直接给出的最终回答
    - 字典：模型要求调用工具的消息（形如 tool_call_reply() 的返回值）
    - 异常实例：这次调用失败，抛出该异常
    最后一项用完会一直重复，方便测试「模型停不下来」这类场景。
    """

    def __init__(self, replies=("fake answer",)):
        self.replies = list(replies)
        self.calls = []  # 每次调用收到的 messages 快照
        self.tool_schemas = []  # 每次调用收到的 tools 参数

    def _next_reply(self):
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])
        reply = self._next_reply()
        return reply if isinstance(reply, str) else reply.get("content", "")

    def chat_with_tools(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        self.tool_schemas.append(tools)
        reply = self._next_reply()
        if isinstance(reply, str):
            return {"role": "assistant", "content": reply}
        return reply


def tool_call_reply(*calls):
    """构造「模型要求调用工具」的回复。

    calls 里每项是 (工具名, 参数 JSON 字符串)，例如 ("calculator", '{"expression": "1+1"}')。
    """
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
            for index, (name, arguments) in enumerate(calls, start=1)
        ],
    }


def roles(messages):
    return [m["role"] for m in messages]


class AgentSingleTurnTest(unittest.TestCase):
    def test_returns_answer_and_records_history(self):
        client = FakeLLMClient(["你好，我是 Agent。"])
        agent = Agent(client=client)

        answer = agent.run("你好")

        self.assertEqual(answer, "你好，我是 Agent。")
        self.assertEqual(roles(agent.messages), ["system", "user", "assistant"])
        self.assertEqual(agent.messages[1]["content"], "你好")
        self.assertEqual(agent.messages[2]["content"], "你好，我是 Agent。")

    def test_sends_system_and_user_to_llm(self):
        client = FakeLLMClient()
        agent = Agent(client=client, system_prompt="SYS")

        agent.run("问题")

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(
            client.calls[0],
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "问题"},
            ],
        )

    def test_input_is_stripped(self):
        client = FakeLLMClient()
        agent = Agent(client=client)

        agent.run("  前后有空格  ")

        self.assertEqual(agent.messages[1]["content"], "前后有空格")


class AgentMultiTurnTest(unittest.TestCase):
    def test_history_accumulates_across_turns(self):
        client = FakeLLMClient(["第一轮回答", "第二轮回答"])
        agent = Agent(client=client, system_prompt="SYS")

        agent.run("第一个问题")
        agent.run("第二个问题")

        self.assertEqual(
            roles(agent.messages),
            ["system", "user", "assistant", "user", "assistant"],
        )

    def test_second_call_receives_full_history(self):
        client = FakeLLMClient(["第一轮回答", "第二轮回答"])
        agent = Agent(client=client, system_prompt="SYS")

        agent.run("第一个问题")
        agent.run("第二个问题")

        # 第二次请求必须带上第一轮的上下文，否则不是多轮对话。
        self.assertEqual(
            [(m["role"], m["content"]) for m in client.calls[1]],
            [
                ("system", "SYS"),
                ("user", "第一个问题"),
                ("assistant", "第一轮回答"),
                ("user", "第二个问题"),
            ],
        )


class AgentErrorTest(unittest.TestCase):
    def test_llm_error_propagates(self):
        client = FakeLLMClient([LLMError("boom")])
        agent = Agent(client=client)

        with self.assertRaises(LLMError):
            agent.run("会失败的问题")

    def test_failed_turn_is_rolled_back(self):
        client = FakeLLMClient([LLMError("boom")])
        agent = Agent(client=client)

        with self.assertRaises(LLMError):
            agent.run("会失败的问题")

        # 历史不能留下没有回答的 user 消息，否则重试会重复提问。
        self.assertEqual(roles(agent.messages), ["system"])

    def test_empty_input_rejected(self):
        client = FakeLLMClient()
        agent = Agent(client=client)

        with self.assertRaises(ValueError):
            agent.run("   ")

        self.assertEqual(roles(agent.messages), ["system"])
        self.assertEqual(client.calls, [])


class AgentToolsTest(unittest.TestCase):
    def test_holds_given_registry(self):
        registry = ToolRegistry([CalculatorTool()])
        agent = Agent(client=FakeLLMClient(), tools=registry)

        self.assertIs(agent.tools, registry)
        self.assertIn("calculator", agent.tools)

    def test_defaults_to_empty_registry(self):
        agent = Agent(client=FakeLLMClient())

        self.assertEqual(len(agent.tools), 0)

    def test_registered_tools_are_offered_to_llm(self):
        # Phase 3 时工具只是被持有；Phase 4 起会把 schema 发给模型
        client = FakeLLMClient(["回答"])
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        self.assertEqual(agent.run("1+1 等于几"), "回答")
        self.assertEqual(len(client.tool_schemas), 1)
        self.assertEqual(client.tool_schemas[0][0]["function"]["name"], "calculator")

    def test_no_tools_means_plain_chat(self):
        client = FakeLLMClient(["回答"])
        agent = Agent(client=client)

        self.assertEqual(agent.run("你好"), "回答")
        self.assertEqual(client.tool_schemas, [])  # 没工具就不发 tools 参数


class AgentToolCallingTest(unittest.TestCase):
    """Phase 4 核心：模型要求调工具 -> 执行 -> 结果回灌 -> 再问模型。"""

    def setUp(self):
        self.client = FakeLLMClient(
            [
                tool_call_reply(("calculator", '{"expression": "3*7"}')),
                "结果是 21",
            ]
        )
        self.agent = Agent(client=self.client, tools=ToolRegistry([CalculatorTool()]))

    def test_returns_final_answer_after_tool_call(self):
        self.assertEqual(self.agent.run("3*7 等于几"), "结果是 21")
        self.assertEqual(
            roles(self.agent.messages),
            ["system", "user", "assistant", "tool", "assistant"],
        )

    def test_tool_message_carries_real_calculator_result(self):
        self.agent.run("3*7 等于几")

        tool_message = self.agent.messages[3]
        # 结果必须来自 CalculatorTool，而不是模型自己算的
        self.assertEqual(tool_message["content"], "21")
        self.assertEqual(tool_message["tool_call_id"], "call_1")

    def test_second_request_contains_tool_result(self):
        self.agent.run("3*7 等于几")

        # 第二次请求必须带上 Observation，否则模型看不到工具结果
        self.assertEqual(
            [(m["role"], m.get("content")) for m in self.client.calls[1]],
            [
                ("system", self.agent.system_prompt),
                ("user", "3*7 等于几"),
                ("assistant", ""),
                ("tool", "21"),
            ],
        )

    def test_assistant_message_keeps_tool_calls_in_history(self):
        self.agent.run("3*7 等于几")

        assistant_message = self.agent.messages[2]
        self.assertEqual(
            assistant_message["tool_calls"][0]["function"]["name"], "calculator"
        )

    def test_two_rounds_of_tool_calls(self):
        client = FakeLLMClient(
            [
                tool_call_reply(("calculator", '{"expression": "1+1"}')),
                tool_call_reply(("calculator", '{"expression": "2+2"}')),
                "两题都算完了",
            ]
        )
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        self.assertEqual(agent.run("算两道题"), "两题都算完了")
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(
            [m["content"] for m in agent.messages if m["role"] == "tool"], ["2", "4"]
        )

    def test_multiple_tool_calls_in_one_message(self):
        client = FakeLLMClient(
            [
                tool_call_reply(
                    ("calculator", '{"expression": "1+1"}'),
                    ("calculator", '{"expression": "2*3"}'),
                ),
                "算完了",
            ]
        )
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        agent.run("算两个式子")

        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual([m["content"] for m in tool_messages], ["2", "6"])
        self.assertEqual(
            [m["tool_call_id"] for m in tool_messages], ["call_1", "call_2"]
        )

    def test_tool_call_with_empty_content_is_not_treated_as_answer(self):
        # 模型要求调工具时 content 是空的，不能被当最终回答直接返回
        self.assertEqual(self.agent.run("3*7 等于几"), "结果是 21")
        self.assertEqual(len(self.client.calls), 2)


class AgentToolErrorFeedbackTest(unittest.TestCase):
    """工具出错不能让 Agent 崩，而是把错误回灌给模型，让它自己决定下一步。"""

    FINAL = "好的，我换个方式"

    def _run_with_call(self, name, arguments):
        client = FakeLLMClient([tool_call_reply((name, arguments)), self.FINAL])
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))
        return agent.run("算一下"), agent

    def test_tool_error_is_fed_back(self):
        answer, agent = self._run_with_call("calculator", '{"expression": "1/0"}')

        self.assertEqual(answer, self.FINAL)
        self.assertTrue(agent.messages[3]["content"].startswith("错误："))
        self.assertIn("除数不能为 0", agent.messages[3]["content"])

    def test_unknown_tool_name_is_fed_back(self):
        _, agent = self._run_with_call("nope", "{}")

        self.assertIn("未注册的工具", agent.messages[3]["content"])

    def test_wrong_argument_name_is_fed_back(self):
        _, agent = self._run_with_call("calculator", '{"expr": "1+1"}')

        self.assertIn("不接受参数", agent.messages[3]["content"])

    def test_missing_argument_is_fed_back(self):
        _, agent = self._run_with_call("calculator", "{}")

        self.assertIn("缺少必需参数", agent.messages[3]["content"])

    def test_invalid_json_arguments_are_fed_back(self):
        _, agent = self._run_with_call("calculator", "不是 JSON")

        self.assertIn("不是合法 JSON", agent.messages[3]["content"])

    def test_non_object_arguments_are_fed_back(self):
        _, agent = self._run_with_call("calculator", "[1, 2]")

        self.assertIn("JSON 对象", agent.messages[3]["content"])

    def test_agent_survives_every_tool_error(self):
        for arguments in ("1/0", "abc", "1++"):
            with self.subTest(arguments=arguments):
                answer, agent = self._run_with_call(
                    "calculator", f'{{"expression": "{arguments}"}}'
                )
                self.assertEqual(answer, self.FINAL)


class AgentToolArgumentsShapeTest(unittest.TestCase):
    """`arguments` 的形状不对时不许把异常穿透出去（13 阶段回归）。

    规范说 arguments 是 JSON 字符串，但那是**模型 / 网关给的**，形状不完全
    由我们说了算。以前只挡了「JSON 坏了」这一种，`json.loads` 拿到数字 /
    列表 / None 会抛 TypeError——它不在 run() 认得的那几种异常里，回滚不会
    发生，历史里就此留下一条没有 tool 回复的 assistant tool_calls：上游
    之后**每一次**请求都会拒掉它，这段会话等于废了。
    """

    FINAL = "好的，我换个方式"

    def run_with_raw_arguments(self, arguments):
        """让模型要求调工具，arguments 想给什么形状就给什么形状。"""
        client = FakeLLMClient([tool_call_reply(("calculator", arguments)), self.FINAL])
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))
        return agent.run("算一下"), agent

    def test_arguments_that_are_not_a_string_are_fed_back(self):
        for arguments in (5, 1.5, None, True, ["1+1"], b"{}"):
            with self.subTest(arguments=arguments):
                answer, agent = self.run_with_raw_arguments(arguments)

                self.assertEqual(answer, self.FINAL)  # 没炸，这一轮照常走完
                # 回灌的是一条 tool 消息，形状配得上上面那条 tool_calls
                self.assertEqual(agent.messages[3]["role"], "tool")
                self.assertTrue(agent.messages[3]["content"].startswith("错误："))

    def test_the_history_stays_acceptable_to_the_upstream(self):
        # 这一轮完整走完（没有半截 tool_calls），而且能接着开下一轮
        answer, agent = self.run_with_raw_arguments(5)

        roles = [message["role"] for message in agent.messages]

        self.assertEqual(answer, self.FINAL)
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(
            len(agent.messages[2]["tool_calls"]),
            1,  # 要了一次工具
        )
        self.assertEqual(agent.messages[3]["tool_call_id"], "call_1")

    def test_arguments_already_parsed_into_an_object_still_work(self):
        # 有些 OpenAI 兼容网关直接把对象给过来。值是对的，收下就是——
        # 这里钉住「正常 tool calling 一步没变」
        answer, agent = self.run_with_raw_arguments({"expression": "1+1"})

        self.assertEqual(answer, self.FINAL)
        self.assertEqual(agent.messages[3]["role"], "tool")
        self.assertNotIn("错误", agent.messages[3]["content"])
        self.assertIn("2", agent.messages[3]["content"])  # 工具真跑了

    def test_a_tool_call_that_is_not_an_object_rolls_the_run_back(self):
        # 连 call_id 都取不出来，造不出配得上的 tool 回复：只能整轮回滚，
        # 历史里绝不能留下「要了工具却没有回复」的半截结构
        client = FakeLLMClient([{"role": "assistant", "content": "", "tool_calls": [7]}])
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        with self.assertRaises(AgentError):
            agent.run("算一下")

        self.assertEqual([m["role"] for m in agent.messages], ["system"])

    def test_a_function_that_is_not_an_object_rolls_the_run_back_too(self):
        client = FakeLLMClient(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "function": "calculator"}],
                }
            ]
        )
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        with self.assertRaises(AgentError):
            agent.run("算一下")

        self.assertEqual([m["role"] for m in agent.messages], ["system"])


class AgentToolLoopLimitTest(unittest.TestCase):
    def test_limit_reached_raises_agent_error_and_rolls_back(self):
        # 模型一直要求调工具，永远不给最终回答
        client = FakeLLMClient(
            [tool_call_reply(("calculator", '{"expression": "1+1"}'))]
        )
        agent = Agent(
            client=client, tools=ToolRegistry([CalculatorTool()]), max_steps=2
        )

        with self.assertRaises(AgentError):
            agent.run("停不下来")

        self.assertEqual(roles(agent.messages), ["system"])
        self.assertEqual(len(client.calls), 2)

    def test_max_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            Agent(client=FakeLLMClient(), max_steps=0)

    def test_default_budget_leaves_room_for_a_real_task(self):
        # 这条不是测功能，是钉住那个默认值别被改回去。
        # 实测：单是问「工具是怎么注册进去的」，模型就要跑 4~6 轮工具调用；
        # 默认值设成 5 时，同一个问题跑两次，一次正好答完，一次撞墙。
        self.assertGreaterEqual(DEFAULT_MAX_STEPS, 10)

    def test_llm_error_mid_loop_rolls_back_whole_turn(self):
        client = FakeLLMClient(
            [
                tool_call_reply(("calculator", '{"expression": "1+1"}')),
                LLMError("boom"),
            ]
        )
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        with self.assertRaises(LLMError):
            agent.run("会失败的问题")

        # 中途产生的 assistant/tool 消息也要一起回滚，不能留下半截对话
        self.assertEqual(roles(agent.messages), ["system"])

    def test_history_is_still_usable_after_rollback(self):
        client = FakeLLMClient(
            [
                tool_call_reply(("calculator", '{"expression": "1+1"}')),
                LLMError("boom"),
                "这次成功了",
            ]
        )
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]))

        with self.assertRaises(LLMError):
            agent.run("会失败的问题")

        self.assertEqual(agent.run("再试一次"), "这次成功了")


if __name__ == "__main__":
    unittest.main()
