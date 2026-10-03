"""LLMClient 测试：验证 ask() 复用 chat()，不重复实现调用逻辑。

用假 SDK 客户端替代真实 openai 客户端，不产生网络请求。
"""

import unittest
from types import SimpleNamespace

from openai import OpenAIError

from config import LLMConfig
from llm import LLMClient, LLMError
from tools import CalculatorTool


def _fake_openai_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _tool_call_response(name="calculator", arguments='{"expression": "3*7"}'):
    """模拟 SDK 返回的「要求调用工具」的 message（content 为 None）。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                )
            )
        ]
    )


def _blank_response():
    return SimpleNamespace(choices=[])


class LLMClientTest(unittest.TestCase):
    def _client_with(self, create):
        """构造 LLMClient，但把内部 SDK 客户端换成假的。"""
        client = LLMClient(config=LLMConfig(api_key="dummy", model="fake-model"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client

    def test_ask_delegates_to_chat(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return _fake_openai_response("ok")

        client = self._client_with(create)

        self.assertEqual(client.ask("  你好  "), "ok")
        self.assertEqual(captured["model"], "fake-model")
        self.assertEqual(captured["messages"], [{"role": "user", "content": "你好"}])

    def test_chat_passes_messages_through(self):
        captured = {}
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]

        def create(**kwargs):
            captured.update(kwargs)
            return _fake_openai_response("A2")

        client = self._client_with(create)

        self.assertEqual(client.chat(messages), "A2")
        self.assertEqual(captured["messages"], messages)

    def test_empty_messages_rejected(self):
        client = self._client_with(lambda **_: _fake_openai_response("x"))

        with self.assertRaises(ValueError):
            client.chat([])

    def test_openai_error_wrapped_as_llm_error(self):
        def create(**_):
            raise OpenAIError("network down")

        client = self._client_with(create)

        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_empty_choices_raise_llm_error(self):
        client = self._client_with(lambda **_: _blank_response())

        with self.assertRaises(LLMError):
            client.chat([{"role": "user", "content": "hi"}])


class LLMClientToolCallingTest(unittest.TestCase):
    """chat_with_tools()：把 tools 传给接口，把 tool_calls 转成普通字典。"""

    def setUp(self):
        self.tools = [CalculatorTool().to_schema()]
        self.messages = [{"role": "user", "content": "3*7 等于几"}]

    def _client_with(self, create):
        client = LLMClient(config=LLMConfig(api_key="dummy", model="fake-model"))
        client._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return client

    def test_tools_are_passed_to_api(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return _fake_openai_response("ok")

        client = self._client_with(create)

        client.chat_with_tools(self.messages, self.tools)

        self.assertEqual(captured["tools"], self.tools)
        self.assertEqual(captured["messages"], self.messages)

    def test_chat_does_not_send_tools(self):
        # 普通 chat() 不能带上 tools，否则模型可能无缘无故要求调工具
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return _fake_openai_response("ok")

        client = self._client_with(create)

        client.chat(self.messages)

        self.assertNotIn("tools", captured)

    def test_plain_answer_returns_plain_message_dict(self):
        client = self._client_with(lambda **_: _fake_openai_response("21"))

        message = client.chat_with_tools(self.messages, self.tools)

        self.assertEqual(message, {"role": "assistant", "content": "21"})

    def test_tool_calls_are_converted_to_dicts(self):
        client = self._client_with(lambda **_: _tool_call_response())

        message = client.chat_with_tools(self.messages, self.tools)

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "")  # content 为 None 时归一成空串
        self.assertEqual(
            message["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "3*7"}',
                    },
                }
            ],
        )

    def test_empty_tools_rejected(self):
        client = self._client_with(lambda **_: _fake_openai_response("x"))

        with self.assertRaises(ValueError):
            client.chat_with_tools(self.messages, [])

    def test_empty_messages_rejected(self):
        client = self._client_with(lambda **_: _fake_openai_response("x"))

        with self.assertRaises(ValueError):
            client.chat_with_tools([], self.tools)

    def test_openai_error_wrapped_as_llm_error(self):
        def create(**_):
            raise OpenAIError("network down")

        client = self._client_with(create)

        with self.assertRaises(LLMError):
            client.chat_with_tools(self.messages, self.tools)

    def test_empty_choices_raise_llm_error(self):
        client = self._client_with(lambda **_: _blank_response())

        with self.assertRaises(LLMError):
            client.chat_with_tools(self.messages, self.tools)


if __name__ == "__main__":
    unittest.main()
