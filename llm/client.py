"""LLM 客户端：一问一答，以及带工具调用的一轮对话。

只负责「发消息、收消息」，不关心工具怎么执行——那是 Agent 的事。
"""

from typing import Any

from openai import OpenAI, OpenAIError

from config import LLMConfig, load_llm_config


class LLMError(RuntimeError):
    """调用 LLM 失败。"""


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


class LLMClient:
    """对 OpenAI 兼容接口的最小封装。"""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_config()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    def ask(self, question: str) -> str:
        """把 question 发给模型，返回模型回复的纯文本（单轮）。"""
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        return self.chat([{"role": "user", "content": question}])

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """把完整消息列表发给模型，返回模型回复的纯文本。"""
        return self._complete(messages).content or ""

    def chat_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """带上工具清单发一轮，返回 assistant 消息字典。

        返回值不是纯文本，因为这一轮模型可能不回答，而是要求调用工具；
        这些要求放在返回字典的 "tool_calls" 里。
        """
        if not tools:
            raise ValueError("tools 不能为空")

        return _message_to_dict(self._complete(messages, tools=tools))

    def _complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """请求接口，返回第一条 choice 里的 message 对象。"""
        if not messages:
            raise ValueError("messages 不能为空")

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMError(f"调用 LLM 失败：{exc}") from exc

        if not response.choices:
            raise LLMError("LLM 返回了空结果（choices 为空）")

        return response.choices[0].message
