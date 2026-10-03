"""发送给 LLM 之前的上下文裁剪。

self.messages 始终是完整历史，本模块只决定「这一次请求该发什么」：

    self.messages → ContextManager.build() → LLM

裁剪分两步，先轻后重：
    1. 单条 tool 结果太长 → 截断，并明确告诉模型这不是完整内容
    2. 总量还是超 → 从最老的整轮工具调用开始删

不碰 RAG、向量库、长期记忆、LLM 摘要——那些是以后的事。
这一版只保证一件事：请求不会因为历史太长而超限。
"""

from typing import Any

# 一次请求里所有消息 content 加起来的字符上限
MAX_TOTAL_CHARS = 100_000

# 单条 tool 结果的字符上限。
# 要比 ReadFileTool.max_return_chars（40000）宽松：否则 read_file 自己写在末尾的
# 截断说明会被我们再砍一次，模型就看不到「这个文件本来就只读了一半」了。
MAX_TOOL_CHARS = 50_000

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_TOOL = "tool"


def _truncation_note(total: int, kept: int) -> str:
    """截断说明。必须点明「不是完整内容」，否则模型会以为拿到了全文。"""
    return f"\n…（该内容已截断，不是完整结果：原有 {total} 字符，只发送了前 {kept} 个）"


def _total_chars(groups: list[list[dict[str, Any]]]) -> int:
    """所有消息 content 的字符总数。

    只数 content：tool_calls 里的 arguments 占比很小，
    不值得为了算准它把逻辑搞复杂。
    """
    return sum(
        len(message.get("content") or "")
        for group in groups
        for message in group
    )


def _group(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """把消息切成「不能拆开」的组。

    关键：assistant 要求调用工具的那一条，和它后面几条 tool 回复必须同进同出。
    OpenAI 要求每个 tool_call_id 都有对应的 tool 回复，只删掉一半，
    请求会被接口直接拒掉。所以它们算作一组。

    返回的是原消息的引用，不做复制——复制统一在 _truncate() 里做。
    """
    groups: list[list[dict[str, Any]]] = []
    index = 0

    while index < len(messages):
        message = messages[index]
        group = [message]
        index += 1

        if message.get("tool_calls"):
            while index < len(messages) and messages[index].get("role") == ROLE_TOOL:
                group.append(messages[index])
                index += 1

        groups.append(group)

    return groups


def _pick_victim(groups: list[list[dict[str, Any]]]) -> int | None:
    """挑一个可以删的组，返回下标；没有可删的就返回 None。

    不能删的：system、最后一条用户消息、最后一组。
    最后一组可能是正在进行中的这一轮工具调用，删了模型就看不到自己刚要到的东西。

    删除顺序：先删最老的、带 tool 结果的组；都删完了才轮到普通的旧消息。
    """
    protected = {len(groups) - 1} if groups else set()
    last_user = None

    for index, group in enumerate(groups):
        if any(message.get("role") == ROLE_SYSTEM for message in group):
            protected.add(index)
        if any(message.get("role") == ROLE_USER for message in group):
            last_user = index

    if last_user is not None:
        protected.add(last_user)

    # 第一优先：最老的、含 tool 结果的组
    for index, group in enumerate(groups):
        if index not in protected and any(
            message.get("role") == ROLE_TOOL for message in group
        ):
            return index

    # 第二优先：最老的可删组
    for index in range(len(groups)):
        if index not in protected:
            return index

    return None


class ContextManager:
    """控制单次请求发送多少上下文。"""

    def __init__(
        self,
        max_total_chars: int = MAX_TOTAL_CHARS,
        max_tool_chars: int = MAX_TOOL_CHARS,
    ) -> None:
        if max_total_chars < 1:
            raise ValueError("max_total_chars 至少为 1")
        if max_tool_chars < 1:
            raise ValueError("max_tool_chars 至少为 1")

        self.max_total_chars = max_total_chars
        self.max_tool_chars = max_tool_chars

    def build(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """返回这一次可以发给 LLM 的消息。

        不修改传入的 messages，返回的都是新字典。
        """
        trimmed = [self._truncate(message) for message in messages]
        return self._evict(trimmed)

    def _truncate(self, message: dict[str, Any]) -> dict[str, Any]:
        """复制一条消息；tool 结果过长就截断并附上说明。"""
        copied = dict(message)

        if copied.get("role") != ROLE_TOOL:
            return copied

        content = copied.get("content") or ""
        if len(content) <= self.max_tool_chars:
            return copied

        copied["content"] = content[: self.max_tool_chars] + _truncation_note(
            len(content), self.max_tool_chars
        )
        return copied

    def _evict(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """超量就一组一组删，直到装得下或者没得删。"""
        groups = _group(messages)

        while _total_chars(groups) > self.max_total_chars:
            index = _pick_victim(groups)
            if index is None:
                # 剩下的都是不能删的（system、最后一条用户消息、最后一组）。
                # 宁可稍微超一点发出去，也不能把对话结构删坏。
                break
            del groups[index]

        return [message for group in groups for message in group]
