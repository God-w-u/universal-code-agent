"""Agent 执行过程的记录（Trace）：一次 run() 调了哪些工具、什么参数、成没成、多久。

刻意保持简单：只有一种事件（工具调用）和一个容器，不做通用事件系统。

注意：本模块只依赖标准库，不导入 tools / agent 里的任何东西。
tools/executor.py 会调用这里的 Trace，反过来这里不依赖它，避免形成导入环。
"""

from dataclasses import dataclass, field
from typing import Any

# 目前只有这一种事件。留着这个字段是为了以后要加别的类型时不用改结构。
TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class TraceEvent:
    """一条执行记录。"""

    type: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转成可以直接 json.dumps 的普通字典。

        失败时 error 为空字符串也照样带上，保证各条记录的键完全一致，
        读取方不用先判断键在不在。
        """
        return {
            "type": self.type,
            "tool": self.tool,
            "arguments": self.arguments,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


class Trace:
    """一次 run() 产生的全部执行记录。

    由 Agent 每轮新建一个，不同轮之间互不影响。
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        success: bool,
        error: str = "",
        duration_seconds: float = 0.0,
    ) -> TraceEvent:
        """记下一次工具调用，返回刚记下的那条。"""
        event = TraceEvent(
            type=TOOL_CALL,
            tool=tool,
            # 复制一份：记录的是调用当时的样子，之后谁改了原字典都不影响这里
            arguments=dict(arguments),
            success=success,
            error=error,
            duration_seconds=duration_seconds,
        )
        self.events.append(event)
        return event

    def to_dicts(self) -> list[dict[str, Any]]:
        """整份记录转成普通字典列表，方便存文件或打日志。"""
        return [event.to_dict() for event in self.events]

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)
