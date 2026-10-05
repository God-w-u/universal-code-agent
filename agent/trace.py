"""Agent 执行过程的记录（Trace）：一次 run() 调了哪些工具、什么参数、成没成、多久。

刻意保持简单：只有一种事件（工具调用）和一个容器，不做通用事件系统。

13.2 加了一对可选的观察者回调（on_tool_start / on_tool_end）：HTTP 的流式
接口靠它在「工具要开始跑」和「工具跑完了」两个时刻拿到通知。观察者只管
通知，不进 events、不碰执行——Trace 仍然是那份「跑完了的记录」。

13.5 又把 LLM 调用接进同一套观察者机制：on_llm_start / on_llm_end 一对
（llm/client.py 在每一跳的前后调 start_llm_call / record_llm_call）。和工具
一样，**这两条通知不进 events**——events 里始终只有「跑完的工具调用」，
几十处断言按这个形状写的；LLM 的统计单独攒在 llm_calls 里（只有数字、
model、usage、错误摘要，没有 prompt、没有回复正文、没有 key）。

14.3 给 TraceEvent 加了 locations（位置信息），给前端画「这次调用落在哪个
文件的哪一段」。**正文照旧不进 Trace**：extract 只取 path/line/symbol/kind
这几个字段（见 tools/locations.py 的分寸），提不到就是空列表。

注意：本模块只依赖标准库，不导入 tools / llm / agent 里的任何东西（llm
只在 TYPE_CHECKING 里出现，运行时不存在这条依赖）。tools/executor.py 调用
这里的 Trace，反过来这里不依赖它，避免形成导入环。
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 只为类型检查：运行时不 import，agent 和 llm 之间没有运行期依赖
    from llm.client import LLMCallStats

# 目前只有这一种事件。留着这个字段是为了以后要加别的类型时不用改结构。
TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class TraceEvent:
    """一条执行记录。

    locations（14.3 追加的最后一个字段）：这次调用落在哪些文件的哪几行，
    由 tools/locations.py 从工具返回正文里**只取位置**得来（正文本身照旧
    不进 Trace）。默认空列表——失败、认不出来、白名单外的工具都是空。
    加在最后是为了不打乱既有位置参数的调用。
    """

    type: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str = ""
    duration_seconds: float = 0.0
    locations: list[dict[str, Any]] = field(default_factory=list)

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
            "locations": self.locations,
        }


class Trace:
    """一次 run() 产生的全部执行记录。

    由 Agent 每轮新建一个，不同轮之间互不影响。

    on_tool_start / on_tool_end（13.2，都可以不给）：观察者回调，签名分别是
    `(tool, arguments)` 和 `(event)`。前者在工具**执行之前**调，后者在记下
    这条之后调。两条注意事项：

    - **不记进 events**：events 里始终只有「跑完了」的那些条；开始通知不是
      一条记录，不给它添一个半成品事件。
    - **观察者抛错不影响任何事**：它多半在往网络连接里写，对面断了是常事，
      不能因为没人接了就把这一轮任务带崩。要停由观察者自己停——api/sse.py
      就是这么做的，写失败就把自己关掉，后面的通知直接跳过。

    on_llm_start / on_llm_end（13.5，都可以不给）：同一套机制用在模型调用上。
    签名分别是 `(model, attempt, max_attempts)` 和 `(stats)`（stats 见
    llm/client.py 的 LLMCallStats）。一次调用重试了几次就有几对通知——每
    一跳一对，最后一跳的 stats.attempt_count 就是总共试了几次。llm_calls
    攒着收到的全部 stats，方便跑完回看（同样只有数字，没有正文）。
    """

    def __init__(
        self,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_end: Callable[[TraceEvent], None] | None = None,
        on_llm_start: Callable[[str, int, int], None] | None = None,
        on_llm_end: Callable[["LLMCallStats"], None] | None = None,
    ) -> None:
        self.events: list[TraceEvent] = []
        # 这一轮收到的 LLM 调用统计（不进 events，见类 docstring）
        self.llm_calls: list["LLMCallStats"] = []
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end
        self._on_llm_start = on_llm_start
        self._on_llm_end = on_llm_end

    def start_tool_call(self, tool: str, arguments: dict[str, Any]) -> None:
        """通知一声「要开始调这个工具了」。不产生事件，只给观察者。"""
        self._notify(self._on_tool_start, tool, dict(arguments))

    def record_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        success: bool,
        error: str = "",
        duration_seconds: float = 0.0,
        locations: list[dict[str, Any]] | None = None,
    ) -> TraceEvent:
        """记下一次工具调用，返回刚记下的那条。

        locations 由调用方（tools/executor.py）从返回正文里提好再给进来，
        这一层不认识工具、也不碰正文——给什么记什么，顺手复制一份。
        """
        event = TraceEvent(
            type=TOOL_CALL,
            tool=tool,
            # 复制一份：记录的是调用当时的样子，之后谁改了原字典都不影响这里
            arguments=dict(arguments),
            success=success,
            error=error,
            duration_seconds=duration_seconds,
            locations=[dict(item) for item in (locations or [])],
        )
        self.events.append(event)
        self._notify(self._on_tool_end, event)
        return event

    def start_llm_call(self, model: str, attempt: int, max_attempts: int) -> None:
        """通知一声「要问模型了」。一次调用的每一跳都会来一下。

        和 start_tool_call 一个套路：只给观察者，不往 events 里塞半成品。
        """
        self._notify(self._on_llm_start, model, attempt, max_attempts)

    def record_llm_call(self, stats: "LLMCallStats") -> None:
        """收下一次模型调用的统计（成功、失败、重试各一跳都会来）。

        stats 进 llm_calls 留档，同时通知观察者——llm_end 事件就是它。
        """
        self.llm_calls.append(stats)
        self._notify(self._on_llm_end, stats)

    @staticmethod
    def _notify(callback: Callable[..., None] | None, *args: Any) -> None:
        """叫一声观察者；它炸了就当没发生过（见类 docstring）。"""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            # 故意的：观察者是外人（网络连接之类），它的错不归 Trace 管，
            # 更不能顺着 ToolExecutor 的调用栈把这一轮任务变成失败
            pass

    def to_dicts(self) -> list[dict[str, Any]]:
        """整份记录转成普通字典列表，方便存文件或打日志。"""
        return [event.to_dict() for event in self.events]

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)
