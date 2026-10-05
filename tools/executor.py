"""统一的工具执行层：Agent 不直接碰 ToolRegistry，一律走这里。

调用链：Agent → ToolExecutor → ToolRegistry → Tool

把「执行」这件事的横切关注点集中到一处：
    - 工具按预期报错（ToolError）
    - 工具实现有 bug 抛出的意外异常
    - 工具没按要求返回 str
    - 执行耗时
以上都收敛成 ToolResult，Agent 只管把 ToolResult 转成回灌给模型的消息。

另外：如果给了 trace，每次调用都会往里记一条（工具名、参数、成败、耗时）。
记录是纯附加的，不影响返回值，也不影响工具怎么执行。

13.2 起还多一句「要开始跑了」的通知（trace.start_tool_call）：在工具真正
执行**之前**发出去，给流式接口实时推进度用。它同样只通知、记录不了什么。

14.3 起，成功的那次调用还顺手从返回正文里提一次位置（tools/locations.py，
只取 path/line/symbol/kind，正文不进 Trace）——工作台靠它画出「落在哪个
文件的哪一段」。提不出来就是空列表，不影响任何既有行为。
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tools import locations
from tools.base import ToolError
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    # 只在做类型检查时导入。运行时不导入，否则 tools 和 agent 会互相导入成环
    # （谁先被 import 决定成败，很脆）。执行层只调 trace.record_tool_call()，
    # 不需要真的认识 Trace 这个类。
    from agent.trace import Trace


@dataclass(frozen=True)
class ToolResult:
    """一次工具调用的统一结果。

    success 为真时看 content，为假时看 error。
    metadata 里固定带两个键：tool（工具名）和 duration_seconds（耗时秒数）。
    """

    success: bool
    content: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_observation(self) -> str:
        """转成回灌给 LLM 的那段文本：成功给结果，失败给「错误：...」。"""
        return self.content if self.success else f"错误：{self.error}"


class ToolExecutor:
    """统一执行工具，永不向外抛异常。"""

    def __init__(
        self, registry: ToolRegistry, trace: "Trace | None" = None
    ) -> None:
        self.registry = registry
        # 不传就是不记录，执行语义完全不变
        self.trace = trace

    def execute(self, name: str, **arguments: Any) -> ToolResult:
        """执行一个工具，无论成功失败都返回 ToolResult。"""
        self._notify_start(name, arguments)
        result = self._run(name, arguments)
        self._record(name, arguments, result)
        return result

    def _run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        try:
            content = self.registry.run(name, **arguments)
        except ToolError as exc:
            # 预期内的失败：工具不存在、参数不合法、工具自己拒绝执行
            return self._failed(name, started, str(exc))
        except Exception as exc:
            # 预期外的失败：工具实现里有 bug。也必须兜住，
            # 否则一个写坏的工具会让整个 Agent 崩掉。
            return self._failed(name, started, f"{type(exc).__name__}: {exc}")

        if not isinstance(content, str):
            # 工具返回了 None、数字之类的东西（多半是 execute() 忘了写 return）。
            # 在这儿拦下来：不然 content=None 会一路带到请求体里，
            # 报错的时候已经离现场很远了，很难查。
            return self._failed(
                name,
                started,
                f"工具 {name} 必须返回 str，实际返回 {type(content).__name__}",
            )

        return ToolResult(
            success=True, content=content, metadata=self._metadata(name, started)
        )

    def _notify_start(self, name: str, arguments: dict[str, Any]) -> None:
        """跑之前的通知，和 _record（跑完记一条）是两件事。

        参数不合法、工具不存在这类事发生在这之后——也就是说前端会先看到
        「开始跑 echo 了」，再看到「echo 失败」。这没错：模型确实要求调它了。
        """
        if self.trace is not None:
            self.trace.start_tool_call(name, arguments)

    def _record(
        self, name: str, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        """把这次调用记进 Trace。成功失败走同一条路，记录不会漏。

        失败的不提位置：错误文本里不会有可信的位置（见 tools/locations.py
        的白名单那条）。成功的那一下顺手提一次——**只留位置，不留正文**，
        正文在这里还是原样返回给 Agent 的，只是不进 Trace。
        """
        if self.trace is None:
            return
        self.trace.record_tool_call(
            tool=name,
            arguments=arguments,
            success=result.success,
            error=result.error,
            duration_seconds=result.metadata["duration_seconds"],
            locations=self._locations(name, result),
        )

    @staticmethod
    def _locations(name: str, result: ToolResult) -> list[dict[str, Any]]:
        """从返回正文里提出位置信息；提不出来（或这层出了岔子）就是空。

        失败要兜住：执行层答应过永不向外抛异常，提位置是纯附加的活儿，
        不能因为它把正常的一轮任务带崩。
        """
        if not result.success:
            return []
        try:
            return locations.extract(name, result.content)
        except Exception:
            return []

    def _failed(self, name: str, started: float, message: str) -> ToolResult:
        return ToolResult(
            success=False, error=message, metadata=self._metadata(name, started)
        )

    @staticmethod
    def _metadata(name: str, started: float) -> dict[str, Any]:
        return {"tool": name, "duration_seconds": time.perf_counter() - started}
