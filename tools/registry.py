"""工具注册表：按名称注册、查询、列表和执行工具。"""

from typing import Any

from tools.base import Tool, ToolError


class ToolRegistry:
    """持有若干 Tool 并按名称索引。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """注册一个工具；名称缺失或重复则抛 ToolError。"""
        if not tool.name:
            raise ToolError(f"{type(tool).__name__} 未声明 name，无法注册")
        if tool.name in self._tools:
            raise ToolError(f"工具名重复：{tool.name!r} 已注册")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """按名称取工具；未注册则抛 ToolError。"""
        try:
            return self._tools[name]
        except KeyError:
            available = sorted(self._tools) or "无"
            raise ToolError(f"未注册的工具：{name!r}。已注册：{available}") from None

    def run(self, name: str, **kwargs: Any) -> str:
        """按名称执行工具，返回文本结果。"""
        return self.get(name).run(**kwargs)

    def list(self) -> list[Tool]:
        """返回全部工具。"""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
