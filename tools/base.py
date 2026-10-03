"""工具的基础抽象：统一名称、描述、参数声明与执行接口。"""

from abc import ABC, abstractmethod
from typing import Any


class ToolError(RuntimeError):
    """工具执行失败，或调用参数不合法。"""


# parameters 里声明的类型名 -> Python 类型
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _type_ok(value: Any, type_name: str) -> bool:
    """判断 value 是否匹配声明的类型名。"""
    if type_name == "boolean":
        return isinstance(value, bool)
    # bool 是 int 的子类，不应被当作 number / integer
    if isinstance(value, bool):
        return False
    if type_name == "number":
        return isinstance(value, (int, float))
    expected = _TYPE_MAP.get(type_name)
    return expected is not None and isinstance(value, expected)


class Tool(ABC):
    """所有工具的基类。

    子类声明 name / description / parameters，并实现 execute()；
    对外统一用 run() 调用：先校验参数，再交给 execute()；
    to_schema() 把上面的声明转成 LLM 的 tool calling 格式。
    """

    name: str = ""
    description: str = ""
    # 形如 {"expression": {"type": "string", "description": "...", "required": True}}
    parameters: dict[str, dict[str, Any]] = {}

    def run(self, **kwargs: Any) -> str:
        """校验参数后执行，返回文本结果。"""
        self.validate_args(kwargs)
        return self.execute(**kwargs)

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """真正的执行逻辑，由子类实现。调用到这里时参数已校验通过。"""

    def validate_args(self, kwargs: dict[str, Any]) -> None:
        """按 parameters 声明校验调用参数，不合法则抛 ToolError。"""
        for key in kwargs:
            if key not in self.parameters:
                available = sorted(self.parameters) or "无"
                raise ToolError(
                    f"工具 {self.name!r} 不接受参数 {key!r}，可用参数：{available}"
                )

        for param, spec in self.parameters.items():
            if param not in kwargs:
                if spec.get("required", True):
                    raise ToolError(f"工具 {self.name!r} 缺少必需参数 {param!r}")
                continue

            type_name = spec.get("type", "string")
            if type_name not in _TYPE_MAP:
                raise ToolError(
                    f"工具 {self.name!r} 的参数 {param!r} 声明了未知类型 {type_name!r}"
                )
            if not _type_ok(kwargs[param], type_name):
                raise ToolError(
                    f"工具 {self.name!r} 的参数 {param!r} 期望 {type_name}，"
                    f"实际是 {type(kwargs[param]).__name__}"
                )

    def to_schema(self) -> dict[str, Any]:
        """转成 OpenAI tool calling 要求的工具描述。"""
        if not self.name:
            raise ToolError(f"{type(self).__name__} 未声明 name，无法生成 schema")

        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for param, spec in self.parameters.items():
            # required 是本项目自己的约定，不属于 JSON Schema 的属性描述，要摘掉。
            properties[param] = {
                key: value for key, value in spec.items() if key != "required"
            }
            if spec.get("required", True):
                required.append(param)

        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:  # 没有必需参数时这个键应当缺席，而不是空列表
            parameters["required"] = required

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }
