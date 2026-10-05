"""示例工具：计算数学表达式。

用 AST 白名单求值，不使用 eval()——表达式后续可能由 LLM 产生，属于不可信输入。
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

from tools.base import Tool, ToolError

# 幂运算的指数上限，避免 9**9**9 这类表达式把进程挂死。
_MAX_POW_EXPONENT = 1000

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # 只允许数字字面量，排除字符串、None、布尔等
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"表达式里只允许数字，收到 {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        func = _BIN_OPS.get(type(node.op))
        if func is None:
            raise ToolError(f"不支持的运算符：{type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT and abs(left) > 1:
            raise ToolError(f"指数过大（{right}），已拒绝执行")
        try:
            return func(left, right)
        except ZeroDivisionError:
            raise ToolError("除数不能为 0") from None

    if isinstance(node, ast.UnaryOp):
        # 单独一个名字：和上面二元那张表取出来的可调用对象不是一个签名
        # （一个收两个参数、一个收一个），共用一个变量类型对不上
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ToolError(f"不支持的一元运算符：{type(node.op).__name__}")
        return unary(_eval_node(node.operand))

    # 变量、函数调用、属性访问等一律拒绝
    raise ToolError(f"表达式里不允许出现 {type(node).__name__}")


def evaluate(expression: str) -> float:
    """计算表达式并返回数值结果。"""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误：{exc.msg}") from exc
    return _eval_node(tree)


class CalculatorTool(Tool):
    """计算数学表达式。"""

    name = "calculator"
    description = "计算一个数学表达式，支持 + - * / // % ** 和括号。"
    parameters = {
        "expression": {
            "type": "string",
            "description": "要计算的表达式，例如 (1+2)*3",
            "required": True,
        },
    }

    def execute(self, expression: str) -> str:
        result = evaluate(expression.strip())
        # 5.0 这种整数结果按 5 输出，读起来更自然
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
