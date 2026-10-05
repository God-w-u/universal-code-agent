"""Python 的调用点提取：走一遍 ast，只回答「第几行在调谁」。

定义、父级、限定名都归 tools/pyast.py 和 SymbolIndex 那一套（一份实现），
这里不重复；这里只做一个新问题——把 ast 里的 Call 节点捞出来。

认哪些：

- `foo()` → "foo"
- `obj.foo()` / `self.foo()` / `a.b.foo()` → 整条链原样记（"obj.foo"）
- 链子读不全的（`f().foo()`、`x[0].foo()`）只记最后一段（"foo"）：确实
  在调一个叫 foo 的东西，只是对象是谁说不清
- `f()(x)`、`handlers[k](x)`：调的是「算出来的东西」，没有名字，这一条
  不记；里面包着的真实调用（`get()`）照记

**装饰器和默认参数里的调用不记**：它们在**外层作用域**执行，可行号落在
被定义的那个函数的范围里，按行认主会算到被装饰的函数头上——宁可漏一条，
也不记一条错的。lambda 体、推导式里的调用照记（它们本来就跑在所在函数里）。
"""

import ast

from callgraph.base import CallSite


def extract_sites(text: str) -> list[CallSite]:
    """一个 Python 文件里的调用点，按行号顺序返回。

    解析不了（语法错）就照原样抛 SyntaxError——跳过这个文件是调用方
    （CallGraphIndex）的事，这里不吞。
    """
    visitor = _Calls()
    visitor.visit(ast.parse(text))
    # 按行号排一下（稳定排序）：同一行上的嵌套调用保持遍历顺序，跨行的
    # 一律从头到尾——和 cpp.py 交出来的顺序一个规矩
    return sorted(visitor.sites, key=lambda site: site.line)


class _Calls(ast.NodeVisitor):
    """收集调用点。进签名区域（装饰器 / 默认值 / 注解）的调用不算。"""

    def __init__(self) -> None:
        self.sites: list[CallSite] = []
        self._in_signature = 0

    def visit_Call(self, node: ast.Call) -> None:
        if not self._in_signature:
            name = _callee(node.func)
            if name:
                self.sites.append(CallSite(node.lineno, name))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition(node)

    def _visit_definition(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        # 签名区域整体跳过（见模块开头），函数体正常走。
        # node.args 里带着默认值和参数注解，走它一个就够了。
        self._in_signature += 1
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._in_signature -= 1

        for statement in node.body:
            self.visit(statement)


def _callee(node: ast.expr) -> str | None:
    """调用表达式里被调的那个名字；认不出来返回 None。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _dotted(node: ast.expr) -> str | None:
    """把纯名字 / 属性链读成点连接的名字；读不出来返回 None。

    读得出来的只有 `a.b.c` 这种一路 Name + Attribute 的；`f().b`、
    `x[0].b` 里的对象是算出来的，链子到那儿就断了。
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None
