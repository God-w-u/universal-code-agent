"""从 Python 源码里提取结构：class、def、async def。

用标准库 ast——Python 的语法树是现成的，名字和行号（含结束行）都是准的，
不用像 C/C++（tools/cpp.py）那样自己逐行扫。两个模块是一对：结构提取
按语言分家，谁都能用——tools/code_summary.py 拿它出摘要，index/symbols.py
拿它建符号索引。**只有这一份实现**，两边不会各认各的。

提取的是定义，不是内容：函数体里的一律不算（里面是代码，不是结构）。
"""

import ast
from typing import NamedTuple


class Definition(NamedTuple):
    """一条定义。行号都是 1 起、闭区间，直接对应源文件的行。

    - `kind`：class / function / method。直接写在 class 体里的函数是 method，
      其余都是 function（函数里再套函数也是 function）。
    - `start`：定义的第一行。**带装饰器就从装饰器那一行算起**——切块的时候
      `@app.route(...)` 属于这个函数，切掉就不是完整的定义了。没有装饰器
      就是 def / class 那一行。
    - `end`：定义的最后一行（含整个函数体）。
    - `depth`：嵌套层级，0 是顶层。只有 class / def 会让它加一；
      if / try / with 只是把定义包在里面，不算一层。
    - `parent`：父级限定名（`Agent.Inner`），顶层是 None。用点连接——
      统一写法，和 C++ 的 `::` 无关。
    - `is_async`：是不是 async def。
    """

    kind: str
    name: str
    start: int
    end: int
    depth: int
    parent: str | None
    is_async: bool


# ast 把 except / case 的子句建模成独立节点，不是 Try / Match 的直接子节点，
# 所以要单独列出来，否则 try/except 里包着的定义会被漏掉
_CLAUSES = (ast.ExceptHandler, ast.match_case)


def extract(text: str) -> list[Definition]:
    """提取一个 Python 文件的结构，按出现顺序返回。

    解析不了（语法错、ast 本身拒绝的输入、嵌套过深）会抛
    SyntaxError / ValueError / RecursionError，由调用方决定是「跳过这个文件」
    还是整件事失败——这里不替它决定。
    """
    tree = ast.parse(text)

    found: list[Definition] = []
    _walk(tree, found, depth=0, parent=None, in_class=False)
    return found


def _walk(
    node: ast.AST,
    found: list[Definition],
    depth: int,
    parent: str | None,
    in_class: bool,
) -> None:
    """深度优先收集定义。in_class 说的是「紧挨着的上一层定义是不是 class」。"""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            is_class = isinstance(child, ast.ClassDef)
            if is_class:
                kind = "class"
            elif in_class:
                # 类体里直接写的函数是方法；方法里套的函数不是
                kind = "method"
            else:
                kind = "function"

            qualified = f"{parent}.{child.name}" if parent else child.name
            found.append(
                Definition(
                    kind=kind,
                    name=child.name,
                    start=_start(child),
                    end=child.end_lineno or child.lineno,
                    depth=depth,
                    parent=parent,
                    is_async=isinstance(child, ast.AsyncFunctionDef),
                )
            )
            _walk(child, found, depth + 1, qualified, is_class)
        elif isinstance(child, ast.stmt) or isinstance(child, _CLAUSES):
            # 其它语句（if / try / for / with ...）本身不是定义，但里面可能
            # 包着定义，所以继续往下找：层级不加，父亲也不变
            _walk(child, found, depth, parent, in_class)


def _start(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """定义从第几行开始：有装饰器就是最上面那个装饰器所在的行。"""
    if node.decorator_list:
        return min(item.lineno for item in node.decorator_list)
    return node.lineno
