"""CallGraphIndex：在 ProjectIndex / SymbolIndex 之上建整份调用图。

两步，各管各的：

1. **调用点**（callgraph/python.py、callgraph/cpp.py）：每种语言一份提取器
   （和 index/symbols.py 的 EXTRACTORS 一个写法），只回答「第几行、在调谁」
   ——纯语法问题。以后加 Java / Go / Rust，就是往 SITE_EXTRACTORS 补一条。
2. **认主**：拿 SymbolIndex 里这个文件的符号，行号落在**最内层**哪个函数 /
   方法里，这条边就算谁的——语义问题。定义和行范围已经有一份实现（Python
   走 pyast、C/C++ 走 tools/cpp.py），这里不重算，所以各种语言共用同一套
   认主逻辑。

读文件走 SymbolIndex.read_verified：文件被改过 / 读不动就和符号索引一样整个
跳过——拿旧行号去建边，建出来的是错的。跳过原因和 SymbolIndex.skipped 里
写的是同一套（同一个 read_verified、同一种解析），这里不另记一份。

第一阶段只做离线索引和查询：不接 Agent Tool、不改 Agent Loop，也不碰
Agent / RepairLoop / Retriever / Sandbox。
"""

from collections.abc import Iterable
from pathlib import Path

from callgraph import cpp as cpp_calls
from callgraph import python as python_calls
from callgraph.base import DEFAULT_MAX_DEPTH, CallEdge, CallGraph, CallPath
from index.symbols import FUNCTION, METHOD, Symbol, SymbolIndex
from tools.base import ToolError

# 语言 id → 调用点提取器。加一门语言就是往这儿补一条（和 index/symbols.py
# 的 EXTRACTORS 同一个规矩）：提取器吃文件文本，交出 CallSite 列表。
SITE_EXTRACTORS = {
    "python": python_calls.extract_sites,
    "cpp": cpp_calls.extract_sites,
}

# 只有函数 / 方法才当调用方：类体里（成员默认值之类）的调用没有「哪个
# 函数调的」，模块级同理。
_CALLABLE_KINDS = frozenset({FUNCTION, METHOD})


class CallGraphIndex:
    """整份调用图。用法：建一个 → build() → 查 callers / callees。

        index = CallGraphIndex()                    # 默认整个项目
        index = CallGraphIndex(root=some_dir)       # 指定目录（测试里用）
        index.build()
        index.callees("Agent.run")                  # run 调了谁
        index.callers("helper")                     # 谁调了 helper
        index.find_path("A", "B")                   # A 到 B 的最短调用链
        index.edges(path="agent/agent.py")          # 原始边（带行号）
        index.symbol("Agent.run")                   # 对回符号索引里的 Symbol

    符号索引由谁管，和 SymbolIndex 接 project= 是同一条规矩：`symbols=`
    传一份现成的就归调用方管（build 不会替它刷新），`root=`（或默认）是
    自己建一份、每次 build() 都是全量刷新。build() 之前先看 built。

    **图跟着符号索引走**：build 的时候把当时那份符号索引的 `generation`
    记下来，之后谁（比如工具层的 refresh()）重扫了符号索引，`built` 就
    变回 False——旧图上那些行号已经对不上新代码了，宁可重建也不能接着用。
    要自己主动重建就调 `refresh()`（连符号索引和文件清单一起重扫）。
    """

    def __init__(
        self,
        symbols: SymbolIndex | None = None,
        root: Path | None = None,
    ) -> None:
        if symbols is not None and root is not None:
            raise ToolError("symbols 和 root 只能给一个：传了 symbols 就由调用方管")
        self.symbols = symbols if symbols is not None else SymbolIndex(root=root)
        self._own_symbols = symbols is None
        self._graph = CallGraph()
        self._built = False
        self._generation = 0
        # 建图时符号索引是第几版。对不上就说明图已经过期（见 built）
        self._symbols_generation = 0

    def build(self, *, force: bool = False) -> None:
        """全量重建：把已索引的文件逐个读回来，提取调用点、认到函数头上。

        `force=True`：连调用方管的符号索引也重扫（`refresh()` 走这条）。
        """
        if force or self._own_symbols or not self.symbols.built:
            self.symbols.build(force=force)

        # 每个文件的符号先按路径收好：认主是「行号落在哪个符号区间里」，
        # 文件扫到哪儿查哪儿，不用每次过一遍全部符号
        by_path: dict[str, list[Symbol]] = {}
        for symbol in self.symbols.symbols():
            by_path.setdefault(symbol.path, []).append(symbol)

        graph = CallGraph()
        for entry in self.symbols.project.files():
            extractor = SITE_EXTRACTORS.get(entry.language)
            if extractor is None:
                continue  # 这门语言还没有提取器：不是出错，是还没轮到

            text, reason = self.symbols.read_verified(entry)
            if reason is not None:
                continue  # 读不成 / 改过了：和符号索引一样整个跳过
            # read_verified 的约定：reason 为 None 时 text 一定有
            assert text is not None

            try:
                sites = extractor(text)
            except (SyntaxError, ValueError, RecursionError):
                continue  # 解析不了只影响这一个文件

            for site in sites:
                caller = _caller_at(by_path.get(entry.path, ()), site.line)
                if caller is None:
                    continue  # 不在任何函数 / 方法里：没有 caller，不记边
                graph.add(
                    CallEdge(
                        caller=caller.qualified,
                        callee=site.callee,
                        path=entry.path,
                        line=site.line,
                        language=entry.language,
                    )
                )

        # 全部扫完才换上新的：中途出意外的话，手上这份旧图还是完整的
        self._graph = graph
        self._built = True
        self._generation += 1
        self._symbols_generation = self.symbols.generation

    def refresh(self) -> None:
        """显式重建：连符号索引和文件清单一起重扫。

        文件增、删、改之后调它（或者调 SymbolIndex.refresh()——图会自动
        作废，下次 build() 再重建，两条路都是一个结果）。
        """
        self.build(force=True)

    @property
    def generation(self) -> int:
        """这份图换过几版：每建一次 +1，0 表示一版都没有。"""
        return self._generation

    @property
    def built(self) -> bool:
        """建过没有，而且是**对着当前这份符号索引**建的吗。

        没建过的图是空的，但那不等于「项目里没有调用」；符号索引被人重扫过
        （generation 变了）的图也算没建——边上的行号来自旧代码，接着用就是
        拿过期数据回答。调用方（工具层）看到 False 会重建，这正是想要的。
        """
        return self._built and self._symbols_generation == self.symbols.generation

    @property
    def graph(self) -> CallGraph:
        """底下的图。edges / callers / callees 都转发到它，拿它也行。"""
        return self._graph

    def edges(
        self,
        path: str | None = None,
        caller: str | None = None,
        callee: str | None = None,
    ) -> tuple[CallEdge, ...]:
        """按条件筛原始边（与的关系），都不给就是全部；带行号、路径、语言。"""
        return self._graph.edges(path=path, caller=caller, callee=callee)

    def callees(self, function: str) -> tuple[str, ...]:
        """某函数调用了谁：去重、排好序的名字。"""
        return self._graph.callees(function)

    def callers(self, function: str) -> tuple[str, ...]:
        """谁调用了某函数：去重、排好序的名字（调用方的限定名）。"""
        return self._graph.callers(function)

    def find_path(
        self, start: str, target: str, max_depth: int = DEFAULT_MAX_DEPTH
    ) -> CallPath | None:
        """start → target 的一条最短调用链（BFS）；没有就 None。

        只查已经建好的图，不重扫、不重建——要新数据先 build()。
        """
        return self._graph.find_path(start, target, max_depth=max_depth)

    def symbol(self, name: str) -> Symbol | None:
        """把一个名字对回符号索引里的 Symbol（简单名 / 限定名都认）。

        找不到返回 None——callee 里很多是库函数、宏、类型名，本来就不在
        这个项目的符号索引里，这不是错误。重名时给第一个，要全部就用
        self.symbols.symbols(name=...)。
        """
        found = self.symbols.symbols(name=name)
        return found[0] if found else None


def _caller_at(symbols: Iterable[Symbol], line: int) -> Symbol | None:
    """这一行落在哪个函数 / 方法里：最内层的那个。

    嵌套时取最里面（起点最靠后）：内层函数自己调用的行，往外层头上算就
    错了。类不参与——类体里的调用没有「哪个函数调的」。
    """
    best: Symbol | None = None
    for symbol in symbols:
        if symbol.kind not in _CALLABLE_KINDS:
            continue
        if symbol.start_line <= line <= symbol.end_line:
            if best is None or symbol.start_line > best.start_line:
                best = symbol
    return best
