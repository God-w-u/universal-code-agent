"""Call Graph 的数据结构：一条调用边 + 装着边的图（Phase 12.1）。

语言无关是这一层的硬要求：CallEdge 里全是字符串和数字，Python 的 ast
节点、C/C++ 的 cpp.Span 都进不来。以后加 Java / Go / Rust，提取器交出
同样形状的边就行，图和查询一个字都不用改。

**名字的约定**（和 SymbolIndex 对齐）：

- caller 是**限定名**、点连接（`Agent.run`、`app.Renderer.draw`）——
  它本来就是 SymbolIndex 里的 Symbol.qualified，两边一定对得上。
- callee 是**调用点原样记下来的名字**、也是点连接（`foo`、`obj.foo`、
  `ns.foo`）。不做类型推导：`obj.foo` 里的 obj 只是个名字，没人知道
  它是哪个类，所以拿**简单名**查询时「最后一段」也认——`callers("foo")`
  会把上面三种都算上；带点的限定名则精确匹配（见 _matches）。

**两个近似**：

- 同一行调同一个名字两次（`f(); f();`）只记一条边——这一阶段回答的是
  「谁调用了谁」，不是调用了几次；
- 重名不去猜：`Agent.run` 和 `Cli.run` 都叫 run，用简单名查询会把两边
  的边都算上；要精确就给限定名。
"""

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class CallEdge:
    """一条调用边：caller 在 path 的第 line 行调用了 callee。

    字段全是值，不指向别处：

    - `caller`：调用方的限定名（和 Symbol.qualified 一个写法）。
    - `callee`：被调用方的名字，调用点怎么写就怎么记（点连接）。
    - `path`：项目相对路径，和 IndexedFile.path 一个写法。
    - `line`：调用点所在行（1 起），拿它去 read_file 能对上。
    - `language`：语言 id（python / cpp），和 index.languages 一套。
    """

    caller: str
    callee: str
    path: str
    line: int
    language: str

    @property
    def key(self) -> tuple[str, str, str, int]:
        """去重用的身份：同一个文件里、同一行、同一个 caller 调同一个
        callee 算同一条边。

        行号在内，所以第 3 行和第 9 行各调一次是两条——它们本来就不是
        「重复」；同一行调两次同一个名字才是一条。
        """
        return (self.path, self.caller, self.callee, self.line)


class CallSite(NamedTuple):
    """提取器交出来的半成品：第几行、在调谁。

    缺 caller / path / language——那三样由索引统一补。「这一行属于哪个
    函数」不在提取器里回答：那是拿 SymbolIndex 的行区间去认的（见
    callgraph/index.py），各语言不用各算各的。
    """

    line: int
    callee: str


# 调用链的默认深度上限（跳数）：8 跳够覆盖常见的分层调用，再深多半是
# 顺着框架代码一路钻下去了。工具那边另有硬上限（20），这里只是默认值。
DEFAULT_MAX_DEPTH = 8


@dataclass(frozen=True)
class CallPath:
    """一条调用链：从起点一路沿着 callee 走到终点。

    - `nodes`：沿途的节点名，nodes[0] 是起点、nodes[-1] 是终点。
    - `edges`：每一步的那条边，edges[i] 是 nodes[i] → nodes[i+1] 的调用。

    节点名就是图里存的写法：起点是 caller 的限定名，之后每一步是调用点
    原样记下来的 callee 名（`foo`、`self.step`）。条目里不带 Symbol——
    定位要用 SymbolIndex，那是上面一层的事（CallGraphIndex.symbol）。
    """

    nodes: tuple[str, ...]
    edges: tuple[CallEdge, ...]

    def __post_init__(self) -> None:
        # 形状对不上就是构造错了：多一条边少一个节点，渲染出来会骗人
        if len(self.nodes) != len(self.edges) + 1:
            raise ValueError("调用链的节点数必须是边数 + 1")

    @property
    def length(self) -> int:
        """跳数：A→B 是 1，起点和终点是同一个函数是 0。"""
        return len(self.edges)


def _matches(stored: str, query: str) -> bool:
    """查询名字对不对得上：全名相等；查询是简单名时，再按最后一段比。

    不做类型推导，所以**简单名**会命中所有同名函数（`run` 命中
    `Agent.run`、`Cli.run`）——这是躲不掉的近似，要精确就给限定名。
    **带点的限定名要的是精确**：`Alpha.go` 不该把 `Beta.go` 的边也算上。
    （12.2 的测试发现这里原先对限定名也按最后一段比，越界匹配。）
    空名字谁都对不上（调用点和符号本来就不会是空名字）。
    """
    if stored == query:
        return True
    if not query or not stored:
        return False
    if "." in query:
        return False
    return stored.rsplit(".", 1)[-1] == query


def _edge_order(edge: CallEdge) -> tuple[str, int, str, str]:
    """边的排序键：路径、行号、caller、callee——同一份图跑两次，顺序一样。"""
    return (edge.path, edge.line, edge.caller, edge.callee)


class CallGraph:
    """调用边的容器 + 两个方向的查询 + 一条链的搜索。

        graph = CallGraph()
        graph.add(CallEdge(...))       # 重复的边进不来（返回 False）
        graph.callees("Agent.run")     # run 调了谁
        graph.callers("helper")        # 谁调了 helper
        graph.edges(path="a.py")       # 原始边（带行号、路径、语言）
        graph.find_path("A", "B")      # A 到 B 的最短调用链（BFS）

    查询返回的都是**排好序的名字元组**：同一个名字有多条边也只出现一次
    （要行号就去 edges() 里筛）。排序是为了结果稳定——建两次图、换个
    遍历顺序，输出都一样。
    """

    def __init__(self) -> None:
        self._edges: list[CallEdge] = []
        self._keys: set[tuple[str, str, str, int]] = set()

    def add(self, edge: CallEdge) -> bool:
        """记一条边；这条已经在了就返回 False（不重复记）。"""
        if not isinstance(edge, CallEdge):
            raise TypeError(f"add 需要 CallEdge，收到 {type(edge).__name__}")
        if edge.key in self._keys:
            return False
        self._keys.add(edge.key)
        self._edges.append(edge)
        return True

    def edges(
        self,
        path: str | None = None,
        caller: str | None = None,
        callee: str | None = None,
    ) -> tuple[CallEdge, ...]:
        """按条件筛原始边，条件是「与」的关系，都不给就是全部。

        path 是全路径；caller / callee 的比对规则见 _matches（简单名和
        限定名都认）。结果按路径、行号、名字排好，位置稳定。
        """
        found = [
            edge
            for edge in self._edges
            if (path is None or edge.path == path)
            and (caller is None or _matches(edge.caller, caller))
            and (callee is None or _matches(edge.callee, callee))
        ]
        return tuple(sorted(found, key=_edge_order))

    def callees(self, function: str) -> tuple[str, ...]:
        """某函数调用了谁：去重、排好序的名字。

        简单名会把这些函数的边都算上（`run` 命中 `Agent.run` 和
        `Cli.run`）——不做类型推导，这一点躲不掉，要精确就给限定名。
        """
        return tuple(
            sorted(
                {
                    edge.callee
                    for edge in self._edges
                    if _matches(edge.caller, function)
                }
            )
        )

    def callers(self, function: str) -> tuple[str, ...]:
        """谁调用了某函数：去重、排好序的名字（调用方的限定名）。"""
        return tuple(
            sorted(
                {
                    edge.caller
                    for edge in self._edges
                    if _matches(edge.callee, function)
                }
            )
        )

    def find_path(
        self, start: str, target: str, max_depth: int = DEFAULT_MAX_DEPTH
    ) -> CallPath | None:
        """沿 callee 方向找一条 start → target 的最短调用链；没有就 None。

        BFS 逐层展开，所以第一次撞上 target 的那条就是最短的；同一层里
        按节点名、边的顺序定先后，同一张图跑多少次结果都一样。走过的节点
        记在 parents 里不再走——A→B→C→A 这种环转不完的问题就断在这。

        `max_depth` 是跳数上限：1 只找直接调用，2 允许中间隔一个……
        起点给**简单名**时所有同名的函数都算起点（和 callers / callees
        一个规矩）；带点的限定名精确匹配。起点和终点是同一个函数时给一条
        零跳的链（名字在图里压根不出现的话返回 None，不是零跳）。

        「中间这一步」按全名精确展开：起点用简单名是靠 _matches 找出来的，
        走到中间就不能再按最后一段比了，否则会把同名函数的边算到这一步头上。
        代价是经对象调的中间跳接不下去：`self.helper` 这步在图上挂在
        `Obj.helper` 名下，链就断在这儿——宁可说没找到，也不猜 self 是谁。
        """
        if not start or not target:
            return None
        if _matches(start, target) or _matches(target, start):
            # 起终点是同一个函数：零跳的链。名字对不上图里任何东西时不算数
            return CallPath(nodes=(start,), edges=()) if self._mentions(start) else None

        outgoing: dict[str, list[CallEdge]] = {}
        for edge in sorted(self._edges, key=_edge_order):
            outgoing.setdefault(edge.caller, []).append(edge)

        frontier = sorted(
            {edge.caller for edge in self._edges if _matches(edge.caller, start)}
        )
        # parents 既是「从哪儿来」也是 visited：建好链之后照着它往回走
        parents: dict[str, tuple[str, CallEdge] | None] = {
            node: None for node in frontier
        }
        depth = 0
        while frontier and depth < max_depth:
            following: list[str] = []
            for node in frontier:
                for edge in outgoing.get(node, ()):
                    child = edge.callee
                    if child in parents:
                        continue  # 走过的不再走：环就断在这
                    parents[child] = (node, edge)
                    if _matches(child, target):
                        return _build_path(parents, child)
                    following.append(child)
            frontier = following
            depth += 1
        return None

    def _mentions(self, name: str) -> bool:
        """图里有没有名字对得上这个查询的节点（caller 或 callee 都算）。"""
        return any(
            _matches(edge.caller, name) or _matches(edge.callee, name)
            for edge in self._edges
        )

    def __len__(self) -> int:
        return len(self._edges)


def _build_path(
    parents: dict[str, tuple[str, CallEdge] | None], node: str
) -> CallPath:
    """顺着 parents 从终点往回走，翻过来就是起点到终点的链。"""
    nodes = [node]
    edges: list[CallEdge] = []
    step = parents[node]
    while step is not None:
        node, edge = step
        nodes.append(node)
        edges.append(edge)
        step = parents[node]
    nodes.reverse()
    edges.reverse()
    return CallPath(nodes=tuple(nodes), edges=tuple(edges))
