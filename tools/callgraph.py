"""工具：查一个函数的调用关系（谁调它 / 它调谁 / 怎么走到另一个函数），
只给位置，不给正文。

调用图本身全在 callgraph/ 那条线里（Phase 12.1 建图，12.3 加调用链）：
CallGraphIndex 建图，CallGraph 查 callers / callees / find_path。这里一件
新活都不干——查询转过去，再把名字对回符号索引，凑出一行位置信息。查询
逻辑一行都不复制，也不自己解析源码：那样就会出现两份「谁调用了谁」，
迟早对不上。

和 search_code_chunks 同一个分寸：**不返回代码正文**，只给
`相对路径:起止行号` + 名字 + 类型，模型要读原文就拿位置去 read_file——
省上下文，而且它读到的是磁盘上当前的代码，不是索引里那份快照。

图是懒建的、建一次缓存：第一次真查才扫项目，之后同一个工具实例直接用
（12.2 的「一次 Agent 任务里允许缓存」）。代价是任务中途改了文件的话，
缓存里的行号可能已经偏了——read_file 读回来的是当前代码，一眼能对上；
要立刻对齐就调 `refresh()`（13.6），它把整张图按当前代码重建。图自己也
盯着符号索引的版本：别处刷新过符号索引，这里的图就作废、下次查询重建。

多个会话共用同一个工具实例（13.3）：懒建那一下用自己的小锁护着，
同时进来的第一次查询不会各扫一遍项目。
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from tools.base import Tool, ToolError
from tools.paths import check_root

if TYPE_CHECKING:
    # 只在做类型检查时导入，运行时不导——callgraph 自己也 import tools.base，
    # 模块顶层真去 import 会和 tools/__init__.py 撞成循环（见 __init__）。
    from callgraph import CallEdge, CallPath
    from index.symbols import Symbol

# 支持的三种查法：谁调用了它 / 它调用了谁 / 它到 target 怎么走
_DIRECTIONS = ("callers", "callees", "path")


class CallGraphTool(Tool):
    """查一个函数的调用方 / 被调方 / 到另一个函数的调用链，返回名字和位置。"""

    name = "call_graph"
    description = (
        "查一个函数的调用关系：direction=callers 查「谁调用了它」，callees 查"
        "「它调用了谁」，path 查「从 function 怎么走到 target」（要一起给 target，"
        "BFS 找最短的一条调用链，最多往下 max_depth 层，省略默认 8 层）。"
        "path 只顺着源码里写出来的调用点走：链中间那一步写成 self.x / obj.x 时"
        "就接不下去了（对象是谁不做类型推导），这种用 callers / callees 分头查。"
        "每条给名字和位置（相对路径:起止行号、类型），不返回代码正文——"
        "要读代码拿位置去 read_file。"
        "function 简单名（run）和限定名（Agent.run）都认：限定名精确匹配，"
        "简单名是把网撒大——同名函数的边全算上。调用点经对象写的（agent.run()、"
        "self.run()）记下来的是对象名，限定名对不上，这时用简单名再查一次。"
        "它认的是源码里看得出来的调用点，宏、函数指针、重载这些认不出来，"
        "「没找到」只代表没找到。"
    )
    parameters = {
        "function": {
            "type": "string",
            "description": "函数名：简单名（run）或限定名（Agent.run）",
            "required": True,
        },
        "direction": {
            "type": "string",
            "description": (
                "callers（谁调用了它）、callees（它调用了谁）或 path（它到 target 的调用链）"
            ),
            "required": True,
        },
        "target": {
            "type": "string",
            "description": "direction=path 时的终点函数；查 callers / callees 时不用给",
            "required": False,
        },
        "top_k": {
            "type": "integer",
            "description": "查 callers / callees 时最多返回几条，省略默认 10 条",
            "required": False,
        },
        "max_depth": {
            "type": "integer",
            "description": "direction=path 时最多往下找几层调用，省略默认 8，上限 20",
            "required": False,
        },
    }

    # top_k / max_depth 的上限：一条结果一行，50 行够挑了；调用链 20 跳
    # 已经是在顺着框架代码一路钻了，再深不如直接搜目标函数
    default_top_k = 10
    max_top_k = 50
    max_depth_limit = 20

    def __init__(self, root: Path | None = None) -> None:
        """root：项目根；省略就是项目根目录。构造只校验路径，不扫项目。"""
        self.root = check_root(root)

        # 到这儿才 import callgraph，而不是模块顶层：tools/__init__.py 会把这个
        # 模块连同所有工具一起拉起来，而 callgraph 自己也 import tools.base——
        # 顶层 import 会让「先 import tools」的那条路撞上只加载了一半的
        # callgraph（循环 import）。走到构造函数时两边都已加载完，怎么进来都不会炸。
        from callgraph import DEFAULT_MAX_DEPTH, CallGraphIndex

        self.index = CallGraphIndex(root=self.root)
        self.default_max_depth = DEFAULT_MAX_DEPTH
        # 13.3：工具实例被各会话共用，第一次查询可能同时进来——建图只做
        # 一次（见 _ensure_index）
        self._build_lock = threading.Lock()

    def execute(
        self,
        function: str,
        direction: str,
        target: str | None = None,
        top_k: int | None = None,
        max_depth: int | None = None,
    ) -> str:
        """查一次：callers / callees 给最多 top_k 行，path 给一条调用链。"""
        name = function.strip()
        if not name:
            raise ToolError("函数名不能为空")

        asked_direction = direction.strip().lower()
        if asked_direction not in _DIRECTIONS:
            raise ToolError(
                f"direction 只能是 callers（谁调用了它）、callees（它调用了谁）"
                f"或 path（它到 target 的调用链），收到 {direction!r}"
            )

        if asked_direction == "path":
            return self._execute_path(name, target, max_depth)

        asked = self.default_top_k if top_k is None else top_k
        if asked < 1:
            raise ToolError(f"top_k 至少为 1，收到 {top_k}")

        self._ensure_index()
        names = (
            self.index.callers(name)
            if asked_direction == "callers"
            else self.index.callees(name)
        )
        if not names:
            return self._empty(name, asked_direction)

        # 先截断再渲染：一条结果一行，没必要为了丢掉的行号先拼一次字符串
        shown = names[: min(asked, self.max_top_k)]
        text = self._render(name, asked_direction, shown, len(names))
        if len(names) > len(shown):
            # 截断了就得说：不说的话模型分不清「就这么多」和「只给了前几条」
            text += f"\n…（共 {len(names)} 个，只显示前 {len(shown)} 个）"
        if asked > self.max_top_k:
            text += f"\n…（top_k 最多给到 {self.max_top_k}）"
        return text

    def refresh(self) -> None:
        """显式重建调用图：文件增删改之后调它，下一次查询就是最新的图。

        全量重建，连符号索引和文件清单一起重扫——行号、边、名字全是当前
        代码的。和懒建共用一把锁：多会话同用一个工具实例时，刷新和别人
        的第一次查询不会撞在一起。建图失败照样包成 ToolError。

        索引别处（直接动 CallGraphIndex 的调用方）被刷新时不用管：图自己
        认得出符号索引换过版本，`built` 变回 False，下一次查询会重建。
        """
        with self._build_lock:
            self._build()

    def _ensure_index(self) -> None:
        """第一次真查时建图，之后直接用——一次任务里只扫一遍项目。

        13.3 起同一个工具实例可能被多个会话同时用：先无锁快查一眼，没建
        才进锁；进锁后再查一次（别人可能刚建完），两道都过了才真去建。
        CallGraphIndex.build() 本来就是「全扫完才换上新图」的原子换，加锁
        只是省掉并发时的重复扫描，不是修正确性。

        13.6 起 `built` 还认版本：符号索引被人刷新过（generation 变了）时
        这里的 False 会让图重建，不会拿着旧行号接着查。

        建图失败包成 ToolError：工具报错走的是「错误文本回灌给模型」那条
        路，Agent 不会因为调用图建不起来而崩。
        """
        if self.index.built:
            return
        with self._build_lock:
            if self.index.built:
                return
            self._build()

    def _build(self) -> None:
        """真去建图（调用方负责拿着锁），失败包成 ToolError。"""
        try:
            self.index.build()
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"调用图建不起来：{type(exc).__name__}: {exc}") from exc

    def _render(
        self,
        function: str,
        direction: str,
        names: tuple[str, ...],
        total: int,
    ) -> str:
        """每条一行：位置 + 名字 + 类型（对不上符号就给调用点）。"""
        if direction == "callers":
            header = f"{function} 的调用者（共 {total} 个）："
        else:
            header = f"{function} 调用了谁（共 {total} 个）："
        lines = [header]
        for position, name in enumerate(names, start=1):
            lines.append(f"{position}. {self._describe(name, function, direction)}")
        return "\n".join(lines)

    def _execute_path(
        self, start: str, target: str | None, max_depth: int | None
    ) -> str:
        """path 模式：start 沿调用方向走到 target 的最短一条链。"""
        if target is None or not target.strip():
            raise ToolError(
                'direction=path 要一起给 target：function="A", direction="path", '
                'target="B"'
            )
        goal = target.strip()

        asked = self.default_max_depth if max_depth is None else max_depth
        if asked < 1:
            raise ToolError(f"max_depth 至少为 1，收到 {max_depth}")

        self._ensure_index()
        depth = min(asked, self.max_depth_limit)
        path = self.index.find_path(start, goal, max_depth=depth)
        if path is None:
            # 找不到不是错误：名字对不上 / 太深了，都照原样说清楚
            return self._no_path(start, goal, depth)

        text = self._render_path(path)
        if asked > self.max_depth_limit:
            text += f"\n…（max_depth 最多给到 {self.max_depth_limit}）"
        return text

    def _render_path(self, path: "CallPath") -> str:
        """整条链一行一条：位置 + 名字 + 类型（对不上符号就给调用点）。

        位置给的是每一步函数自己的定义处，拿它去 read_file 就行；至于
        「是上一步的哪一行调过来的」不重复写——链的顺序本身就说清了。
        """
        if not path.edges:
            header = f"{path.nodes[0]} → {path.nodes[0]}（起点就是终点，长度 0）："
        else:
            header = (
                f"{' → '.join(path.nodes)}"
                f"（{len(path.nodes)} 个函数，{path.length} 次调用）："
            )

        lines = [header]
        for position, node in enumerate(path.nodes, start=1):
            # 进入这一步的那条边：nodes[i] 的上一步是 edges[i-1]
            incoming = path.edges[position - 2] if position > 1 else None
            lines.append(f"{position}. {self._describe_node(node, incoming)}")
        return "\n".join(lines)

    def _describe_node(self, node: str, incoming: "CallEdge | None") -> str:
        """链上一步的定位：symbol 定义优先，对不上退回进入这步的调用点。"""
        symbol = self._symbol_for(node)
        if symbol is not None:
            return self._symbol_text(symbol)
        if incoming is not None:
            return self._edge_text(incoming, node)
        return f"{node}  （不在符号索引里，也没找到调用点）"

    def _no_path(self, start: str, target: str, max_depth: int) -> str:
        """找不到链时的清楚说法：名字对不上，还是深度不够。"""
        clues = []
        if not self.index.edges(caller=start):
            clues.append(
                f"起点 {start!r} 在调用图里没有出边（名字对不上，或者它谁也不调）"
            )
        if not self.index.edges(callee=target):
            clues.append(
                f"目标 {target!r} 在调用图里没有入边（名字对不上，或者没人调它）"
            )
        if not clues:
            # 两头都在，那就是中间断了或深度不够——两种补救都说一句。
            # 「链中间」那种断法加深度也没用，所以要指出来
            deeper = (
                f"可以加大 max_depth（上限 {self.max_depth_limit}）再试"
                if max_depth < self.max_depth_limit
                else "已经到 max_depth 上限了"
            )
            clues.append(
                f"{deeper}；链中间经对象调的那一步（self.x、obj.x）接不下去，"
                f"用 callers / callees 分头查"
            )
        detail = "；" + "；".join(clues) if clues else ""
        return (
            f"（没有找到 {start} → {target} 的调用链："
            f"在 {max_depth} 层内没走通{detail}）"
        )

    def _describe(self, name: str, function: str, direction: str) -> str:
        """一条结果的定位信息：优先给符号定义处，对不上就退回调用点。

        只给位置不给正文——正文交给 read_file。
        """
        symbol = self._symbol_for(name)
        if symbol is not None:
            return self._symbol_text(symbol)
        edge = self._call_site(name, function, direction)
        if edge is None:
            return f"{name}  （不在符号索引里，也没找到调用点）"
        return self._edge_text(edge, name)

    @staticmethod
    def _symbol_text(symbol: "Symbol") -> str:
        """一个符号的定位：路径:起止行 + 限定名 + 类型。"""
        return (
            f"{symbol.path}:{symbol.start_line}-{symbol.end_line}"
            f"  {symbol.qualified}  [{symbol.kind}]"
        )

    @staticmethod
    def _edge_text(edge: "CallEdge", name: str) -> str:
        """对不上符号时的定位：退回这次调用发生的位置。"""
        return f"{edge.path}:{edge.line}  {name}  （调用点；符号索引里没有定义）"

    def _symbol_for(self, name: str) -> "Symbol | None":
        """把名字对回符号索引；`obj.foo` / `ns.foo` 按最后一段再试一次。

        callee 里很多是库函数、宏、类型名，对不上是常态——对不上返回 None，
        调用方退回调用点位置，不当成错误。
        """
        symbol = self.index.symbol(name)
        if symbol is None and "." in name:
            # 记下来的是 obj.foo / ns.foo 这种：obj 是谁没有类型信息说不清，
            # 但最后一段的符号名往往就是它（和 callers / callees 的认法一致）
            symbol = self.index.symbol(name.rsplit(".", 1)[-1])
        return symbol

    def _call_site(
        self, name: str, function: str, direction: str
    ) -> "CallEdge | None":
        """名字对不上符号时退回调用点：这次调用在哪个文件的哪一行。

        方向反过来找：查调用者（callers）时 name 是调用方，边记的是
        「name 调 function」；查被调用方（callees）时反过来。取第一条
        （边按路径、行号排好序），够模型去 read_file 看上下文了。
        """
        if direction == "callers":
            found = self.index.edges(caller=name, callee=function)
        else:
            found = self.index.edges(caller=function, callee=name)
        return found[0] if found else None

    def _empty(self, name: str, direction: str) -> str:
        """查不到时的两种说法：函数压根不在，还是它在、只是没这种关系。

        「不在」和「没有调用者」是两回事——不分开说，模型会拿后一种去
        下「这个函数没被用过」的结论，而事实可能是名字写错了。
        """
        symbol = self._symbol_for(name)
        if symbol is None:
            return (
                f"（没有找到 {name!r}：符号索引里没有这个函数，"
                f"调用图里也没有以它结尾的调用关系）"
            )
        if direction == "callers":
            return f"（{symbol.qualified} 没有已知的调用者）"
        return f"（{symbol.qualified} 没有调用任何已知的函数）"
