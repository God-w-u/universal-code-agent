"""Code Chunk：按 Symbol 的边界把真实代码切出来，一段一块。

检索这条线的第三步。前两步只记「哪儿有什么」——文件清单（Project Index）、
符号位置（Symbol Index）；这一步才把**代码正文**取出来：一个类 / 函数 / 方法
一段，起止行就是 Symbol 的行范围，正文是从文件里按这个范围读出来的原文。
后续的 Hybrid Search 和 Code RAG 就检索这些块，Agent 也能顺着 path +
start_line / end_line 用 read_file 回去看原文。

几条刻意的边界：

- **代码是真读出来的**：按 Symbol 的行范围从文件里切，不拿符号名或摘要
  拼一段出来。切出来的行和 read_file 显示的那几行逐字节一致（只是没有
  `42|` 那样的行号前缀）。
- **不产生过期 chunk**：读文件用的是 SymbolIndex.read_verified——和符号索引
  同一份实现，digest 对不上就整段不切，记一条 Skipped(changed)。行号错了的
  chunk 比没有 chunk 更糟：检索命中之后，Agent 会照着它去 read_file 看一段
  完全不相干的代码。
- **一个 symbol 一条 chunk**：class / struct / function / method 各一条，
  `kind` 和 `symbol`（限定名，`Agent.run`）都带在记录里，`parent` 挂着父级
  限定名——父子关系就是检索时说的「Class.method」。
- **嵌套只去重「完全一样」的那种**：类里套方法，两条 chunk 的行范围本来
  就不同（类的范围更大、正文里含着方法），这是有意保留的父子块；
  只有行范围**一模一样**时（一行写完的 `struct A { void f(); };`）才只留
  外层那条——内层那条的代码会和外层一个字不差，白占一份。
- **没有 symbol 的文件本阶段不切**：切的入口只有一个，扩展点留在 build()
  里那段注释上，等真需要（比如纯配置文件也要检索）再接。

本阶段不做 embedding、向量库、语义搜索、Hybrid Search、RAG Tool，也不改
Agent / RepairLoop / Verifier / 编译测试 / EditFile 的行为。
"""

from dataclasses import dataclass
from pathlib import Path

from index.project import IndexedFile
from index.symbols import Skipped, Symbol, SymbolIndex
from tools.base import ToolError
from tools.textfile import split_lines


@dataclass(frozen=True)
class CodeChunk:
    """一段可以拿去检索 / 给模型看的代码。冻结的记录，全是算好的值。

    - `path` / `start_line` / `end_line`：这段代码在项目里的位置，1 起、
    闭区间、`/` 分隔——原样就能交给 read_file 复核。
    - `symbol`：**限定名**（`Agent.run`），和 Symbol.qualified 一个写法。
    - `kind`：class / struct / function / method，符号类型。
    - `language`：语言 id，和 Project Index 里的写法一致。
    - `digest`：**所属文件**的内容摘要（不是这段代码自己的）。文件一改，
      这条 chunk 就作废。
    - `parent`：父级 symbol 的限定名，顶层是 None。它**可能指向一个没有
      chunk 的名字**——C/C++ 的 namespace 不产出符号，被上面去重规则去掉的
      内层符号也没有；所以它是「名字」而不是「另一条 chunk 的 id」。
    - `code`：从真实文件里按 start_line..end_line 切出来的原文。换行统一成
      `\\n`、末尾不带换行。
    """

    path: str
    symbol: str
    kind: str
    language: str
    start_line: int
    end_line: int
    digest: str
    code: str
    parent: str | None = None

    @property
    def id(self) -> str:
        """chunk 的标识：`agent/agent.py:Agent.run:40-52`。

        「同一个文件里、同一个符号、同一段行」就是同一条 chunk。用可读的
        拼法而不是哈希：翻日志、看测试失败时一眼能认出说的是哪段代码。
        """
        return f"{self.path}:{self.symbol}:{self.start_line}-{self.end_line}"


class ChunkIndex:
    """切块索引。用法：建一个 → build() → 查 chunks()。

        index = ChunkIndex()                       # 默认整个项目
        index = ChunkIndex(root=some_dir)          # 指定目录（测试里用）
        index.build()
        index.chunks()                             # 全部
        index.chunks(path="agent/agent.py")        # 某个文件里的
        index.chunks(symbol="Agent.run")           # 某个符号的（简单名 / 限定名都认）
        index.chunks(kind=METHOD)                  # 所有方法
        index.chunks(parent="Agent")               # Agent 里面那一层
        index.skipped                              # 想切却没切成的，和为什么

    符号索引由谁管，和 SymbolIndex 一个规矩：`root=`（或默认）是自己管的，
    build() 开头先建一遍、每次都是全量刷新；`symbols=` 传来的归调用方管，
    没扫过会替它扫一次，扫过的（哪怕已经过时）不碰。
    """

    def __init__(
        self,
        symbols: SymbolIndex | None = None,
        root: Path | None = None,
    ) -> None:
        if symbols is not None and root is not None:
            raise ToolError("symbols 和 root 只能给一个：传了 symbols 就由调用方管索引")
        self.symbols = symbols if symbols is not None else SymbolIndex(root=root)
        self._own_symbols = symbols is None
        self._chunks: list[CodeChunk] = []
        self._skipped: list[Skipped] = []
        self._built = False

    def build(self) -> None:
        """全量重建：按符号边界把已索引文件的代码切出来。再调一次就是刷新。"""
        if self._own_symbols or not self.symbols.built:
            self.symbols.build()

        # 符号按文件归堆，一个文件只读一次（digest 也只用核一次）
        grouped: dict[str, list[Symbol]] = {}
        for item in self.symbols.symbols():
            grouped.setdefault(item.path, []).append(item)

        chunks: list[CodeChunk] = []
        skipped: list[Skipped] = []

        for entry in self.symbols.project.files():
            items = grouped.get(entry.path)
            if not items:
                # 没有 symbol 的已索引文本：本阶段不切块（不是出错）。
                # 以后要按行窗口 / 顶层语句切这类文件，就从这里接——先
                # self.symbols.read_verified(entry) 拿正文，再切。现在刻意
                # 不做：为覆盖所有文件设计一套 fallback，会把「按 symbol
                # 边界切」这条主线冲淡，切出来的边界也说不清是什么。
                continue

            text, reason = self.symbols.read_verified(entry)
            if reason is not None:
                skipped.append(Skipped(entry.path, reason))
                continue

            chunks.extend(_file_chunks(entry, split_lines(text), items))

        self._chunks = chunks
        self._skipped = skipped
        self._built = True

    @property
    def built(self) -> bool:
        """切过没有。从没切过的索引是**空的**，但那不等于「项目里没有 chunk」——
        没切过和切完为空是两回事，检索器靠这个标记决定要不要先替它切一次。"""
        return self._built

    def chunks(
        self,
        path: str | None = None,
        symbol: str | None = None,
        kind: str | None = None,
        parent: str | None = None,
    ) -> tuple[CodeChunk, ...]:
        """按条件查 chunk，条件是「与」的关系，都不给就是全部。

        symbol 简单名（`run`）和限定名（`Agent.run`）都认；parent 认父级的
        限定名（`Agent` 里直接包着的那一层）。
        结果按文件路径、文件内的先后排。
        """
        return tuple(
            item
            for item in self._chunks
            if (path is None or item.path == path)
            and (
                symbol is None
                or item.symbol == symbol
                or _simple(item.symbol) == symbol
            )
            and (kind is None or item.kind == kind)
            and (parent is None or item.parent == parent)
        )

    @property
    def skipped(self) -> tuple[Skipped, ...]:
        """想切却没切成的文件：读不到、或者索引建好之后被改过。"""
        return tuple(self._skipped)


def _file_chunks(
    entry: IndexedFile, lines: list[str], items: list[Symbol]
) -> list[CodeChunk]:
    """一个文件里的符号 → chunk。

    先按 (起始行, 结束行倒序) 排：外层的 symbol 排在前。两条 symbol 的行范围
    一模一样时（一行写完的 `struct A { void f(); };`）只留先来的那条，也就是
    外层——内层那条的代码会和它一个字不差。父子关系由 parent 字段挂着，
    外层那条的正文里本来就含着内层。
    """
    chunks: list[CodeChunk] = []
    seen: set[tuple[int, int]] = set()

    for item in sorted(items, key=lambda one: (one.start_line, -one.end_line)):
        span = (item.start_line, item.end_line)
        if span in seen:
            continue
        seen.add(span)

        chunks.append(
            CodeChunk(
                path=entry.path,
                symbol=item.qualified,
                kind=item.kind,
                language=entry.language,
                start_line=item.start_line,
                end_line=item.end_line,
                digest=entry.digest,
                # 行号是 1 起的闭区间，切出来正好是这几行；
                # 行号和 read_file 是同一套（tools/textfile.py 的 split_lines）
                code="\n".join(lines[item.start_line - 1 : item.end_line]),
                parent=item.parent,
            )
        )

    return chunks


def _simple(qualified: str) -> str:
    """限定名的最后一段：`Agent.Inner.run` → `run`。"""
    return qualified.rsplit(".", 1)[-1]
