"""Symbol Index：每个已索引的文件里有哪些类 / 函数 / 方法。

在 Project Index 之上做第二步。文件清单已经有了（哪个文件、什么语言、收没
收进来），这一层回答「每个文件里有什么」：每个符号记一条 Symbol——名字、
类型、项目相对路径、起止行、所属文件的内容摘要、父级符号。后面的
Code Chunk（按符号边界切块）和 Hybrid Search（按名字 / 类型找）都从这份
数据上接着做。

几条刻意的边界：

- **只认已索引的文件**：遍历的是 ProjectIndex.files()，不自己扫目录——
  「哪些文件算数」只有 tools/tree.py 那一份实现。
- **不产生过期符号**：读回来的内容如果和建索引时记的 digest 对不上
  （文件被人改过），这个文件整个跳过、记一条 Skipped(CHANGED)——而不是
  拿旧行号去切已经变了的文件。文件没了、读不动也一样跳过。
- **行号必须能在真实源文件里对上**：后面的 Code Chunk 要拿 start_line /
  end_line 去 read_file，所以 Python 用 ast 的准行号，C/C++ 用
  tools/cpp.py 扫出来的行范围（近似结果，宏和复杂模板可能漏）。
- **一份实现**：解析按语言分家——Python 走 tools/pyast.py，C/C++ 走
  tools/cpp.py，和 code_summary 用的是同两套，这里不重复实现。
- **C/C++ 的方法 / 函数是近似的**：类体里的函数算方法；名字带 `::`
  限定的也算方法（`Renderer::draw`），因为带限定的定义绝大多数是成员函数
  ——`void app::run() {}`（命名空间里的自由函数）会被认成方法，这是已知
  的偏差。转换运算符（`operator std::string`）的名字不拆。

本阶段不做 embedding、向量库、语义搜索、Code Chunk、RAG Tool，也不改
Agent / RepairLoop / Verifier / 编译测试 / EditFile 的行为。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from index.project import UNREADABLE, IndexedFile, ProjectIndex, digest_of
from tools import cpp, pyast
from tools.base import ToolError
from tools.textfile import decode_text, read_bytes

# 符号类型。英文短 id，和数据里的其它状态字段一个规矩（给程序读，
# 要展示的时候由调用方翻译）。enum / namespace 现在只参与父级链、
# 不产出符号，以后要加就是往 _SYMBOL_KIND 里补一行。
CLASS = "class"
STRUCT = "struct"
FUNCTION = "function"
METHOD = "method"

# 跳过某个文件的原因。只记「本来要处理、但没处理成」的，
# 语言还没有提取器的（.md、.java）不算——那不是出错，是还没轮到。
# UNREADABLE 借的是 index.project 里那个同名字的文件状态：打不开就是打不开，
# 不需要第二个常量来说同一件事。
CHANGED = "changed"  # 内容变了（或长大了），索引里的行号已经不作数
UNPARSABLE = "unparsable"  # 内容读到了，但解析不了（Python 语法错之类）


@dataclass(frozen=True)
class Symbol:
    """一个符号——一条记录，全是算好的值，不指向别处。

    - `name`：简单名（`draw`），不带父级前缀。
    - `kind`：CLASS / STRUCT / FUNCTION / METHOD。
    - `path`：项目相对路径，和 IndexedFile.path 是同一个写法。
    - `start_line` / `end_line`：1 起、闭区间，直接对应源文件的行，
      能原样拿去 read_file 切出这个符号。
      带装饰器的 Python 定义从装饰器那一行算起，C++ 的模板 / 返回类型
      单独占行时也算进去——切块要切出完整的定义。
    - `digest`：**所属文件**的内容摘要（index.project.digest_of），
      不是符号自己的。文件一改，这条就作废，拿它和新的摘要一比就知道。
    - `parent`：父级符号的限定名，顶层是 None。统一用点连接
      （`Agent.Inner`、`app.Renderer`），语言之间写法一致。
    """

    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    digest: str
    parent: str | None = None

    @property
    def qualified(self) -> str:
        """限定名：`Agent.Inner.run`。查名字时简单名和它都认。"""
        return f"{self.parent}.{self.name}" if self.parent else self.name


@dataclass(frozen=True)
class Skipped:
    """一个没进符号索引的文件，和为什么。"""

    path: str
    reason: str  # CHANGED / UNREADABLE / UNPARSABLE


class _Part(NamedTuple):
    """提取器交出来的半成品：缺路径和 digest，那两样由索引统一补。

    两个语言的提取器都归到这个形状，Symbol 的构造就只有一处。
    """

    kind: str
    name: str
    start: int
    end: int
    parent: str | None


def _python_symbols(text: str) -> list[_Part]:
    """Python：直接用 ast 的结果，父级限定名本来就是点连接的。"""
    return [
        _Part(item.kind, item.name, item.start, item.end, item.parent)
        for item in pyast.extract(text)
    ]


# C/C++ 里有这些「包着别人」的结构。它们自己不一定产出符号，
# 但里面的东西要按它们定父级。
_CONTAINERS = frozenset({"class", "struct", "union", "enum", "namespace"})
# 这几个是「类」：里面的函数算方法
_AGGREGATES = frozenset({"class", "struct", "union"})
# tools/cpp.py 的种类 → 符号类型，一对一的那几种。union / enum / namespace
# 暂时只进父级链、不产出符号；以后要给它们建符号，在这里补一条、
# 再加一个类型常量就行（函数不在这张表里，见下面 _cpp_symbols）。
_SYMBOL_KIND = {"class": CLASS, "struct": STRUCT, "function": FUNCTION}


def _cpp_symbols(text: str) -> list[_Part]:
    """C/C++：把 tools/cpp.py 扫出来的行范围整理成符号。

    spans() 是按行排好的，所以顺着走一遍、用一个栈就能还原「谁包着谁」——
    进了外层结构的范围就压栈，出了它的结束行就弹掉，不用再扫一遍源码。
    """
    parts: list[_Part] = []
    stack: list[tuple[str, str, int]] = []  # (种类, 限定名, 结束行)

    for span in cpp.spans(text):
        while stack and span.start > stack[-1][2]:
            stack.pop()  # 已经出了外层结构的范围

        if span.kind not in _CONTAINERS:
            # "function" 是 tools/cpp.py 那边的种类名（和符号类型正好同名）。
            # 它不走 _SYMBOL_KIND 那张表：函数还要看上下文才知道是方法还是
            # 自由函数，映射不是一对一。
            if span.kind == "function":
                parts.append(_cpp_function(span, stack))
            continue  # include 之类：不产出符号

        simple, prefix = _split_qualified(span.name)
        if not simple:
            continue  # 匿名 namespace：没名字，没法用它定父级

        outer = _outer(stack, prefix)
        kind = _SYMBOL_KIND.get(span.kind)
        if kind is not None:
            parts.append(
                _Part(kind, simple, span.start, span.end, ".".join(outer) or None)
            )
        # 自己也算一级：里面的东西的父级是「外层 + 自己」
        stack.append((span.kind, ".".join(outer + [simple]), span.end))

    return parts


def _cpp_function(span: cpp.Span, stack: list[tuple[str, str, int]]) -> _Part:
    """C/C++ 的函数：类里的、或者名字带 `::` 限定的算方法，其余是自由函数。"""
    simple, prefix = _split_qualified(span.name)
    outer = _outer(stack, prefix)
    is_method = bool(prefix) or any(item[0] in _AGGREGATES for item in stack)
    return _Part(
        METHOD if is_method else FUNCTION,
        simple,
        span.start,
        span.end,
        ".".join(outer) or None,
    )


def _outer(stack: list[tuple[str, str, int]], prefix: list[str]) -> list[str]:
    """父级链，从外到内。

    栈里最内层的限定名本身就是完整路径（从最外层一路带上来的，`app.Renderer`），
    所以只要它一个，再拼上名字里带的 `::` 前缀——`void Renderer::draw()` 在
    类外定义时，父级是从名字里读出来的。
    """
    if stack:
        return [stack[-1][1]] + prefix
    return prefix


def _split_qualified(name: str) -> tuple[str, list[str]]:
    """把 `Renderer::draw` 拆成 ("draw", ["Renderer"])。

    `operator` 开头的名字不拆：转换运算符的名字里带 `::`
    （`operator std::string`），拆了就散架。
    """
    if "::" not in name or name.startswith("operator"):
        return name, []
    pieces = [piece for piece in name.split("::") if piece]
    if not pieces:
        return name, []
    return pieces[-1], pieces[:-1]


# 语言 id → 提取器。加一门语言（或者给 java / go 补上）就是往这儿补一条，
# 别的都不用动。key 是 index.languages 里的 id。
EXTRACTORS = {"python": _python_symbols, "cpp": _cpp_symbols}


class SymbolIndex:
    """符号索引。用法：建一个 → build() → 查 symbols()。

        index = SymbolIndex()                        # 默认整个项目
        index = SymbolIndex(root=some_dir)           # 指定目录（测试里用）
        index.build()
        index.symbols()                          # 全部符号
        index.symbols(path="agent/agent.py")     # 某个文件里的
        index.symbols(name="run")                # 叫 run 的（简单名 / 限定名都认）
        index.symbols(kind=METHOD)               # 所有方法
        index.skipped                            # 没进去的，和为什么

    文件清单由谁管，看清单是怎么来的：`root=`（或默认）是自己管的，build() 开头
    先扫一遍、每次都是全量刷新；`project=` 传来的归调用方管，刷新由调用方决定
    ——它也得能拿一份「建好之后文件又改过」的旧清单来验证 digest 保护。
    build() 之前先看 built。

    **要立刻拿到最新的符号**（文件增删改之后）就调 `refresh()`：它连调用方
    管的清单也一起重扫（`build(force=True)`），一路级联到项目清单。默认的
    `build()` 一个字不改——digest 保护靠的正是「调用方没点头就不动它的清单」。
    """

    def __init__(
        self,
        project: ProjectIndex | None = None,
        root: Path | None = None,
    ) -> None:
        if project is not None and root is not None:
            raise ToolError("project 和 root 只能给一个：传了 project 就由调用方管清单")
        self.project = project if project is not None else ProjectIndex(root)
        self._own_project = project is None
        self._symbols: list[Symbol] = []
        self._skipped: list[Skipped] = []
        self._built = False
        self._generation = 0

    def build(self, *, force: bool = False) -> None:
        """全量重建：把已索引的文件逐个读回来提取符号。再调一次就是刷新。

        `force=True`：连外面传进来的文件清单也重扫（`refresh()` 走这条）。
        默认 False 时调用方管的清单一次都不碰——digest 保护就靠这条。
        """
        if force or self._own_project or not self.project.built:
            # 自己管的清单（root= 或者默认）：每次刷新。
            # 外面传进来但从没扫过的：替它扫一次——不扫就是个空清单，
            # 会静默地一个符号都拿不到。扫过的（哪怕已经过时）一律不碰：
            # 那是调用方的东西，也正是 digest 保护能生效的场景。
            # force=True 是调用方自己点头：这次的清单也得是最新的。
            self.project.build()

        symbols: list[Symbol] = []
        skipped: list[Skipped] = []

        for entry in self.project.files():
            extractor = EXTRACTORS.get(entry.language)
            if extractor is None:
                continue  # 这门语言还没有提取器：不是出错，是还没轮到

            text, reason = self.read_verified(entry)
            if reason is not None:
                skipped.append(Skipped(entry.path, reason))
                continue
            # read_verified 的约定（见它的文档）：reason 为 None 时 text 一定有。
            # digest 同理——它是拿 entry.digest 核对过内容才算通过的，能走到
            # 这儿说明这条记录就是 INDEXED，digest 一定有值。
            assert text is not None
            assert entry.digest is not None

            try:
                parts = extractor(text)
            except (SyntaxError, ValueError, RecursionError):
                # 语法错、ast 本身拒绝的输入、嵌套过深：只影响这一个文件
                skipped.append(Skipped(entry.path, UNPARSABLE))
                continue

            symbols.extend(
                Symbol(
                    name=part.name,
                    kind=part.kind,
                    path=entry.path,
                    start_line=part.start,
                    end_line=part.end,
                    digest=entry.digest,
                    parent=part.parent,
                )
                for part in parts
            )

        # 全部扫完才换上新的：中途出意外的话，手上这份旧索引还是完整的
        self._symbols = symbols
        self._skipped = skipped
        self._built = True
        self._generation += 1

    def refresh(self) -> None:
        """显式刷新：连文件清单一起重扫，拿到当前的最新符号。

        文件增、删、改之后调它。调用图那一层盯着 `generation`：这份一刷新，
        建在它上面的调用图就自动作废、下次查询重建。
        """
        self.build(force=True)

    @property
    def generation(self) -> int:
        """这份符号索引换过几版：每扫成一次 +1，0 表示一版都没有。"""
        return self._generation

    @property
    def built(self) -> bool:
        """扫过没有。没扫过的索引是空的，但那不等于「项目里没有符号」。"""
        return self._built

    def symbols(
        self,
        path: str | None = None,
        name: str | None = None,
        kind: str | None = None,
    ) -> tuple[Symbol, ...]:
        """按条件查符号，条件是「与」的关系，都不给就是全部。

        name 简单名（`run`）和限定名（`Agent.run`）都认。
        结果按文件路径、文件内出现顺序排——就是提取时的顺序。
        """
        return tuple(
            item
            for item in self._symbols
            if (path is None or item.path == path)
            and (name is None or item.name == name or item.qualified == name)
            and (kind is None or item.kind == kind)
        )

    @property
    def skipped(self) -> tuple[Skipped, ...]:
        """没进符号索引的文件：读不到、改过了、解析不了。"""
        return tuple(self._skipped)

    # ------------------------------------------------------------------
    # 读文件
    # ------------------------------------------------------------------

    def read_verified(self, entry: IndexedFile) -> tuple[str | None, str | None]:
        """按索引里的路径把文件读回来，顺便核对它还是不是当初那份。

        返回 (文本, None)；读不成就是 (None, 原因)。**只读，不写。**

        「读一个已索引的文件」只有这一份实现：符号索引用它提取结构，
        切块索引（index/chunks.py）用它取代码正文——两处对「这个文件还能不能
        用」的判断必须是同一个，不然一边说新鲜、一边说过期就说不清了。
        """
        path = self.project.root / entry.path

        try:
            raw, too_big = read_bytes(path, self.project.max_file_bytes)
        except OSError:
            return None, UNREADABLE  # 列出来之后被删了 / 被占着 / 没权限

        if too_big or digest_of(raw) != entry.digest:
            # 内容对不上：索引里记的行号已经不作数了。这时候提取出来的
            # 只会是「过期符号」，不如一条不给——切块的人拿到错行号更糟。
            return None, CHANGED

        try:
            return decode_text(raw, entry.path), None
        except ToolError:
            # 索引说是文本、现在解不出来：当成读不到，不硬猜
            return None, UNREADABLE
