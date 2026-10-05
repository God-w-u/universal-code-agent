"""从 C/C++ 源码里提取结构：include、namespace、class/struct/union/enum、函数。

没有用 Clang，也没有真正的 C++ 语法分析——那需要能展开宏、能解析模板的真编译器。
这里是一行行扫出来的**近似**结果，规矩是：宁可漏，不能把不相干的语句报成结构，
更不能崩。宏展开、宏生成的名字、复杂模板、跨行拼出来的怪写法都可能漏掉。

怎么扫的
--------
1. 预处理指令（``#`` 开头）单独处理：``#include`` 记下来，其余整行不参与后面的
   扫描——``#define`` 里的花括号不是代码里的花括号。反斜杠续行的下一行也跟着排除。

2. 剩下的每行变成**两份**文本，长度一一对应，可以用同一套下标同时操作：

   - 代码版：注释和字符串都换成空格。**数括号只用它**——注释里的花括号、
     字符串里的括号，都不能参与层级计算。
   - 干净版：注释换成空格，字符串留着。识别出来的东西要给模型看，
     参数里的 ``"abc"`` 不该显示成一串空格。

   原始字符串 ``R"delim(...)delim"`` 里可以有花括号和引号，得整体跳过。

3. 把源码按括号深度为 0 的 ``;`` ``{`` ``}`` 切成一段段「声明」，每段记住起始行号。
   参数表跨行的签名就是这样拼回来的。

4. 给每段分类。开出 ``{`` 的段压栈、``}`` 弹出；**函数体里**一律不算结构，
   不然函数体里每个函数调用看着都像声明。栈里有没有 ``function`` 是这条的分界线。

行号用 tools/textfile.py 的 split_lines() 数，和 read_file / search_code 是同一套。

出门两个函数，同一条扫描：extract() 给「看一眼结构」用（code_summary）；
spans() 每条多带种类、名字和行范围，给「按范围切代码」用（index/symbols.py）。
code_only() 是扫描的另一半：把注释和字符串换成空格后的代码版文本，调用图
（callgraph/cpp.py）找调用点也用它，不然注释里写一句 `helper()` 就成了调用。
"""

import re
from typing import NamedTuple

from tools.textfile import split_lines


class Entry(NamedTuple):
    """一条提取结果：行号、嵌套层级、显示文本。"""

    line: int
    depth: int
    text: str


class Span(NamedTuple):
    """一条结构 + 它占的行范围。行号都是 1 起、闭区间，直接对应源文件的行。

    - `kind`：class / struct / union / enum / namespace / function，
      `#include` 是 "include"。**没认出来的块不会出现在这里**，
      它们只是压栈占个位，不会变成一条既不像声明、也不像定义的东西。
    - `name`：简单名字。函数可能带 `::` 限定（`Renderer::draw`）；
      匿名 namespace 和 `#include` 没有名字，是空串。
    - `text`：给人看的签名，和 Entry.text 一模一样。
    - `start`：这段声明的第一行。`template <...>`、返回类型常常单独占一行，
      切块得从那里开始，所以它和 line 不一定相同。
    - `line`：名字所在行（老规矩：报给模型去核对的行号）。
    - `end`：最后一行。开了 `{` 的就是配对 `}` 那一行；没配上的（文件被截断、
      `#ifdef` 掐掉一半）算到文件末尾；`;` 结尾的就是分号那一行。
    - `depth`：嵌套层级，和 Entry.depth 一个算法。
    """

    kind: str
    name: str
    text: str
    start: int
    line: int
    end: int
    depth: int


class Found(NamedTuple):
    """一段声明的识别结果：种类、名字、显示文本、名字在段里的位置。

    位置是给调用方换算行号用的。
    """

    kind: str
    name: str
    text: str
    offset: int


# 一段声明超过这么长就不认了。正则碰上一段又长又全是词字符、却始终配不上括号的
# 文本，回溯会退化成平方级——几万字符就足够让工具卡在那儿。宁可漏一条。
MAX_STATEMENT = 2000
# 签名最多显示这么长，超了截断
MAX_SIGNATURE = 80

_INCLUDE = re.compile(r'#\s*include\s*([<"][^>"]*[>"])')
# 原始字符串的前缀：R"(...)" / u8R"(...)" / LR"(...)"
_RAW = re.compile(r'(?:u8|u|U|L)?R"([^()\\ ]{0,16})\(')


def extract(text: str) -> list[Entry]:
    """提取一个 C/C++ 文件的结构，按行号顺序返回。"""
    return [Entry(item.line, item.depth, item.text) for item in spans(text)]


def spans(text: str) -> list[Span]:
    """和 extract 一样，但每条多带着：种类、名字，和它占的行范围。

    extract 是给「看一眼结构」用的（code_summary），spans 是给「按范围切代码」
    用的（index/symbols.py 建符号索引）。两者走**同一条扫描**，不是两份实现，
    所以报出来的行号、层级、文本不会各说各话。
    """
    entries: list[Span] = []
    sources: list[str] = []
    continuation = False

    for number, line in enumerate(split_lines(text), 1):
        stripped = line.lstrip()
        if continuation or stripped.startswith("#"):
            # 反斜杠续行的第二行不以 # 开头，但同样是宏的一部分，一起排除
            continuation = line.rstrip().endswith("\\")
            found = _INCLUDE.match(stripped)
            if found:
                entries.append(
                    Span(
                        kind="include",
                        name="",
                        text=f"#include {found.group(1)}",
                        start=number,
                        line=number,
                        end=number,
                        depth=0,
                    )
                )
            sources.append("")  # 整行作废
            continue
        sources.append(line)

    code, clean = _blank("\n".join(sources))
    entries.extend(_scan(code.split("\n"), clean.split("\n")))
    # 指令和结构是分开收的，按行号合流（sort 是稳定排序，同行的保持原顺序）
    return sorted(entries, key=lambda entry: entry.line)


def code_only(text: str) -> str:
    """注释和字符串都换成空格的「代码版」文本（和原文逐行一一对应）。

    spans() 数括号用的就是它（经 _blank）；调用图（callgraph/cpp.py）找
    调用点也借它——注释里写一句 `helper()`、字符串里出现 `run(`，都不该
    被当成一次调用。长度和原文完全一致，行号可以直接用。
    """
    return _blank(text)[0]


def _blank(text: str) -> tuple[str, str]:
    """把注释和字符串换成空格，返回 (代码版, 干净版)。两份长度完全一致。"""
    code: list[str] = []
    clean: list[str] = []
    index = 0
    size = len(text)

    while index < size:
        char = text[index]

        if text.startswith("//", index):
            stop = text.find("\n", index)
            stop = size if stop < 0 else stop
            _mask(code, clean, text[index:stop], keep=False)
        elif text.startswith("/*", index):
            stop = text.find("*/", index + 2)
            stop = size if stop < 0 else stop + 2  # 没闭合就吃到文件尾
            _mask(code, clean, text[index:stop], keep=False)
        elif (raw := _RAW.match(text, index)) is not None:
            end = ")" + raw.group(1) + '"'
            stop = text.find(end, raw.end())
            stop = size if stop < 0 else stop + len(end)
            _mask(code, clean, text[index:stop], keep=True)
        elif char in "\"'":
            stop = _end_of_quoted(text, index)
            _mask(code, clean, text[index:stop], keep=True)
        else:
            code.append(char)
            clean.append(char)
            stop = index + 1

        if stop <= index:  # 兜底：卡在原地比漏一条糟得多
            stop = index + 1
        index = stop

    return "".join(code), "".join(clean)


def _mask(code: list[str], clean: list[str], piece: str, keep: bool) -> None:
    """把一段文本追加到两份结果里。keep=False 时干净版也换成空格。"""
    code.append(_spaces(piece))
    clean.append(piece if keep else _spaces(piece))


def _spaces(piece: str) -> str:
    """换成等长空格，**换行符保留**——不然行号就全乱了。"""
    return "".join("\n" if char == "\n" else " " for char in piece)


def _end_of_quoted(text: str, start: int) -> int:
    """从引号处往后找配对的引号，返回结束位置（含引号）。"""
    quote = text[start]
    index = start + 1

    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2  # 转义：下一个字符不算数
        elif char == quote:
            return index + 1
        elif char == "\n":
            break  # 字符串不跨行，多半是别处混进来的撇号
        else:
            index += 1

    # 没配上对：只当它一个字符，别把后面全吃掉
    return start + 1


def _scan(code_lines: list[str], clean_lines: list[str]) -> list[Span]:
    """按括号把源码切成一段段声明，逐段识别。"""
    entries: list[Span] = []
    # 每开一个 { 压一个 (种类, 记录下标)，} 弹出。下标是这条结构在 entries 里的
    # 位置，等配对的 } 出现时好回去把 end 补上；认不出的块是 None（没有记录）。
    scopes: list[tuple[str, int | None]] = []
    buffer: list[str] = []
    start = 0  # 这段声明从第几行开始
    parens = 0  # () 的嵌套深度

    for number, (code, clean) in enumerate(zip(code_lines, clean_lines), 1):
        if buffer:
            # 段里得留着换行：跨行的声明要靠它把「名字在前面第几行」换算出来。
            # 切行时换行符被切掉了，这里补回来。
            buffer.append("\n")
        for index, char in enumerate(code):
            if char == "(":
                parens += 1
            elif char == ")":
                parens = max(0, parens - 1)
            elif parens == 0 and char in "{};":
                if char == "}":
                    # 多出来的 }（代码不完整）忽略掉就是，别让它把栈弹穿
                    if scopes:
                        _close(entries, scopes.pop(), number)
                else:
                    # 段里一个非空白字符都没有的话，start 还是 0，退回用当前行
                    statement = "".join(buffer).strip()
                    _collect(entries, scopes, start or number, statement, char, number)
                buffer.clear()
                start = 0
                continue

            if start == 0 and not char.isspace():
                start = number
            buffer.append(clean[index])

    tail = "".join(buffer).strip()  # 文件末尾没闭合的那一段
    if tail:
        _collect(entries, scopes, start, tail, "", len(code_lines))

    # 一直没配上 } 的（文件被截断、#ifdef 掐掉一半）算到文件末尾：
    # 留着半空的范围，切块的人不知道切到哪儿
    for scope in scopes:
        _close(entries, scope, len(code_lines))
    return entries


def _close(entries: list[Span], scope: tuple[str, int | None], line: int) -> None:
    """一个 { 配上了 }：把当初那条记录的范围补到这一行。"""
    _, index = scope
    if index is not None:
        entries[index] = entries[index]._replace(end=line)


# 会「占一层」的种类：namespace 里的类、类里的方法，都要缩进一级。
# 认不出来的块压一个空标记，只用来和右边的 } 对齐，不占层级。
_STRUCTURAL = frozenset({"class", "struct", "union", "enum", "namespace", "function"})


def _collect(
    entries: list[Span],
    scopes: list[tuple[str, int | None]],
    start: int,
    statement: str,
    terminator: str,
    here: int,
) -> None:
    """识别一段声明：是结构就记一条，开了块就压栈。

    `start` 是这段声明的第一行，`here` 是终结符（`;` / `{` / 文件末尾）所在行。
    """
    if not statement:
        return

    # 两个都必须在压栈之前算：函数自己也占一层，
    # 压完再算就会把自己也算进去，层级差一位
    depth = sum(1 for kind, _ in scopes if kind in _STRUCTURAL)
    inside_function = any(kind == "function" for kind, _ in scopes)
    found = _classify(statement, terminator)

    # 这条记录一会儿会落在 entries 的哪个位置；不记就是 None。
    # 压栈时要把这个位置一起带上，配对的 } 出现时才找得回来。
    index = len(entries) if found is not None and not inside_function else None

    if terminator == "{":
        if found is not None:
            scopes.append((found.kind, index))
        elif "=" in statement:
            # 带 = 的 { 是初始化列表或者 λ 的体，里面是代码不是声明。
            # 不压这个标记的话，λ 里的每个函数调用都会被报成函数声明。
            scopes.append(("function", None))
        else:
            # 认不出的块，最常见的是 extern "C" {——里面照样是声明，
            # 所以只压个不挡事的空标记
            scopes.append(("", None))

    # found 为 None 时 index 一定也是 None（上面同一行算出来的）；这里一并
    # 写上，是给读代码的人和类型检查看的：往下 found 一定有值
    if index is None or found is None:
        return

    # 行号报「名字在哪一行」，不是「这段声明从哪一行开始」：`template <...>` 和
    # 返回类型常常另起一行，报起始行的话模型去那一行核对时什么都对不上。
    # 段里的换行都留着，所以数一下名字前面有几个换行就换算得出来。
    # end 先填终结符那一行：`;` 结尾的到这儿就完了，`{` 开的等配对的 } 来改。
    entries.append(
        Span(
            kind=found.kind,
            name=found.name,
            text=found.text,
            start=_declaration_start(start, statement),
            line=start + statement[: found.offset].count("\n"),
            end=here,
            depth=depth,
        )
    )


def _declaration_start(start: int, statement: str) -> int:
    """这段声明的第一行。

    段首可能是 `public:` / `case 3:` 这类标签——它们没有 `;`，会和后面的声明
    粘成一段。标签不算声明的一部分，切块的时候不该带进来，所以往下跳。

    只跳「整行就是个标签」的行（`:` 结尾）和空行；声明自己那一行（段的最后
    一行）永远不跳。
    """
    offset = 0
    for line in statement.split("\n")[:-1]:
        if line.strip() and not line.strip().endswith(":"):
            break
        offset += 1
    return start + offset


# enum 必须排在 class/struct 前面：`enum class Color` 里也有一个 class
_ENUM = re.compile(r"^enum\s+(?:(class|struct)\s+)?([A-Za-z_]\w*)\s*(?=[:{;]|$)")
_NAMESPACE = re.compile(
    r"^(?:inline\s+)?namespace\b\s*([A-Za-z_]\w*(?:::\w+)*)?\s*(?=[{;]|$)"
)
# 名字后面得跟 { : ; 或者 final，才算定义。这一条同时挡住了两种认错：
# `template <class T>` 里的 T 后面是 > ，`struct Point p;` 里的 Point 后面是个变量名。
_AGGREGATE = re.compile(
    r"\b(?P<kind>class|struct|union)\s+(?P<name>[A-Za-z_]\w*(?:::\w+)*)\s*"
    r"(?=[{:;]|final\b)"
)

# 函数名：普通标识符（可以带 :: 限定和析构的 ~），或者 operator 重载。
# operator 单列是因为它的「名字」可以是符号：operator=、operator<<、operator()。
_NAME = r"(?:operator\s*(?:\[\s*\]|\(\s*\)|[^\s(]+)|~?[A-Za-z_]\w*(?:::[~\w]+)*)"
_FUNCTION = re.compile(
    r"^(?P<prefix>[\w:<>*&\[\]\s,]*?)"
    r"(?P<name>" + _NAME + r")"
    r"\s*\((?P<params>[^()]*(?:\([^()]*\)[^()]*)*)\)"
    r"\s*(?P<suffix>[^{;]*?)"
    r"(?P<end>[;{]?)$"
)

# 括号里以数字、引号或者负号开头的，是「用括号初始化成员」而不是函数声明
# （`int port_(8080);`）。C++ 里没有哪种类型是这么开头的，所以这条很干脆。
_NOT_A_PARAM = re.compile(r"""^\s*[\d"'-]""")

# 签名里留下的限定词，按出现顺序去重
_QUALIFIER = re.compile(
    r"\b(?:const|volatile|noexcept|override|final|mutable)\b"
    r"|=\s*(?:0|delete|default)\b"
)

# 这些词出现在「名字」的位置，说明这段根本不是函数声明。两种来路：
# 控制语句（if / while / catch），以及类型名被当成了函数名——
# `typedef void (*cb)(int);` 会先匹配到 void。
_NOT_A_NAME = frozenset(
    {
        "if", "for", "while", "switch", "catch", "return", "sizeof", "alignas",
        "decltype", "static_assert", "static_cast", "dynamic_cast",
        "reinterpret_cast", "const_cast", "throw", "new", "delete", "do", "else",
        "case", "void", "bool", "char", "short", "int", "long", "float", "double",
        "signed", "unsigned", "auto", "const", "volatile", "static", "inline",
        "virtual", "explicit", "extern", "mutable", "constexpr", "typedef",
        "typename", "template", "class", "struct", "union", "enum", "namespace",
        "using", "public", "private", "protected", "friend",
    }
)


def _classify(statement: str, terminator: str) -> Found | None:
    """给一段声明分类；不是结构就返回 None。

    认不出来是常态——大多数语句都不是结构，这不算错。
    """
    code = statement + terminator
    if len(code) > MAX_STATEMENT:
        return None

    found = _ENUM.match(code)
    if found:
        keyword = f"enum {found.group(1)}" if found.group(1) else "enum"
        name = found.group(2)
        return Found("enum", name, f"{keyword} {name}", found.start(2))

    found = _NAMESPACE.match(code)
    if found:
        name = found.group(1) or ""
        where = found.start(1) if name else found.start()
        text = f"namespace {name}" if name else "namespace (匿名)"
        return Found("namespace", name, text, where)

    if re.search(r"\bfriend\b", code):
        # friend 声明的不是自己的成员，报出来会让人以为类里有这么个方法
        return None

    found = _AGGREGATE.search(code)
    if found:
        kind = found.group("kind")
        name = found.group("name")
        return Found(kind, name, f"{kind} {name}", found.start("name"))

    found = _FUNCTION.match(code)
    if (
        found
        and found.group("name") not in _NOT_A_NAME
        and not _NOT_A_PARAM.match(found.group("params"))
    ):
        return Found("function", found.group("name"), _signature(found), found.start("name"))

    return None


def _signature(found: re.Match) -> str:
    """拼出显示用的签名：`名字(参数) 限定词`。

    返回类型不要——`static inline std::vector<std::string> ` 这一串太占地方，
    要看它 read_file 就行；参数要留下，重载之间就靠它区分。
    """
    params = " ".join(found.group("params").split())
    qualifiers = " ".join(dict.fromkeys(_QUALIFIER.findall(found.group("suffix"))))
    text = " ".join(f"{found.group('name')}({params}) {qualifiers}".split())

    if len(text) > MAX_SIGNATURE:
        text = text[:MAX_SIGNATURE] + "…"
    return text
