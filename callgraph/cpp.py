"""C/C++ 的调用点提取：只扫函数体，认不出来就跳过。

没有编译器、没有类型信息，能回答的只有「这一行上写着『名字(』」。所以
规矩和 tools/cpp.py 一样——**宁可漏，不记错的**：

- 只扫**函数体**里（行范围从 cpp.spans 来）：全局区、类体里的「名字(」
  多半是声明（`void f(int);`、`int get() const;`），不算调用。函数体的
  开头是签名后第一个不在括号里的 `{`——`Foo::Foo() : x_(bar()) {}` 这种
  初始化列表里的调用因此漏掉（它在 `{` 前面），这是有意的。
- 名字前面像「类型 + 名字」的（`int x(3);`、`Point p(1, 2);`）跳过：
  函数体里用括号初始化变量和调用长得一模一样，不挡就会多出一堆假边。
- 控制语句和转型（if / while / return / sizeof / static_cast …）整词
  跳过；`#define FOO(x)` 这类预处理行整行不扫。
- 注释和字符串里的东西不算——先过一遍 tools/cpp.py 的 code_only，
  两样都换成空格了。

认出来的名字按调用点原样记，`::` 和 `->` 统一写成点：`foo`、`obj.foo`、
`ns.foo`、`this.step`。**已知的近似**：显式模板实参的调用（`foo<int>(x)`）
会漏；宏调用（`assert(...)`）看起来就是调用，分不出来，照样记一笔。
"""

import re

from callgraph.base import CallSite
from tools import cpp
from tools.textfile import split_lines

# 调用点长什么样：一条点连接的链子后面跟参数表。
#
# 链子从最左边一个名字开始整条匹配（`obj.foo(` 匹配 "obj.foo"，不是在
# "foo" 处再匹配一次）；前面的负向后顾挡住「链子中间起头」的那种匹配。
# 名字后面可以换行跟 `(`，链子里不让有空格——`obj . foo(` 这种写法极少，
# 漏掉就漏掉。
_CALL = re.compile(
    r"(?<![\w:.>~])"
    r"(?P<name>[A-Za-z_]\w*(?:(?:::|\.|->)~?[A-Za-z_]\w*)*)"
    r"\s*\("
)
# 名字紧跟在标识符后面、而这个标识符又以这些字符结尾时，看着就是
# 「类型 / 对象 名字(」的声明（`std::vector<int> v(3);`、`Foo* p(a);`）
_BEFORE_TYPE = ">*&]"
# 紧跟在前面的词是这些的话，「标识符 + 名字(」仍是调用：`return foo(`、
# `case 1: foo(`、`throw Foo(` ……
_EXPRESSION_WORDS = frozenset(
    {
        "return", "case", "throw", "co_return", "co_await", "new", "delete",
        "and", "or", "not", "if", "while", "for", "switch", "else", "do",
    }
)
# 出现在「名字」位置但不是函数调用的词：控制语句、转型、以及写起来
# 就像函数调用的类型名（`int(x)` 是转型不是调用）。
# 这张表和 tools/cpp.py 的 _NOT_A_NAME 各管各的：那边是「这段声明像不像
# 函数」，这边是「这个调用点像不像调用」，标准不一样，不共用。
_NOT_A_CALL = frozenset(
    {
        "if", "for", "while", "switch", "catch", "return", "do", "else",
        "case", "sizeof", "alignas", "alignof", "decltype", "static_assert",
        "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast",
        "typeid", "new", "delete", "throw", "noexcept", "co_await",
        "co_return", "co_yield", "requires", "and", "or", "not", "operator",
        "using", "typedef", "template", "__attribute__", "__declspec",
        "void", "bool", "char", "short", "int", "long", "float", "double",
        "signed", "unsigned", "auto", "const", "volatile", "static", "inline",
        "virtual", "explicit", "extern", "mutable", "constexpr",
    }
)
_WORD_END = re.compile(r"[A-Za-z_]\w*$")


def extract_sites(text: str) -> list[CallSite]:
    """一个 C/C++ 文件里的调用点，按行号顺序返回。

    不是解析器，只是保守的文本扫描：认不出来的一律跳过，不抛异常。
    """
    lines = split_lines(cpp.code_only(text))
    directives = _directive_lines(split_lines(text))

    sites: list[CallSite] = []
    for start, column, end in _body_regions(lines, cpp.spans(text)):
        for number in range(start, end + 1):
            if number > len(lines):
                break  # 行号本该一一对应；对不上就停在这，别把整个索引带崩
            if number in directives:
                continue
            line = lines[number - 1]
            for match in _CALL.finditer(line):
                if number == start and match.start("name") < column:
                    # 签名部分（函数体从 { 后面才开始扫）
                    continue
                name = _normalized(match.group("name"))
                if _not_a_call(name) or _looks_like_declaration(
                    line, match.start("name")
                ):
                    continue
                sites.append(CallSite(number, name))
    return sites


def _body_regions(
    lines: list[str], spans: list[cpp.Span]
) -> list[tuple[int, int, int]]:
    """要扫的范围：(起始行, 起始列, 结束行)，1 起、闭区间。

    只有「真的有函数体」的函数才有范围：`void f(int);` 这样的声明找不到
    开体的 `{`，整条跳过。嵌在别的函数里的函数（函数里的局部类的方法）
    不用单独算——外面那层的范围已经把它包住了。
    """
    functions = [span for span in spans if span.kind == "function"]
    regions: list[tuple[int, int, int]] = []

    for span in functions:
        if any(
            other is not span
            and other.start <= span.start
            and span.end <= other.end
            for other in functions
        ):
            continue  # 被别的函数包着，外层会扫到
        opening = _opening_brace(lines, span)
        if opening is None:
            continue  # 只有声明，没有体
        number, column = opening
        regions.append((number, column + 1, span.end))
    return regions


def _opening_brace(lines: list[str], span: cpp.Span) -> tuple[int, int] | None:
    """函数体的 `{` 在哪一行、哪一列；声明（没有体）返回 None。

    从名字那一行往后找，找的是括号深度为 0 的 `{`：签名里的括号、默认
    参数里的花括号都不算数。`{` 之前碰上别的东西（比如只有 `;`）就到头。
    """
    parens = 0
    for number in range(span.line, span.end + 1):
        for column, char in enumerate(lines[number - 1]):
            if char == "(":
                parens += 1
            elif char == ")":
                parens = max(0, parens - 1)
            elif char == "{" and parens == 0:
                return number, column
    return None


def _directive_lines(lines: list[str]) -> set[int]:
    """预处理指令占的行号（含反斜杠续行）。

    和 tools/cpp.py 排除的是同一批：`#define SQUARE(x)` 也长得像调用，
    不排除就会多一条名字是宏的边。
    """
    marked: set[int] = set()
    continuation = False
    for number, line in enumerate(lines, 1):
        if continuation or line.lstrip().startswith("#"):
            marked.add(number)
            continuation = line.rstrip().endswith("\\")
        else:
            continuation = False
    return marked


def _normalized(name: str) -> str:
    """`ns::foo`、`obj->foo` 统一写成点连接（`ns.foo`、`obj.foo`）。"""
    return name.replace("::", ".").replace("->", ".")


def _not_a_call(name: str) -> bool:
    """名字（或它的最后一段）是不是控制语句 / 转型这类不是调用的词。"""
    last = name.rsplit(".", 1)[-1]
    return name in _NOT_A_CALL or last in _NOT_A_CALL


def _looks_like_declaration(line: str, start: int) -> bool:
    """这个名字前面像不像「类型 / 对象 + 名字」——`int x(`、`Point p(`。

    不是解析器，只挡最显眼的一类：函数体里用括号初始化变量很常见，不挡
    就会多出一堆假调用。宁可把 `b * c(3)` 这种真调用也跳过去，不赌。
    """
    prefix = line[:start].rstrip()
    if not prefix:
        return False
    if prefix[-1] in _BEFORE_TYPE:
        return True
    word = _WORD_END.search(prefix)
    return word is not None and word.group() not in _EXPRESSION_WORDS
