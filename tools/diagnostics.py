"""把编译器吐出来的那段文本读成一条条结构化诊断。

编译失败时模型最需要的就三件事：哪个文件、第几行、什么问题。可偏偏这三个
编译器把这件事写成了两种格式：

    msvc.cpp(3,3): error C2065: “x”: 未声明的标识符          MSVC
    msvc.cpp:3:3: error: 'x' was not declared               GCC / Clang

模型当然看得懂，但它每读一次都要重新认一遍这个格式，还得自己从一屏文本里
把这几行挑出来（cl 一次能喷几十条连带错误）。认格式这件事只该做一次。

**行号可以直接拿去用 read_file 核对，不用换算**：编译器报的行号和
read_file / search_code 数行号用的是同一套规则（都从 1 开始），
这条是整个解析器里最容易出错也最要紧的地方。
路径同理——命令里的源文件本来就写成相对 root 的路径，编译器原样回显，
所以 file 字段直接就是 read_file 想要的那个相对路径。

认不出来的行不丢，原样收在 Report.other 里：编译器还会打源文件上下文、
"In file included from ..."、命令行回显这些东西，它们有用，只是不是诊断。
"""

import re
from dataclasses import dataclass

from tools.textfile import split_lines

# 级别只有这三种。
# MSVC 的 "fatal error C1083" 和 GCC 的 "fatal error:" 都归一成 error——
# 「编译当场停下」这件事已经写在错误码里了（C1083 之类），
# 再多一个级别，调用方就得为它多写一层判断，而判断的结果从来都一样。
ERROR = "error"
WARNING = "warning"
NOTE = "note"

# 渲染时按这个顺序排级别，不跟着诊断在原文里的先后走
_ORDER = (ERROR, WARNING, NOTE)

# 有 error 或 warning 才算「出了问题」。只有 note 时不值得单独给模型看一段：
# note 说的是「这个标识符是在那儿声明的」，脱离它跟的那条 error 没有意义。
_PROBLEM = (ERROR, WARNING)


# MSVC：`file.cpp(12): error C2143: ...`，带列号时是 `file.cpp(12,5): ...`。
# 错误码是「字母 + 数字」——C2143 是编译器、D8000 是命令行、LNK2019 是链接器，
# 都长这样。它可以没有，所以整段是可选的。
_MSVC = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+)(?:,(?P<column>\d+))?\)\s*:\s*"
    r"(?P<level>fatal error|error|warning|note)\s*"
    r"(?:(?P<code>[A-Za-z]+\d+)\s*)?:\s*(?P<message>.*)$"
)

# GCC / Clang：`file.cpp:12:5: error: ...`，不带列号时是 `file.cpp:12: error: ...`。
# 开头那个 `(?:[A-Za-z]:)?` 是给 Windows 绝对路径留的（`C:\src\a.cpp`）：
# 不单独认盘符的话，后面的 `[^:]*` 会在 `C:` 那儿就断掉，把盘符当成文件名。
_GCC = re.compile(
    r"^(?P<file>(?:[A-Za-z]:)?[^:]*):(?P<line>\d+)"
    r"(?::(?P<column>\d+))?:\s*"
    r"(?P<level>fatal error|error|warning|note)\s*:\s*(?P<message>.*)$"
)

# GCC / Clang 把「是哪个开关报告的这一条」挂在消息末尾：`... [-Wunused-variable]`。
# 单独抽出来当 code，消息里就只剩下人话了。
# 要求以 - 开头是有意的：模板诊断末尾的 `[with T = int]` 不是开关，
# 不能当成错误码抽走。
_TAIL_CODE = re.compile(r"\s*\[(?P<code>-[^\]]*)\]\s*$")


@dataclass(frozen=True)
class Diagnostic:
    """一条诊断。file 和 line 可以直接交给 read_file 去核对。"""

    file: str
    line: int
    column: int | None
    level: str
    message: str
    code: str | None = None

    def as_line(self) -> str:
        """渲染成一行。

        统一用 `文件:行[:列]: 级别 [错误码]: 消息` 的写法，也就是 g++ 那一套——
        不管这条原来是 MSVC 报的还是 GCC 报的，模型看到的形状都一样。
        行号落在前缀里，模型可以拿这一行直接去 read_file 核对。
        """
        where = f"{self.file}:{self.line}"
        if self.column is not None:
            where += f":{self.column}"
        code = f" {self.code}" if self.code else ""
        return f"{where}: {self.level}{code}: {self.message}"


@dataclass(frozen=True)
class Report:
    """一次解析的全部结果：认出来的，和没认出来的。"""

    diagnostics: tuple[Diagnostic, ...] = ()
    other: tuple[str, ...] = ()

    @property
    def has_problems(self) -> bool:
        """有没有 error 或 warning。只有 note 时返回 False。"""
        return any(item.level in _PROBLEM for item in self.diagnostics)


def parse(text: str) -> Report:
    """把一段文本里所有认得的诊断挑出来，认不得的行原样收着。

    切行用的是 textfile.split_lines()——和 read_file、search_code 同一套规矩。
    自己再写一遍切行，就等于又多了一个「行号怎么数」的说法，
    而这条正是本模块最不能出错的地方。
    """
    found: list[Diagnostic] = []
    other: list[str] = []

    for raw in split_lines(text):
        # 前后空白清掉再认。cl 的输出里偶尔会挂一个 \r 在行首，
        # 不strip 的话 `^` 锚点就对不上，整条诊断会被漏掉
        hit = _match(raw.strip())
        if hit is None:
            other.append(raw)
        else:
            found.append(hit)

    return Report(tuple(found), tuple(other))


def render(report: Report) -> str:
    """把诊断排成给模型看的一段。没有诊断时返回空串。

    保持原文里的先后顺序：note 跟在它解释的那条 error 后面，
    按级别重排会把这条线索扯断。
    """
    if not report.diagnostics:
        return ""

    listed = "\n".join(f"  {item.as_line()}" for item in report.diagnostics)
    total = len(report.diagnostics)
    return f"诊断（共 {total} 条：{_counts(report.diagnostics)}）：\n{listed}"


def _match(line: str) -> Diagnostic | None:
    """这一行是哪家的诊断；都不是就返回 None。

    MSVC 的先试：它的 `file.cpp(12): error C2143:` 里那个 `(12)` 是 GCC
    那套匹配不了的，反过来 GCC 的行里也没有这种括号加数字的写法，
    所以两条规则不会抢同一行。
    """
    hit = _MSVC.match(line)
    if hit is not None:
        return Diagnostic(
            file=_path(hit["file"]),
            line=int(hit["line"]),
            column=_number(hit["column"]),
            level=_level(hit["level"]),
            message=hit["message"].strip(),
            code=hit["code"],
        )

    hit = _GCC.match(line)
    if hit is None:
        return None

    message, code = _split_code(hit["message"].strip())
    return Diagnostic(
        file=_path(hit["file"]),
        line=int(hit["line"]),
        column=_number(hit["column"]),
        level=_level(hit["level"]),
        message=message,
        code=code,
    )


def _level(raw: str) -> str:
    """归一级别。"""
    return ERROR if raw == "fatal error" else raw


def _number(raw: str | None) -> int | None:
    return None if raw is None else int(raw)


def _path(raw: str) -> str:
    """路径统一成 `/` 分隔。

    和 list_files / search_code 返回的路子保持一致——模型拿到的手感应当是
    一样的，不然它得先判断「这条路是哪个工具给的」才知道要不要转斜杠。

    代价是 Unix 上文件名里真有一个反斜杠时会被转错。那种文件几乎不存在，
    而这个项目当前跑在 Windows 上，编译器回显的每一段路径都是反斜杠。
    """
    return raw.strip().replace("\\", "/")


def _split_code(message: str) -> tuple[str, str | None]:
    """把 GCC/Clang 消息末尾的 `[-Wxxx]` 摘出来当错误码。"""
    hit = _TAIL_CODE.search(message)
    if hit is None:
        return message, None
    return message[: hit.start()].rstrip(), hit["code"]


def _counts(items: tuple[Diagnostic, ...]) -> str:
    """排成 `2 个 error，1 个 warning` 这样一段，只写出现过的级别。"""
    total: dict[str, int] = {}
    for item in items:
        total[item.level] = total.get(item.level, 0) + 1
    return "，".join(
        f"{total[level]} 个 {level}" for level in _ORDER if level in total
    )
