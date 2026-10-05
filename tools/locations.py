"""工具的返回正文 → 位置信息（path / line / symbol / kind），**只给位置，不给正文**。

为什么要有这一层：工作台要显示「这次调用落在哪个文件的哪一段」，而工具的
返回正文从 13.2 起就**不进 Trace、也不进任何一帧 SSE**——那条分寸不能破
（正文里可能有整份源码、编译输出，甚至密钥）。所以在这里做一次极窄的提取：
只认下面这几个工具**自己固定的**输出格式，只取出位置字段，正文一个字都不
往外带。认不出来就返回空列表——不猜、不推断、不硬凑。

分寸全在这几条上：

- **白名单**：只解析 _PARSERS 里登记的工具。别的工具（run_command /
  run_tests / calculator……）一律返回空。往任意命令输出里捞「文件:行号」会
  捞出一堆似是而非的东西（测试输出里随便一行都可能是）。
- **严格正则、按行匹配**：对不上就跳过那一行，不做模糊匹配。
- **绝不带正文**：search_code 命中的那行代码、compile 的诊断消息，都留在
  正文里，这里只取「哪个文件第几行」。
- **有上限**：最多 MAX_LOCATIONS 条，路径 / 符号名 / 类型都截断——这是给
  界面显示用的，不是给机器读的。

用的人：tools/executor.py（工具跑完、正文还在手里那一刻提取一次），提取
结果随 TraceEvent 走到 api/sse.py，进 tool_result 帧的 locations 字段。
"""

import re
from typing import Any

# 一次工具调用最多带出多少条位置：一条位置在界面上就是一行，50 行够翻的了
MAX_LOCATIONS = 50
MAX_PATH = 200
MAX_SYMBOL = 120
MAX_KIND = 32

# search_code_chunks / call_graph 的符号行：
#   1. index/chunks.py:12-40  ChunkIndex.build  [method]  42.5  命中原因
_SYMBOL_LINE = re.compile(r"^\d+\.\s+(?P<path>.+?):(?P<line>\d+)-(?P<end>\d+)\s+(?P<symbol>\S+)\s+\[(?P<kind>[^\]]+)\]")

# call_graph 对不上符号时退回的调用点行：
#   1. tools/x.py:3  别的名字  （调用点；符号索引里没有定义）
_CALL_SITE_LINE = re.compile(r"^\d+\.\s+(?P<path>.+?):(?P<line>\d+)\s+(?P<symbol>\S+)\s+（调用点")

# search_code 的每一条：agent/agent.py:42: def run(self):
# **只取路径和行号**，冒号后面那截是源码正文，不取
_SEARCH_LINE = re.compile(r"^(?P<path>.+?):(?P<line>\d+): ")

# compile / build 末尾的诊断块：main.cpp:3:5: error: 'x' was not declared
# 列号可有可无（有的编译器不给）；**不取诊断消息**（它可能带代码片段）
_DIAGNOSTIC_LINE = re.compile(
    r"^\s*(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?:\s*(?P<kind>error|warning|note)\b"
)

# edit_file 的头部两行：文件：{path} / 位置：第 {n} 行 或 第 {n}–{m} 行（{kind}…）
# 起止行的「–」是紧贴着两个数字的（tools/edit_file.py 的 as_text 就是这么拼的），
# 后面才跟空格和「行」——分开写成两截，别把单行那个写法一起带沟里
_EDIT_FILE = re.compile(r"^文件：(?P<path>.+)$")
_EDIT_WHERE = re.compile(r"^位置：第 (?P<line>\d+)(?:–(?P<end>\d+))? 行(?:（(?P<kind>[^，）]+))?")


def extract(tool: str, content: str) -> list[dict[str, Any]]:
    """从一次成功的工具调用结果里取出位置信息；认不出来就是空列表。

    失败的结果（错误文本）不用调这里——错误里没有位置，调用方自己判。
    """
    parser = _PARSERS.get(tool)
    if parser is None:
        return []
    if not content:
        return []
    return parser(content)


def _search_code_chunks(content: str) -> list[dict[str, Any]]:
    """每条：path:起止行 + 符号 + 类型（分数和命中原因不取）。"""
    return _collect(content, _SYMBOL_LINE)


def _call_graph(content: str) -> list[dict[str, Any]]:
    """调用者 / 被调用者列表、调用链的每一跳——两种行都认。

    path 模式下这些行**按链的顺序**排（1 是起点，最后一条是终点），前端
    就是照这个顺序画 A → B → C 的。
    """
    found = _collect(content, _SYMBOL_LINE)
    # 对不上符号的那些节点退回调用点（只有 path:line + 名字，没有类型）
    for item in _collect(content, _CALL_SITE_LINE):
        found.append(item)
    return found[:MAX_LOCATIONS]


def _search_code(content: str) -> list[dict[str, Any]]:
    """每条：path:行号（那行代码本身不取）。"""
    return _collect(content, _SEARCH_LINE)


def _diagnostics(content: str) -> list[dict[str, Any]]:
    """compile / build 末尾的诊断块：path:行[:列] + 级别（消息不取）。"""
    return _collect(content, _DIAGNOSTIC_LINE)


def _edit_file(content: str) -> list[dict[str, Any]]:
    """改文件的结果：文件 + 位置（第几行到第几行）。前几行里就有。"""
    path: str | None = None
    for line in content.split("\n")[:8]:
        matched = _EDIT_FILE.match(line)
        if matched is not None:
            path = _clean(matched.group("path"), MAX_PATH)
            break
    if path is None:
        return []

    for line in content.split("\n")[:8]:
        matched = _EDIT_WHERE.match(line)
        if matched is None:
            continue
        item: dict[str, Any] = {"path": path, "line": _int(matched.group("line"))}
        end = matched.group("end")
        if end is not None:
            item["end_line"] = _int(end)
        kind = matched.group("kind")
        if kind is not None:
            item["kind"] = _clean(kind, MAX_KIND)
        return [item]
    # 有「文件：」却没有「位置：」——格式变了，给一条只有路径的，不编行号
    return [{"path": path, "line": 0}]


def _collect(content: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    """按行套一个正则，把匹配上的那些收成位置；够 MAX_LOCATIONS 就停。"""
    found: list[dict[str, Any]] = []
    for line in content.split("\n"):
        matched = pattern.match(line)
        if matched is None:
            continue
        item = _item(matched)
        if item is not None:
            found.append(item)
        if len(found) >= MAX_LOCATIONS:
            break
    return found


def _item(matched: re.Match[str]) -> dict[str, Any] | None:
    """一条匹配 → 一个位置字典；路径或行号不成立就作废。"""
    groups = matched.groupdict()
    path = _clean(groups.get("path") or "", MAX_PATH)
    line = _int(groups.get("line") or "0")
    if not path or line <= 0:
        return None

    item: dict[str, Any] = {"path": path, "line": line}
    if groups.get("end") is not None:
        end = _int(groups["end"])
        if end > line:
            item["end_line"] = end
    symbol = groups.get("symbol")
    if symbol:
        item["symbol"] = _clean(symbol, MAX_SYMBOL)
    kind = groups.get("kind")
    if kind:
        item["kind"] = _clean(kind, MAX_KIND)
    return item


def _clean(value: str, limit: int) -> str:
    """去空白 + 截断：路径 / 名字 / 类型都只是给界面显示的。"""
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


# 工具名 → 解析函数。**没登记的工具一律没有位置**（见模块说明的白名单那一条）
_PARSERS = {
    "search_code_chunks": _search_code_chunks,
    "call_graph": _call_graph,
    "search_code": _search_code,
    "compile": _diagnostics,
    "build": _diagnostics,
    "edit_file": _edit_file,
}
