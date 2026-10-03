"""工具：改项目里的一个文本文件，只改你指的那一处。

这是第一个会**改**项目文件的工具。前面那些要么只读，要么只往 build/ 里丢产物，
改了源文件就回不去了——所以这里真正要小心的不是「怎么改」，而是
「怎么保证改的**就是**你想改的那一处」。

三道防线：

- **改之前先读**。整份文件读进来、解出来，才谈得上改。读不出来（二进制、
  编码认不得、太大只读了一半）就不改。
- **原文必须唯一**。给一段 old_text，它在文件里只能出现一次。出现 0 次说明
  记错了内容，出现多次说明给的范围不足以定位——两种都不许猜着改。猜错一次
  是静默的：文件改坏了，模型还以为自己成功了。
- **其余部分一个字节都不动**。编码照原样（读的时候试出来是哪种，写回去就用哪种），
  换行照原样（`\\r\\n` 的文件改完还是 `\\r\\n`），不重新排版，结尾有没有换行也不动。

写入走「临时文件 + 原子替换」：要么是改前的，要么是改后的，不会留下半个文件。
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_in_root
from tools.textfile import decode_with_encoding, read_bytes, split_lines

# 按原文替换 / 按行号替换。两种只能选一种。
BY_TEXT = "原文替换"
BY_LINE = "按行替换"

# 最多改多大的文件。和 read_file 同一个上限：只读到了一半就动手，
# 剩下那一半等于没看过——与其改出个半成品，不如让模型换别的办法。
MAX_BYTES = 256 * 1024

# 结果里「修改前 / 修改后」最多各列这么多行，多的用一句话带过。
# 它是给模型看的摘要，不是 diff——完整的改动模型自己会去 read_file 看。
SUMMARY_LINES = 6


@dataclass(frozen=True)
class Change:
    """一次改动。这是「结构化结果」的那一层，execute() 把它渲染成文本。

    text 是**改完之后的整份内容**，还没落盘；before / after 是给人看的那几行
    （已经统一成 \\n，和 read_file 显示的样子一致），只用来写摘要。
    """

    kind: str  # 原文替换 / 按行替换
    text: str  # 改完之后的整份内容
    line: int  # 动到了第几行（1 起，和 read_file 同一套）
    end_line: int  # 到第几行（含）
    before: str  # 原来那一处长什么样
    after: str  # 换成什么样
    hits: int = 1  # 原文匹配到几处。按行号改时没有「几处」这回事，固定 1

    def as_text(self, path: str) -> str:
        """渲染成回给模型的那段。

        四个字段一个不少：改的是哪个文件、动了第几行、原来什么样、现在什么样。
        「成功」也写在里面——失败根本走不到这儿（失败抛 ToolError，
        执行层会把它记成 success=false 并把原因回灌给模型）。
        """
        where = f"第 {self.line} 行"
        if self.end_line != self.line:
            where = f"第 {self.line}–{self.end_line} 行"
        if self.kind == BY_TEXT:
            where += f"（{self.kind}，命中 {self.hits} 处）"
        else:
            where += f"（{self.kind}）"

        return (
            f"结果：成功\n"
            f"文件：{path}\n"
            f"位置：{where}\n"
            f"{_block('修改前', self.before)}\n"
            f"{_block('修改后', self.after)}"
        )


class EditFileTool(Tool):
    """改项目里的一个文本文件。"""

    name = "edit_file"
    description = (
        "改项目里的一个文本文件，只改指定的那一处，别处一个字节都不动。"
        "两种用法，一次只能用一种："
        "给 old_text（要被替换掉的原文）时，它必须在文件里**只出现一次**——"
        "0 次或多次都直接报错，不会猜你指哪一处；"
        "给 start_line / end_line 时按行号改（含两端，行号和 read_file 是同一套）。"
        "new_text 是替换成什么，给空字符串就是删掉。"
        "old_text 从 read_file 读到的内容里原样抄（注意去掉行号和「|」），"
        "拿不准会不会撞车就多带几行上下文，让它唯一。"
        "编码和换行按文件原样保留，不会把整个文件重新排版。"
        "只能改已经存在的文件，不会新建；文件大小有上限，太大了改不了（会明说）。"
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "要改的文件，相对项目根目录，例如 src/app.cpp",
            "required": True,
        },
        "new_text": {
            "type": "string",
            "description": (
                "替换成什么，可以多行；空字符串表示把匹配到的内容删掉"
            ),
            "required": True,
        },
        "old_text": {
            "type": "string",
            "description": (
                "要被替换掉的原文，必须和文件里的内容一模一样（含缩进），"
                "且在文件里只出现一次。和 start_line 只能用一种"
            ),
            "required": False,
        },
        "start_line": {
            "type": "integer",
            "description": (
                "从第几行开始替换，1 起，和 read_file 返回的行号一致。"
                "和 old_text 只能用一种"
            ),
            "required": False,
        },
        "end_line": {
            "type": "integer",
            "description": "替换到第几行（含）；省略就是只改 start_line 这一行",
            "required": False,
        },
    }

    max_bytes = MAX_BYTES

    def __init__(self, root: Path | None = None) -> None:
        self.root = check_root(root)

    def execute(
        self,
        path: str,
        new_text: str,
        old_text: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        by_text = old_text is not None
        by_line = start_line is not None or end_line is not None
        if by_text and by_line:
            raise ToolError(
                "old_text 和 start_line 只能用一种：按原文替换，或者按行号替换。"
                "两个一起给的话，改了哪儿就以不明确的那一个为准了"
            )
        if not by_text and not by_line:
            raise ToolError(
                "得说清楚改哪儿：给 old_text（要替换掉的原文），"
                "或者给 start_line（从第几行开始）"
            )
        if by_line and start_line is None:
            # end_line 单独出现没有意义：从哪几行开始都不知道
            raise ToolError("给了 end_line 却没给 start_line")
        if not isinstance(new_text, str):
            raise ToolError(f"new_text 得是字符串，收到：{new_text!r}")

        target = self._target(path)
        raw, encoding = self._read(target, path)

        if by_text:
            change = self._by_text(raw, old_text, new_text, path)
        else:
            change = self._by_line(
                raw,
                _line(start_line, "start_line"),
                None if end_line is None else _line(end_line, "end_line"),
                new_text,
                path,
            )

        self._write(target, change.text, encoding)
        return change.as_text(self._relative(target))

    def _target(self, path: str) -> Path:
        """把路径参数解析成 root 以内的真文件。

        和别的工具走同一道门（resolve_in_root）：`..`、绝对路径、符号链接
        都会在 resolve() 之后被挡住。这个工具是往里写东西的，逃出去一次
        就不是「看到了不该看的」，而是「改了不该改的」。
        """
        target = resolve_in_root(path, self.root)
        if target.is_dir():
            raise ToolError(f"{path!r} 是目录，不能当文件改")
        if not target.is_file():
            raise ToolError(
                f"文件不存在：{path!r}。edit_file 只改已经存在的文件，不新建"
            )
        return target

    def _read(self, target: Path, path: str) -> tuple[str, str]:
        """读整份文件，返回 (内容, 用的是什么编码)。

        编码要记下来：读的时候是挨个试出来的，写回去的时候不能靠猜——
        GBK 的文件按 UTF-8 写回去，整个文件的中文就全毁了。
        """
        try:
            raw_bytes, too_big = read_bytes(target, self.max_bytes)
        except OSError as exc:
            raise ToolError(f"读取 {path!r} 失败：{exc}") from exc
        if too_big:
            raise ToolError(
                f"{path!r} 超过 {self.max_bytes // 1024} KB，"
                f"edit_file 只改能整份读下来的文件。"
                f"用 run_command 配合专门的工具改，或者先把它拆小"
            )
        return decode_with_encoding(raw_bytes, path)

    # ------------------------------------------------------------------
    # 两种改法
    # ------------------------------------------------------------------

    def _by_text(self, raw: str, old_text: str, new_text: str, path: str) -> Change:
        """按原文替换。原文在文件里必须正好出现一次。"""
        if not old_text:
            # 空串在哪儿都能匹配上，第一处还会是文件开头——这不是「唯一」，
            # 是「到处都算」。放过去的话等于允许了一个「随便改一处」的用法
            raise ToolError("old_text 不能是空字符串——空串每行都能匹配上")

        style = _newline(raw)
        old = _style(old_text, style)
        new = _style(new_text, style)

        hits = raw.count(old)
        if hits == 0:
            raise ToolError(
                f"{path} 里找不到这段原文。一个字都不能差，缩进也算——"
                f"用 read_file 把要改的那几行重新看一眼，"
                f"原样抄过来（行首的「12|」是行号，不要抄进去）"
            )
        if hits > 1:
            places = "、".join(
                str(number) for number in _hit_lines(raw, old)
            )
            raise ToolError(
                f"这段原文在 {path} 里出现了 {hits} 次（第 {places} 行），"
                f"没法确定你要改哪一处。多带几行上下文让它唯一"
                f"（比如把整个函数签名一起带上），"
                f"或者改用 start_line / end_line 按行号改"
            )

        index = raw.index(old)
        return Change(
            kind=BY_TEXT,
            text=raw[:index] + new + raw[index + len(old) :],
            line=raw.count("\n", 0, index) + 1,
            # 用最后一个字符的位置算结束行：原文要是以换行结尾，
            # 那个换行属于它前面那行，不该把行号顶到下一行去
            end_line=raw.count("\n", 0, index + max(len(old) - 1, 0)) + 1,
            before=_flat(old_text),
            after=_flat(new_text),
            hits=hits,
        )

    def _by_line(
        self,
        raw: str,
        start: int,
        end: int | None,
        new_text: str,
        path: str,
    ) -> Change:
        """按行号替换，含两端。行号从 1 起，数法和 read_file 一样。"""
        lines = split_lines(raw)
        last = end if end is not None else start

        if last < start:
            raise ToolError(f"end_line（{last}）比 start_line（{start}）还小")
        if start > len(lines) or last > len(lines):
            raise ToolError(
                f"行号超出范围：{path} 一共 {len(lines)} 行，"
                f"要改的是第 {start}–{last} 行。"
                f"行号以 read_file 返回的为准——每行开头「12|」里的那个数"
            )

        style = _newline(raw)
        starts = _line_starts(raw)
        begin = starts[start - 1]
        # 换掉的范围连着末尾那个换行一起换，否则下一行会被接到替换内容的屁股上
        finish = starts[last] if last < len(starts) else len(raw)
        kept = finish > begin and raw[finish - 1] == "\n"

        replacement = _style(new_text, style)
        if replacement and kept and not replacement.endswith("\n"):
            replacement += style
        # new_text 是空串时上面那个 if 不成立：那几行连同换行一起没了，
        # 「删掉这几行」就该是这个意思，而不是留下一行空白

        return Change(
            kind=BY_LINE,
            text=raw[:begin] + replacement + raw[finish:],
            line=start,
            end_line=last,
            # before 直接用 split_lines 切出来的那几行：和 read_file 显示的
            # 是同一份东西（都已归一成 \n），模型对得上
            before="\n".join(lines[start - 1 : last]),
            after=_flat(new_text),
        )

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------

    def _write(self, target: Path, text: str, encoding: str) -> None:
        """先写临时文件，再原子换过去。

        直接往原文件上写的话，写到一半出事（磁盘满、进程被掐）就留下半个
        文件——那比没改还糟：模型以为改好了，其实源码已经废了。
        临时文件必须和目标是同一个目录，os.replace 才是同一个卷上的原子换名，
        跨目录的话它退化成复制，又不原子了。
        """
        try:
            data = text.encode(encoding)
        except UnicodeEncodeError as exc:
            raise ToolError(
                f"改完的内容没法用 {encoding} 编码写回去（{exc}）。"
                f"这个文件是 {encoding} 的，新内容里有它表示不了的字。"
                f"文件一个字节都没动"
            ) from exc

        try:
            handle = tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            )
        except OSError as exc:
            raise ToolError(
                f"在 {target.parent} 里建不出临时文件，{target.name} 没动：{exc}"
            ) from exc

        temp = Path(handle.name)
        try:
            with handle:
                handle.write(data)
                handle.flush()
                # 刷到盘上再换名：不然系统一崩，换过去的名字指向的可能还是空文件
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise ToolError(
                f"写回 {target.name} 失败，原文件没动：{exc}"
            ) from exc

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


# ---------------------------------------------------------------------------
# 换行、行号这些小工具
# ---------------------------------------------------------------------------


def _flat(text: str) -> str:
    """`\\r\\n` 归一成 `\\n`。和 textfile.split_lines 用的是同一条规矩。"""
    return text.replace("\r\n", "\n")


def _newline(raw: str) -> str:
    """这份文件用哪种换行。两种都有的按多的那一种算。"""
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _style(text: str, newline: str) -> str:
    """把模型给的换行换成这份文件用的那种。

    模型是从 read_file 的输出里抄的原文，那边 `\\r\\n` 已经被抹成 `\\n` 了。
    不换一下的话，CRLF 的文件里一个字都匹配不上——而模型看不出差别在哪儿。
    """
    flat = _flat(text)
    return flat if newline == "\n" else flat.replace("\n", newline)


def _line_starts(raw: str) -> list[int]:
    """每一行在原文里的起点。

    行怎么算和 split_lines 是同一套（只按 `\\n` 切），所以这里算出来的下标
    和 read_file 报的行号能对上。
    """
    starts = [0]
    index = raw.find("\n")
    while index >= 0:
        starts.append(index + 1)
        index = raw.find("\n", index + 1)
    return starts


def _hit_lines(raw: str, old: str) -> list[int]:
    """这段原文都出现在第几行，用来把「找到好几处」说清楚。"""
    numbers: list[int] = []
    index = raw.find(old)
    while index >= 0:
        numbers.append(raw.count("\n", 0, index) + 1)
        index = raw.find(old, index + len(old))
    return numbers


def _line(value: Any, what: str) -> int:
    """行号得是 1 起的整数。True 也是 int，得单独挡掉。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{what} 得是整数，收到：{value!r}")
    if value < 1:
        raise ToolError(f"{what} 从 1 开始，收到：{value}")
    return value


def _block(label: str, text: str) -> str:
    """把一段多行文本缩进着列出来，太长就只列前几行。"""
    if text == "":
        # 删掉东西的时候 after 就是空的，写「（空）」比留两格空白清楚
        return f"{label}：\n  （空）"

    lines = text.split("\n")
    body = "\n".join(f"  {line}" for line in lines[:SUMMARY_LINES])
    if len(lines) > SUMMARY_LINES:
        body += f"\n  …（后面还有 {len(lines) - SUMMARY_LINES} 行）"
    return f"{label}：\n{body}"
