"""工具：读取项目里的一个文本文件，返回带行号的内容。

    print('hello')   →   1|print('hello')

带行号是为了让模型能精确引用位置。不加的话它只能自己数行，而它数不准——
实测过：让模型说某段代码在第几行，它给的行号大半是错的。

安全边界复用 tools/paths.py，编码与读取上限复用 tools/textfile.py。
"""

from pathlib import Path

from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_in_root
from tools.textfile import decode_text, read_bytes, split_lines


class ReadFileTool(Tool):
    """读取项目里的一个文本文件。"""

    name = "read_file"
    description = (
        "读取项目里的一个文本文件，返回带行号的内容，每行形如「42|该行内容」。"
        "行号就是文件里真实的行号，可以直接拿去引用位置。"
        "路径相对项目根目录，例如 agent/agent.py。"
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "要读取的文件路径，相对项目根目录",
            "required": True,
        },
    }

    # 最多从磁盘读这么多字节，避免把超大文件整个读进内存
    max_read_bytes = 256 * 1024
    # 最多返回这么多字符，**行号前缀算在里面**——它是真正发给模型的那串东西
    max_return_chars = 40_000

    def __init__(self, root: Path | None = None) -> None:
        self.root = check_root(root)

    def execute(self, path: str) -> str:
        target = resolve_in_root(path, self.root)

        if target.is_dir():
            raise ToolError(f"{path!r} 是目录，不能当文件读（列目录请用 list_files）")
        if not target.is_file():
            raise ToolError(f"文件不存在：{path!r}")

        try:
            raw, too_big = read_bytes(target, self.max_read_bytes)
        except OSError as exc:
            raise ToolError(f"读取文件失败：{exc}") from exc

        lines = split_lines(decode_text(raw, path))
        if not lines:
            return ""  # 空文件，没有行可编

        kept = self._fit(lines)
        body = "\n".join(kept)
        note = self._note(lines, kept, too_big)
        if not note:
            return body
        # 截断了就必须说清楚，否则模型会以为这就是全文
        return f"{body}\n{note}" if body else note

    def _fit(self, lines: list[str]) -> list[str]:
        """能放下几行就编几行的号，放不下的整行丢掉。

        按行丢而不是按字符砍：砍到半行的话，末尾那行是残的，
        「返回了第 1–N 行」这句话也就不成立了。
        """
        width = len(str(len(lines)))
        kept: list[str] = []
        used = 0
        for number, line in enumerate(lines, 1):
            piece = f"{number:>{width}}|{line}"
            cost = len(piece) + (1 if kept else 0)  # 行与行之间的那个换行符
            if used + cost > self.max_return_chars:
                break
            kept.append(piece)
            used += cost
        return kept

    def _note(self, lines: list[str], kept: list[str], too_big: bool) -> str:
        """末尾那句说明。没截断就返回空串。"""
        total = len(lines)
        notes = []

        if too_big:
            notes.append(f"文件超过 {self.max_read_bytes // 1024} KB，只读了开头")

        if not kept:
            # 一行都放不下，说明第一行本身就超长（整份压成一行的文件就是这样）。
            # 这时候不返回正文：硬塞 4 万个字符的乱码只会白占上下文，
            # 说清楚「这文件是一整行」比给半截内容有用。
            notes.append(
                f"第 1 行就有 {len(lines[0])} 个字符，超过单次返回上限"
                f" {self.max_return_chars} 字符，没有返回内容"
            )
        elif len(kept) < total:
            # 字节上限先砍过一刀的话，总行数只能按「读到的部分」说
            scope = "读到的部分" if too_big else "文件"
            where = f"第 1–{len(kept)} 行" if len(kept) > 1 else "第 1 行"
            notes.append(f"{scope}共 {total} 行，只返回了{where}")

        return f"…（{'；'.join(notes)}）" if notes else ""
