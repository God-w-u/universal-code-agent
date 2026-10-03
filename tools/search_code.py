"""工具：在项目里按关键词搜索文件内容。

安全边界复用 tools/paths.py，遍历规则复用 tools/tree.py，
解码逻辑复用 tools/textfile.py。
"""

from pathlib import Path

from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_in_root
from tools.textfile import decode_text, read_bytes
from tools.tree import iter_files


class SearchCodeTool(Tool):
    """在项目内按关键词搜索文件内容。"""

    name = "search_code"
    description = (
        "在项目里搜索包含指定关键词的行，返回「相对路径:行号: 该行内容」。"
        "按子串匹配，不是正则；默认忽略大小写。"
        "可以用 path 把搜索限定在某个子目录里，缩小范围能搜得更快也更准。"
        "会自动跳过 .git、.venv、__pycache__、node_modules 等噪音目录，"
        "以及二进制和无法解码的文件。"
    )
    parameters = {
        "keyword": {
            "type": "string",
            "description": "要搜索的关键词，按子串匹配",
            "required": True,
        },
        "path": {
            "type": "string",
            "description": "限定搜索的目录，相对项目根目录；省略则搜索整个项目",
            "required": False,
        },
        "case_sensitive": {
            "type": "boolean",
            "description": "是否区分大小写，默认否",
            "required": False,
        },
    }

    # 单个文件最多读这么多字节；更大的文件只搜开头，并在结果里说明
    max_file_bytes = 1024 * 1024
    # 最多返回多少条匹配，避免撑爆 LLM 上下文
    max_matches = 100

    def __init__(self, root: Path | None = None) -> None:
        self.root = check_root(root)

    def execute(
        self, keyword: str, path: str = ".", case_sensitive: bool = False
    ) -> str:
        if not keyword:
            raise ToolError("搜索关键词不能为空")

        directory = resolve_in_root(path, self.root)

        if not directory.is_dir():
            raise ToolError(f"不是目录：{path!r}")

        try:
            files = sorted(iter_files(directory))
        except OSError as exc:
            # 目录读不动（权限、被占用等）就转成 ToolError。
            # 裸抛 OSError 会绕过「错误回灌给模型」这条路，直接让 Agent 崩掉。
            raise ToolError(f"读取目录失败：{exc}") from exc

        matches: list[str] = []
        truncated_files: list[str] = []

        for item in files:
            # 路径统一相对 root 报，不是相对搜索目录——
            # 否则限定子目录搜出来的结果喂给 read_file 会对不上。
            relative = item.relative_to(self.root).as_posix()

            try:
                raw, too_big = read_bytes(item, self.max_file_bytes)
            except OSError:
                continue  # 读不了就跳过，不能因为一个文件让整次搜索失败

            try:
                text = decode_text(raw, relative)
            except ToolError:
                continue  # 二进制 / 解不开的文件，同样跳过

            if too_big:
                truncated_files.append(relative)

            matches.extend(self._match_lines(relative, text, keyword, case_sensitive))
            if len(matches) > self.max_matches:
                break  # 多留一条只用来判断"被截断了"，不需要继续扫

        return self._render(keyword, matches, truncated_files)

    def _match_lines(
        self, relative: str, text: str, keyword: str, case_sensitive: bool
    ) -> list[str]:
        """找出 text 里含关键词的行，格式化成「路径:行号: 内容」。"""
        needle = keyword if case_sensitive else keyword.lower()

        found = []
        # 用 split("\n") 而不是 splitlines()：后者把 \x0c 之类的字符也当换行，
        # 那样报出来的行号会和编译器数的行号对不上。
        for number, line in enumerate(text.split("\n"), start=1):
            haystack = line if case_sensitive else line.lower()
            if needle in haystack:
                found.append(f"{relative}:{number}: {line.strip()}")
        return found

    def _render(
        self, keyword: str, matches: list[str], truncated_files: list[str]
    ) -> str:
        if not matches:
            body = f"（没有找到包含 {keyword!r} 的行）"
        else:
            body = "\n".join(matches[: self.max_matches])
            if len(matches) > self.max_matches:
                body += f"\n…（匹配过多，只显示前 {self.max_matches} 条）"

        if truncated_files:
            # 不说的话，模型会以为自己搜过全文
            body += (
                f"\n…（有 {len(truncated_files)} 个文件超过 "
                f"{self.max_file_bytes // 1024} KB，只搜索了开头部分）"
            )
        return body
