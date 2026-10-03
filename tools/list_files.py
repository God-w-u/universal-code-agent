"""工具：递归列出目录下的文件。

安全边界复用 tools/paths.py，遍历规则复用 tools/tree.py。
"""

from pathlib import Path

from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_in_root
from tools.tree import iter_files


class ListFilesTool(Tool):
    """递归列出目录下的文件，返回相对 root 的路径。"""

    name = "list_files"
    description = (
        "递归列出一个目录下的所有文件，返回相对项目根目录的路径列表。"
        "会自动跳过 .git、.venv、__pycache__、node_modules、build 等噪音目录。"
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "要列出的目录，相对项目根目录；省略则列出整个项目",
            "required": False,
        },
    }

    # 单词最多返回多少个文件，防止一次把上下文撑爆
    max_files = 500

    def __init__(self, root: Path | None = None) -> None:
        self.root = check_root(root)

    def execute(self, path: str = ".") -> str:
        directory = resolve_in_root(path, self.root)

        if not directory.is_dir():
            raise ToolError(f"不是目录：{path!r}")

        try:
            files = sorted(iter_files(directory))
        except OSError as exc:
            raise ToolError(f"读取目录失败：{exc}") from exc

        if not files:
            return f"（{path!r} 下没有文件）"

        lines = [self._relative(item) for item in files[: self.max_files]]
        if len(files) > self.max_files:
            lines.append(
                f"…（共 {len(files)} 个文件，只显示前 {self.max_files} 个）"
            )
        return "\n".join(lines)

    def _relative(self, path: Path) -> str:
        """转成相对 root 的路径，统一用 / 分隔，方便直接喂给后续工具。"""
        return path.relative_to(self.root).as_posix()
