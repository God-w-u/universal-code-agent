"""遍历项目文件：哪些该看，哪些该跳。

list_files 和 search_code 都要走这一套，所以抽出来共用。
"""

from collections.abc import Iterator
from pathlib import Path

# 这些目录里的文件不是项目源码，列出来/搜出来只会淹没上下文。
# 按完整目录名匹配，不做前缀或子串匹配，所以 build_tools/ 不会被误伤。
IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".tox",
        ".eggs",
        "node_modules",
        ".idea",
        ".vscode",
        "dist",
        "build",
        # 测试素材（9.5）：给评估用的迷你 C++ 工程，不是项目自己的源码——
        # 索引进来只会淹没检索结果。评估入口单独给它建一份索引。
        "fixtures",
    }
)


def iter_files(directory: Path) -> Iterator[Path]:
    """递归产出目录下的文件，跳过 IGNORE_DIRS 和符号链接。

    符号链接可能指向 root 外面，一律不跟随。目录读不了（没权限等）
    会把 OSError 抛给调用方，由调用方决定是整体失败还是跳过。
    """
    pending = [directory]

    while pending:
        current = pending.pop()
        for entry in current.iterdir():
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in IGNORE_DIRS:
                    pending.append(entry)
            elif entry.is_file():
                yield entry
