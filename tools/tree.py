"""遍历项目文件：哪些该看，哪些该跳。

list_files、search_code、code_summary 和索引都要走这一套，所以抽出来共用。

两道过滤：

- **目录**：IGNORE_DIRS 里的不进（.git、缓存、构建产物、评估素材……），
  `repair_history/` 也在里面——那些 JSON 存的是任务原文和编译诊断，是
  「发生过什么」的记录，不是项目源码，收进检索 / 索引只会把历史当代码看。
- **文件**：敏感文件（`.env`、密钥、证书……）不产出。名单和判定在
  `tools/paths.py` 的 `is_sensitive_file`——那里的理由说得更全，读文件
  的 `read_file` 也用同一个判断。

还有一条**不跟随链接**：符号链接和 Windows junction 都可能指向 root
外面，两者都不进。junction 要单独认——`Path.is_symlink()` 对 junction
返回 False（它走的是 reparse point 的另一个标记），只查 is_symlink()
会把它当成普通目录跟进去。
"""

from collections.abc import Iterator
from pathlib import Path

from tools.paths import is_sensitive_file

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
        # 修复历史（13.8）：每个 run 一个 JSON，存着任务原文、编译诊断和
        # 本地路径。是运行产物，不是源码；被检索 / 索引收进去等于把历史
        # 当代码看，还可能把上几轮的任务原文带进这一轮的上下文。
        "repair_history",
    }
)


def _is_link(entry: Path) -> bool:
    """符号链接 / Windows junction / 其它重解析点：一律算「链出去的门」。

    `is_symlink()` 只认符号链接——Windows 的 junction（`mklink /J`，
    建它不需要管理员权限）返回 False，但 `resolve()` 同样会展开它，
    一样能指向 root 外面。两个都算上。
    """
    if entry.is_symlink():
        return True
    is_junction = getattr(entry, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def iter_files(directory: Path) -> Iterator[Path]:
    """递归产出目录下的文件，跳过 IGNORE_DIRS、敏感文件和链接。

    符号链接和 junction 都可能指向 root 外面，一律不跟随。目录读不了
    （没权限等）会把 OSError 抛给调用方，由调用方决定是整体失败还是跳过。
    """
    pending = [directory]

    while pending:
        current = pending.pop()
        for entry in current.iterdir():
            if _is_link(entry):
                continue
            if entry.is_dir():
                if entry.name not in IGNORE_DIRS:
                    pending.append(entry)
            elif entry.is_file() and not is_sensitive_file(entry.name):
                yield entry
