"""文件类工具共用的路径安全规则。

规则：每个工具在构造时定一个 root，之后所有路径参数都必须落在 root 以内。

关键在顺序——先 resolve() 再判断。resolve() 会展开 `..` 和符号链接，
所以等它跑完再看结果是不是还在 root 里，`..`、绝对路径、符号链接
这三种逃逸方式就一起被堵住了。
"""

import re
from pathlib import Path

from config import PROJECT_ROOT
from tools.base import ToolError

# 「明显的系统敏感目录」。只在校验 root 时使用，不拿去匹配 root 内部的路径——
# 项目里有 etc/ 之类的目录是正常的，不该因为这个被拦下来。
# 也刻意不收 home / users / dev / var：它们太常出现在正常项目路径里了。
_SENSITIVE_NAMES = frozenset(
    {
        # Windows
        "windows",
        "system32",
        "syswow64",
        "program files",
        "program files (x86)",
        "programdata",
        "$recycle.bin",
        "system volume information",
        # Unix 的虚拟文件系统与配置目录
        "etc",
        "proc",
        "sys",
        "boot",
    }
)


def _sensitive_hit(path: Path) -> str | None:
    """路径里是否含敏感目录名，返回命中的那一段。"""
    for part in path.parts:
        if part.lower() in _SENSITIVE_NAMES:
            return part
    return None


def check_root(root: Path | None) -> Path:
    """校验并返回工具根目录；省略则用项目根目录。

    牢笼只有在 root 靠谱时才成立，所以这里先把危险的 root 挡掉。
    """
    candidate = Path(root).resolve() if root is not None else PROJECT_ROOT

    # 盘符根（C:\）或 Unix 的 / 会让工具能翻整个磁盘
    if candidate.parent == candidate:
        raise ToolError(f"不能把工具根目录设为磁盘根目录：{candidate}")

    sensitive = _sensitive_hit(candidate)
    if sensitive:
        raise ToolError(f"不能把工具根目录设为系统敏感目录：{sensitive!r}")

    if not candidate.is_dir():
        raise ToolError(f"根目录不存在或不是目录：{candidate}")

    return candidate


def resolve_in_root(raw: str, root: Path) -> Path:
    """把路径参数解析成绝对路径，并确认它没有跑出 root。"""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate

    # resolve() 会展开 .. 和符号链接，所以判断必须放在它之后
    resolved = candidate.resolve()

    if resolved != root and root not in resolved.parents:
        raise ToolError(
            f"路径 {raw!r} 超出允许范围，只能访问 {root} 以内的目录"
        )
    return resolved


# 能被安全地拼进 shell 命令的路径长什么样。
#
# 这是给「命令类工具」（compile / build）用的，只读文件的工具不需要：
# resolve_in_root 管的是「能不能指到项目外面」，管不了
# 「文件名里塞一个 & 会发生什么」——命令是拼成字符串交给 shell 的，
# 一个带 ; 或 & 的文件名就能把后面的东西变成另一条命令。
#
# 白名单而不是黑名单，是因为要漏掉一个特殊字符就全废了。
# 代价是带空格、中文的路径用不了——第一版接受这个限制。
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-/]*$")


def resolve_shell_path(raw: str, root: Path, what: str = "路径") -> Path:
    """解析路径参数，并确认它能安全地拼进 shell 命令。

    两道关：resolve_in_root 挡住跑出 root 的，字符白名单挡住会改变
    命令意思的。what 是给报错用的名字（「源文件」「构建目录」…）。
    """
    target = resolve_in_root(raw, root)
    relative = target.relative_to(root).as_posix()

    if target == root:
        # root 本身是合法目标（build 的 source_root 默认就是它），
        # 但它的相对形式是 "."，过不了下面的白名单——单独放行。
        # 能走到这里说明 resolve() 的结果就是 root，没有逃逸可言。
        return target

    if not _SAFE_PATH.match(relative):
        raise ToolError(
            f"{what}只能用字母、数字、下划线、点、横线和斜杠，"
            f"路径也不能跑到项目外面去，收到：{raw!r}"
        )
    return target
