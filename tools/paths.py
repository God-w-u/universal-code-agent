"""文件类工具共用的路径安全规则。

规则：每个工具在构造时定一个 root，之后所有路径参数都必须落在 root 以内。

关键在顺序——先 resolve() 再判断。resolve() 会展开 `..` 和符号链接，
所以等它跑完再看结果是不是还在 root 里，`..`、绝对路径、符号链接
这三种逃逸方式就一起被堵住了。
"""

import fnmatch
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


# 一律不读、不列、不搜的**文件名**（支持通配）。
#
# 和上面的 _SENSITIVE_NAMES 是两件事，别混：那一条管的是「root 能不能
# 选在这儿」，只在构造工具时校验一次；这一条管的是「root 里面哪些文件
# 不能碰」，每次遍历 / 读取都要问一遍。
#
# 为什么需要它：这些文件本来就在 root 里，`resolve_in_root` 拦不住也不
# 该拦。但 list_files / search_code / 索引会按内容把它们吐出来，一句
# `search_code "OPENAI_API_KEY"` 就能把密钥原文送进模型上下文——
# 送出去就收不回来了。所以按名字整条排掉：不看内容（看内容本身就等于
# 读过了），也不给「就这一次」的例外。
SENSITIVE_FILES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*.ppk",
    ".netrc",
    ".pgpass",
    "credentials",
    "credentials.json",
    ".htpasswd",
)

# 明确放行：公开的配置模板，值全是空的，属于项目该给人看的一部分。
# 先判这个再判通配——否则 `.env.*` 会把 `.env.example` 一起吃掉。
SENSITIVE_EXCEPTIONS = frozenset({".env.example", ".env.sample", ".env.template"})


def is_sensitive_file(name: str) -> bool:
    """这个**文件名**要不要整条挡掉。

    只看名字，不看路径、不看内容——大小写不敏感（Windows 上尤其要）。

    末尾的点号和空格先削掉：Windows 的文件名本来就不允许以它们结尾，
    `resolve()` 也会把它们规范掉，所以现在每条调用路径都碰不到。但这是
    个隐性契约——将来谁拿一个没 resolve 过的 `entry.name` 来问，`.env `
    这种名字就会与名单擦肩而过。判断本身不该依赖调用方先替它规范化。
    """
    lowered = name.lower().rstrip(" .")
    if lowered in SENSITIVE_EXCEPTIONS:
        return False
    return any(
        fnmatch.fnmatchcase(lowered, pattern) for pattern in SENSITIVE_FILES
    )


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
