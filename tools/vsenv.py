"""找 Visual Studio，并把它那套编译环境取出来。

cl.exe 和 msbuild.exe 都不在 PATH 上。要用它们得先跑一遍 VS 自带的
vcvars64.bat——那个脚本会把 PATH / INCLUDE / LIB 这些变量设好，
编译器才知道去哪儿找头文件和库。这个模块就是干这件事的。

定位 VS 靠 vswhere.exe：微软官方的查询工具，装在安装器自己的目录里，
位置和 VS 的版本、装在哪都无关。VS 装了几个、装的哪一版由它回答，
所以这个文件里不出现任何具体的 VS 版本号或安装路径。

取环境要起一次 cmd 跑 vcvars（这台机器上实测 2.5 秒），所以结果缓存。
"""

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from tools.base import ToolError
from tools.textfile import decode_text

# 筛掉没装 C++ 的 VS：只要没装这个组件，就没有 cl.exe
VC_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

# vcvars64.bat 在 VS 安装目录里的相对位置。它是 VS 自己的固定布局，
# 跟版本无关——2022 是这样，2019、2017 也是这样。
VCVARS = Path("VC") / "Auxiliary" / "Build" / "vcvars64.bat"

# 找 VS 和跑 vcvars 的超时。实测 vcvars 约 2.5 秒，
# 但机器忙、或者 VS 装在慢盘上时会慢不少，留宽一点。
DISCOVERY_TIMEOUT = 120.0


def find_vswhere() -> Path | None:
    """vswhere.exe 在哪；没有就返回 None。

    它的位置是安装器定的，和 VS 装在哪无关。取 ProgramFiles(x86) 是为了
    不写死 C 盘——环境变量本来就在，用它更老实。
    """
    base = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    path = Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return path if path.is_file() else None


def find_vcvars() -> Path | None:
    """vcvars64.bat 在哪；没有 VS 就返回 None。"""
    vswhere = find_vswhere()
    if vswhere is None:
        return None

    result = _run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            VC_COMPONENT,
            "-property",
            "installationPath",
        ]
    )
    if result is None or result.returncode != 0:
        return None

    # 一台机器上装了好几个 VS 时 -latest 只给一个，但输出仍可能有多行，
    # 逐行试一遍，谁先拼出真实存在的 vcvars 就是谁
    for line in _text(result.stdout).splitlines():
        candidate = Path(line.strip()) / VCVARS
        if line.strip() and candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def vcvars_env() -> dict[str, str] | None:
    """跑一遍 vcvars64.bat，把它设好的环境变量拿回来；没有 VS 则返回 None。

    缓存住了：这一下实测 2.5 秒，每次编译都重跑纯属浪费。
    要强制重新找（测试里要）就调 reset_cache()。
    """
    vcvars = find_vcvars()
    if vcvars is None:
        return None

    # vcvars 自己会往 stdout 打一段横幅，>nul 把它挡掉，
    # 剩下的就全是 set 吐出来的 KEY=VALUE。
    # 这里必须过一层 cmd：.bat 只有 cmd 能跑。
    result = _run(f'cmd /c "call "{vcvars}" >nul && set"', shell=True)
    if result is None or result.returncode != 0:
        return None

    env: dict[str, str] = {}
    for line in _text(result.stdout).splitlines():
        # 值里可以有 =（路径里出现等号是常事），所以只切第一个
        key, separator, value = line.partition("=")
        # 名字是空的那种行要丢掉，它们长这样：`=C:=C:\some\dir`。
        # 那是 cmd 给每个盘记「当前目录」用的，名字以 = 开头，
        # 切出来正好是空串——编译器用不上，塞进子进程环境里只会添乱。
        if separator and key:
            env[key] = value
    return env or None


def find_program(name: str) -> str | None:
    """在 VS 那套环境里找一个可执行文件；没有 VS 就返回 None。

    先看当前 PATH——如果 agent 本来就是在「Developer Command Prompt」里
    跑起来的，cl 本来就能直接找到，没必要再去起一次 vcvars。
    """
    found = shutil.which(name)
    if found:
        return found

    env = vcvars_env()
    if env is None:
        return None
    return shutil.which(name, path=env.get("PATH", ""))


def reset_cache() -> None:
    """丢掉缓存，下次重新找一遍 VS。测试要，运行时基本用不上。"""
    vcvars_env.cache_clear()


def _run(
    command: str | list[str], shell: bool = False
) -> subprocess.CompletedProcess | None:
    """跑一条固定的内部命令；起不来（超时、权限、找不到）返回 None。

    失败一律返回 None 而不是抛异常：这个模块的调用方只想知道
    「VS 能不能用」，找不到就是找不到，不该让整个工具调用炸掉。
    """
    try:
        return subprocess.run(
            command,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=DISCOVERY_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _text(raw: bytes) -> str:
    """解命令输出。

    VS 那些脚本吐的是本地编码（中文 Windows 上是 GBK），和源码一样不保证。
    解不开就退化成替换字符继续——路径解错一位的结果只是「找不到 VS」，
    比整条链路炸掉好。
    """
    try:
        return decode_text(raw, "VS 的输出")
    except ToolError:
        return raw.decode("utf-8", errors="replace")
