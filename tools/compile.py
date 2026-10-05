"""工具：把 C++ 源文件编译成可执行文件。

执行部分同样一行没自己写，全交给 RunCommandTool（subprocess、超时、
输出解码、截断都已经在那边测过了）。

和 run_tests 的区别在于：那边的命令是钉死的常量，这里的命令是**拼出来**的。
模型给的是「编哪个文件、用什么编译器、加什么参数」，不是命令，
所以这个文件里真正要小心的是
「拼进命令里的那几个词，会不会反过来改变命令的意思」。

MSVC 那一条路多两件事，都在别处：
- 找 VS、铺 vcvars 环境在 tools/vsenv.py；
- 参数从 g++ 写法翻成 cl 写法在本文件下半部分，翻不了的一律拒绝——
  cl 碰到不认识的参数只打一句 warning 就跳过，悄悄按错的编比报错糟得多。
"""

import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from tools import diagnostics, vsenv
from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_shell_path
from tools.run_command import RunCommandTool

if TYPE_CHECKING:  # 只为注解；省略 sandbox 时由 RunCommandTool 自己装配
    from sandbox.base import Sandbox

# 按顺序找：g++ 最优先，然后 clang++，最后才是 MSVC 的 cl。
# cl 排最后是因为用它得先铺一遍 vcvars 环境（这台机器上实测 2.5 秒），
# 机器上真装了 g++ 的话没理由付这个成本。
COMPILERS = ("g++", "clang++", "cl")

# MSVC 的编译器就这一个名字。它的参数写法和 g++ 完全不同，单独走一条路。
MSVC = "cl"

# 编译单个文件默认等多久。比 run_command 的 30 秒宽松一点：
# 模板重的 C++ 文件慢起来是真的慢，而慢只是多等一会儿。
DEFAULT_COMPILE_TIMEOUT = 60.0

# 允许出现在 flags 里的字符。比路径宽松一点（flag 里常出现 = . , +），
# 但一样不放引号、分号、与或、重定向、反引号、$ 和括号进来。
# 斜杠和冒号是给 MSVC 写法留的（/std:c++17）——它们不是 shell 的元字符。
_SAFE_FLAGS = re.compile(r"^[A-Za-z0-9_\-=.,+/: ]*$")

# 这几个 flag 决定产物写到哪儿。输出路径是工具自己的事，
# 不接受模型从 flags 里把它们再指定一遍。
# 按前缀匹配，所以 -o other / -oother / --output=x / -MFdeps.txt 都会被拦下；
# 注意 -O2（大写）是优化级别，和 -o（小写）不是一回事，不会误伤。
_OUTPUT_FLAGS = ("-o", "--output", "-MF", "-MT", "-MQ")

# cl 那边把产物写到哪儿的参数。它不分大小写，比对时统一转小写。
# /Fe 是 exe、/Fo 是 .obj、/Fd 是 pdb、/Fa 是汇编清单、/Fi 是预处理结果。
_MSVC_OUTPUT_FLAGS = ("/fe", "/fo", "/fd", "/fa", "/fi", "/fr", "/fm")


def find_compiler() -> str | None:
    """找一个能用的 C++ 编译器，一个都没有则返回 None。"""
    for name in COMPILERS:
        if _which(name):
            return name
    return None


def _which(name: str) -> str | None:
    """这个编译器能不能用。

    cl 要多问一句：它一般不在 PATH 上，装了 VS 也不在——得从 VS 那套
    环境里找（tools/vsenv.py）。别的编译器就是普通的 PATH 查找。
    """
    found = shutil.which(name)
    if found or name != MSVC:
        return found
    return vsenv.find_program(MSVC)


class CompileTool(Tool):
    """编译单个 C++ 源文件。"""

    name = "compile"
    description = (
        "把一个 C++ 源文件编译成可执行文件。"
        "省略 compiler 就让工具自己挑（依次找 g++、clang++、Windows 上的 cl.exe）。"
        "cl.exe 不在 PATH 上也没关系，工具会自己去找 Visual Studio 并把编译环境铺好。"
        "flags 一律按 g++ 的写法给（例如 -std=c++17 -O2 -Wall）；"
        "选到 cl.exe 时工具会翻成 MSVC 的写法，翻不了的参数会明确报错，"
        "那时改用 run_command 自己拼 cl 命令。"
        "编译不通过（退出码非 0）也会正常返回结果，编译器报的错就在输出里，"
        "不算调用失败——注意 MSVC 的报错打在标准输出里，不在标准错误里。"
        "结果末尾还会另附一段解析好的诊断，一行一条，写成"
        "「文件:行:列: 级别: 消息」；那里的行号可以直接拿去用 read_file 核对。"
        "只能编单个文件；不支持 CMake、Makefile 和多文件工程，"
        "那些用 build 或 run_command。"
        "cl 会在源文件旁边留一个 .obj，这是它的正常行为。"
    )
    parameters = {
        "source": {
            "type": "string",
            "description": "要编译的 C++ 源文件，相对项目根目录，例如 main.cpp",
            "required": True,
        },
        "output": {
            "type": "string",
            "description": (
                "输出的可执行文件名，相对项目根目录；"
                "省略则用源文件同目录下的同名文件（main.cpp -> main）"
            ),
            "required": False,
        },
        "compiler": {
            "type": "string",
            "description": (
                f"用哪个编译器，只能是 {' 或 '.join(COMPILERS)}；"
                "省略则优先用装配工具时指定的那个，再不行就自动找"
            ),
            "required": False,
        },
        "flags": {
            "type": "string",
            "description": (
                "附加的编译参数，空格分隔，按 g++ 的写法，"
                "例如 -std=c++11 -O2 -Wall。"
                "不能出现 -o：输出到哪儿由 output 参数决定"
            ),
            "required": False,
        },
        "timeout": {
            "type": "number",
            "description": f"最多等多少秒，默认 {DEFAULT_COMPILE_TIMEOUT:g}",
            "required": False,
        },
    }

    def __init__(
        self,
        root: Path | None = None,
        compiler: str | None = None,
        timeout: float = DEFAULT_COMPILE_TIMEOUT,
        sandbox: "Sandbox | None" = None,
    ) -> None:
        self.root = check_root(root)
        # None 表示「到执行的时候再自动找」——放在构造时找的话，
        # 机器上没装编译器会让 main.build_agent() 直接就装配不起来
        self.compiler = compiler
        # sandbox 只是转交给内层工具：省略时它按 root 自己装一个
        # LocalSandbox（见 RunCommandTool），这里给的是测试口子，
        # 也是以后换实现（Docker）的接缝
        self.command_tool = RunCommandTool(
            root=self.root, timeout=timeout, sandbox=sandbox
        )

    def compile(
        self,
        source: str,
        output: str | None = None,
        compiler: str | None = None,
        flags: str = "",
        timeout: float | None = None,
    ) -> str:
        """编译一个源文件，把 RunCommandTool 的结果原样返回。"""
        source_path = resolve_shell_path(source, self.root, "源文件")
        if not source_path.is_file():
            raise ToolError(f"源文件不存在：{source!r}")

        if output is None:
            # 沿用源文件的目录和名字：src/app.cpp -> src/app
            # 它是从已经校验过的路径推出来的，不用再过一遍
            target = source_path.with_suffix("")
        else:
            target = resolve_shell_path(output, self.root, "输出文件")

        name = self._compiler(compiler)
        command = self._command(name, source_path, target, self._checked_flags(flags))

        # 直接调 execute()：command 是这个工具自己拼的，不需要再过一遍
        # 内层工具的参数校验（显式传 timeout=None 反而会被判成「不是 number」）
        result = self.command_tool.execute(
            command, timeout=timeout, env=self._env(name)
        )
        return self._with_diagnostics(result)

    def _with_diagnostics(self, result: str) -> str:
        """在原始输出后面附一段解析好的诊断。

        原始输出一个字符都不动——编译器报的原文模型仍然看得到，这里只是
        在后面另起一段把「哪个文件、第几行、什么问题」摆出来。
        编译过了但带 warning 时也附：warning 一样是要改代码才能消掉的。

        cl 的报错打在标准输出里、gcc 的打在标准错误里，两种都在传进来的
        这段文本里（run_command 两路都收），所以这里不用挑，整段一起解。
        """
        report = diagnostics.parse(result)
        if not report.has_problems:
            # 一条都没认出来，或者只有 note——后者的意思是「它是在那儿声明的」，
            # 脱离它跟的那条 error 单独看没有意义
            return result
        return f"{result}\n\n{diagnostics.render(report)}"

    def execute(
        self,
        source: str,
        output: str | None = None,
        compiler: str | None = None,
        flags: str = "",
        timeout: float | None = None,
    ) -> str:
        return self.compile(source, output, compiler, flags, timeout)

    def _command(
        self, compiler: str, source: Path, target: Path, flags: str
    ) -> str:
        """把「用哪个编译器、编哪个文件、编到哪儿、加什么参数」拼成一条命令。

        g++/clang++ 和 cl 的参数写法不一样，这里是唯一分岔的地方。
        """
        source_name = self._relative(source)
        target_name = self._relative(target)

        if compiler == MSVC:
            parts = [compiler, *_MSVC_ALWAYS]
            if flags:
                parts.append(_to_msvc(flags))
            parts += [source_name, f"/Fe:{target_name}"]
            return " ".join(parts)

        # 顺序是刻意的：flags 在最前，"-o 输出" 在最后。
        # 这样即使 flags 里混进了 -o，它也会被后面这个覆盖掉，
        # 产物写到哪儿是由命令的**结构**保证的，不只是靠 _checked_flags 那道校验。
        parts = [compiler]
        if flags:
            parts.append(flags)
        parts += [source_name, "-o", target_name]
        return " ".join(parts)

    def _env(self, compiler: str) -> dict[str, str] | None:
        """跑这条命令要给子进程的环境。

        只有 cl 需要：它不在 PATH 上，得先让 vcvars64.bat 把
        PATH / INCLUDE / LIB 铺好。别的编译器继承当前环境就行。
        """
        if compiler != MSVC:
            return None
        return vsenv.vcvars_env()

    def _checked_flags(self, raw: str) -> str:
        """校验编译参数，返回去掉多余空白之后的版本。

        校验的是**模型写的那一份**（g++ 写法），翻译成 cl 写法在 _to_msvc()。
        顺序不能反：先按原文挡住危险字符，再谈翻不翻得动。
        """
        flags = " ".join(raw.split())
        if not _SAFE_FLAGS.match(flags):
            raise ToolError(
                f"编译参数里不能有引号、分号、重定向这类字符"
                f"（它们会被 shell 当成别的意思），收到：{raw!r}"
            )

        for token in flags.split():
            if _is_output_flag(token):
                raise ToolError(
                    f"输出到哪儿由 output 参数决定，编译参数里不能再出现 {token!r}"
                )
        return flags

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _compiler(self, requested: str | None) -> str:
        """决定这次用哪个编译器。

        优先级和 timeout 是同一套：这次调用指定的 > 装配工具时指定的 > 自动找。
        越具体越优先。
        """
        if requested is not None:
            # 只有模型传进来的那个才需要校验：装配时的值是代码写的，
            # 测试还要靠它塞一个假编译器进去。
            # 编译器名字是拼进命令的，所以只能是清单里那几个——
            # 这样模型就没办法拿它塞路径或者夹带别的命令。
            if requested not in COMPILERS:
                raise ToolError(
                    f"不认识的编译器 {requested!r}，只能用：{'、'.join(COMPILERS)}"
                )
            if not _which(requested):
                # 不查的话，模型会拿到 shell 那句「不是内部或外部命令」，
                # 看得懂但没用：它不知道是名字写错了还是根本没装、装了哪个
                found = find_compiler()
                hint = f"装了的是 {found}" if found else "一个都没装"
                raise ToolError(
                    f"指定了 {requested}，但这台机器上没有它（{hint}）"
                )
            return requested

        if self.compiler:
            return self.compiler

        found = find_compiler()
        if found is None:
            raise ToolError(
                "找不到 C++ 编译器：g++、clang++、cl 一个都没找到。"
                "装一个，或者用 run_command 自己指定编译器。"
            )
        return found


# ---------------------------------------------------------------------------
# g++ 写法 -> cl 写法
# ---------------------------------------------------------------------------

# 能一一对应的那几个。只翻常用的，翻不了的一律拒绝——
# cl 碰到不认识的参数只打一句 warning 就跳过，悄悄按错的编比报错糟得多。
_MSVC_FLAGS = {
    # cl 的优化档位：/Od 不优化、/O1 优先小、/O2 优先快，没有 /O3
    "-O0": "/Od",
    "-O1": "/O1",
    "-O2": "/O2",
    "-O3": "/O2",
    # -Os 求小，正好对上 /O1
    "-Os": "/O1",
    "-g": "/Zi",
    # /Wall 在 MSVC 里是「把所有警告全打开」，噪音大到没法用；
    # 通常拿 /W4 对应 g++ 的 -Wall -Wextra
    "-Wall": "/W4",
    "-Wextra": "/W4",
    "-w": "/w",
}

# -std= 的值：g++ 的叫法和 cl 的叫法不一样的那几个
_MSVC_STD = {
    # cl 没有 C++11 模式，c++14 是它的最低档，而且向下兼容 C++11
    "c++11": "c++14",
    "c++0x": "c++14",
    "c++1y": "c++14",
    "c++1z": "c++17",
    "c++2a": "c++20",
    "c++2b": "c++latest",
}
_MSVC_STD_OK = frozenset({"c++14", "c++17", "c++20", "c++latest"})

# cl 自己带上就好，别让模型操心：
# - /nologo 去掉那段版本横幅；
# - /EHsc 打开异常（g++ 默认就开，cl 不写会差一截）；
# - /utf-8 让 cl 按 UTF-8 读源码。这条是实测逼出来的：cl 默认按系统代码页
#   （中文 Windows 上是 936/GBK）读源码，UTF-8 的中文源码会被读歪，
#   报出来的是「C4819 该文件包含不能在当前代码页中表示的字符」加一句
#   「C2001 常量中有换行符」——完全看不出真正的原因是编码。
#   加上它，cl 和 g++ 对同一份源码的理解才一致。
# - /diagnostics:column 让 cl 报列号。默认它只给到行（`a.cpp(3): error`），
#   带上之后是 `a.cpp(3,3): error`，和 g++ 的 `a.cpp:3:3:` 对得上——
#   结构化诊断（tools/diagnostics.py）才拿得到 column。
#   老版本 MSVC 不认这个开关时只会打一句 D9002 警告然后忽略，不会编不过。
_MSVC_ALWAYS = ("/nologo", "/EHsc", "/utf-8", "/diagnostics:column")


def _is_output_flag(token: str) -> bool:
    """这个参数是不是在指定产物路径。"""
    return token.startswith(_OUTPUT_FLAGS) or token.lower().startswith(
        _MSVC_OUTPUT_FLAGS
    )


def _to_msvc(flags: str) -> str:
    """把 g++ 写法的参数翻成 cl 的写法。翻不了就拒绝，不猜。"""
    translated: list[str] = []
    for token in flags.split():
        if token in _MSVC_FLAGS:
            translated.append(_MSVC_FLAGS[token])
        elif token.startswith("-std="):
            translated.append(_msvc_std(token))
        elif token[:2] in ("-D", "-I"):
            # -DDEBUG=1 / -Iinclude -> /DDEBUG=1 /Iinclude
            translated.append("/" + token[1:])
        else:
            raise ToolError(
                f"{token!r} 不知道怎么翻成 cl 的写法。cl 这条路只认常见的几个："
                f"-std= -O0/-O1/-O2/-O3/-Os -g -Wall -Wextra -w -D -I。"
                f"别的参数请用 run_command 自己拼 cl 命令。"
            )
    return " ".join(translated)


def _msvc_std(token: str) -> str:
    """-std=xxx -> /std:xxx。"""
    value = token[len("-std=") :]
    value = _MSVC_STD.get(value, value)
    if value not in _MSVC_STD_OK:
        raise ToolError(
            f"cl 没有 -std={token[len('-std=') :]} 这个档，"
            f"它只支持：{'、'.join(sorted(_MSVC_STD_OK))}"
        )
    return f"/std:{value}"
