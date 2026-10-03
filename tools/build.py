"""工具：调用 CMake / Make / MSBuild 构建整个工程。

前面几个执行类工具都只跑一条命令，这个是第一个要跑**一串**的：
cmake 得先配置再编译，前一步没成功就没必要跑下一步。

执行还是全交给 RunCommandTool（subprocess、超时、输出解码、截断都在那边）。
这里多做的一件事是：要拿到每一步的**退出码**才能决定往不往下走，
所以调的是 run_with_code() 而不是 execute()。

msbuild 是 MSVC 那套的构建入口（和 cl.exe 一样藏在 Visual Studio 里，
见 tools/vsenv.py）。没有引入 Ninja、Bazel 这类新构建系统——
msbuild 是 VS 自带的，装了 VS 就有。

解析诊断还是用 tools/diagnostics.py，这个文件里不另写一套。构建比单文件编译
更值得解一遍：一个失败能牵出一串连带错误，而 msbuild 还会在结尾把每条诊断
再汇总说一遍（所以这里要去重）、并且报的是绝对路径（所以要换成本地的）。
"""

import shutil
from dataclasses import replace
from pathlib import Path

from tools import diagnostics, vsenv
from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_shell_path
from tools.run_command import RunCommandTool

SYSTEMS = ("cmake", "make", "msbuild")

# 这一条走 MSVC 那套：命令藏在 VS 里，得先铺 vcvars 环境
MSVC_SYSTEM = "msbuild"

# 找工程文件时先找解决方案，再退而求其次找单个工程
_PROJECT_SUFFIXES = (".sln", ".vcxproj")

DEFAULT_BUILD_DIR = "build"

# 构建默认等多久。比跑测试还宽松：全量构建动辄几分钟，
# 而且构建和跑测试不一样，半路掐掉留下的半成品还得自己清理。
DEFAULT_BUILD_TIMEOUT = 300.0


class BuildTool(Tool):
    """用 CMake / Make 构建工程。"""

    name = "build"
    description = (
        "用 CMake、Make 或 MSBuild 构建整个工程。"
        "cmake 跑两步：先 cmake -S <源码目录> -B <构建目录> 配置，"
        "再 cmake --build <构建目录> 编译；"
        "make 跑一步（源码目录不是项目根目录时是 make -C <源码目录>）；"
        "msbuild 跑一步，构建源码目录里那个 .sln（没有就找 .vcxproj），"
        "它在 PATH 上找不到也没关系，工具会自己去找 Visual Studio 并铺好环境。"
        "前一步没成功就不跑下一步，省得后面那堆连带错误把真正的原因淹了。"
        "构建失败（退出码非 0）也会正常返回结果，报错就在输出里，不算调用失败。"
        "结果末尾还会另附一段解析好的诊断，一行一条，写成"
        "「文件:行:列: 级别: 消息」；那里的文件路径和行号可以直接拿去用 read_file 核对。"
    )
    parameters = {
        "system": {
            "type": "string",
            "description": f"用哪个构建系统，只能是 {' 或 '.join(SYSTEMS)}",
            "required": True,
        },
        "source_root": {
            "type": "string",
            "description": "源码目录，相对项目根目录；省略就是项目根目录",
            "required": False,
        },
        "build_dir": {
            "type": "string",
            "description": (
                f"构建目录，相对项目根目录，默认 {DEFAULT_BUILD_DIR}；"
                "只对 cmake 有效，make 传了会被拒绝"
            ),
            "required": False,
        },
        "timeout": {
            "type": "number",
            "description": f"每一步最多等多少秒，默认 {DEFAULT_BUILD_TIMEOUT:g}",
            "required": False,
        },
    }

    def __init__(
        self, root: Path | None = None, timeout: float = DEFAULT_BUILD_TIMEOUT
    ) -> None:
        self.root = check_root(root)
        self.command_tool = RunCommandTool(root=self.root, timeout=timeout)

    def build(
        self,
        system: str,
        source_root: str = ".",
        build_dir: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """按顺序跑完这个构建系统的所有步骤，把每一步的结果拼起来返回。"""
        commands = self._plan(system, source_root, build_dir)
        # 整个构建系统一套环境就够了，每一步不用各问一遍
        env = self._env(system)
        total = len(commands)
        blocks: list[str] = []

        for index, command in enumerate(commands, 1):
            code, text = self.command_tool.run_with_code(
                command, timeout=timeout, env=env
            )
            # 只有一步时不用标「第 1/1 步」，那只是噪音
            blocks.append(text if total == 1 else f"[第 {index}/{total} 步]\n{text}")

            if code != 0:  # 超时的退出码是 None，也一样算没成功
                if index < total:
                    blocks.append(
                        f"…（第 {index} 步没成功，剩下 {total - index} 步没有执行）"
                    )
                break

        return self._with_diagnostics("\n\n".join(blocks))

    def execute(
        self,
        system: str,
        source_root: str = ".",
        build_dir: str | None = None,
        timeout: float | None = None,
    ) -> str:
        return self.build(system, source_root, build_dir, timeout)

    def _with_diagnostics(self, result: str) -> str:
        """在原始输出末尾附一段解析好的诊断。

        原始输出一个字符都不动——构建工具报的原文模型仍然看得到，这里只是
        在最后另起一段把「哪个文件、第几行、什么问题」摆出来。
        构建成功但带 warning 时也附：warning 一样是要改代码才能消掉的。

        解的是**整段**输出，不按步骤分开解。构建工具把报错打在哪个流上
        并不统一（cl 打在标准输出，gcc 打在标准错误），而 run_command 两路
        都收下来了、拼在同一段文本里，所以这里不用挑，整段一起解。
        """
        report = self._report(result)
        if not report.has_problems:
            # 一条都没认出来，或者只有 note——后者的意思是「它是在那儿声明的」，
            # 脱离它跟的那条 error 单独看没有意义
            return result
        return f"{result}\n\n{diagnostics.render(report)}"

    def _report(self, result: str) -> diagnostics.Report:
        """把整段构建输出解成诊断：路径换成本地的，说重了的并掉。"""
        found = diagnostics.parse(result)
        # dict 去重且保持原顺序。msbuild 会把每条诊断说两遍——先打在构建
        # 日志里，再在结尾的「生成失败」汇总里重复一遍。不去重的话模型会以为
        # 真有两处一样的错，条数也是虚的。比的是整条诊断，所以同一个文件里
        # 两条不同的错不会并到一起。
        items = dict.fromkeys(
            replace(item, file=self._readable_path(item.file))
            for item in found.diagnostics
        )
        return diagnostics.Report(tuple(items), found.other)

    def _readable_path(self, file: str) -> str:
        """能对上项目根目录的路径换成相对路径，对不上就原样留着。

        msbuild 报的是绝对路径（命令行里写的是相对的，它自己展开成绝对的），
        而 read_file 那条路只认相对 root 的路径——`C:/...` 里那个冒号连
        路径白名单都过不了。所以能换的换掉，模型拿到手就能直接用。

        换不掉的（比如 MSVC 自带的头文件）留着绝对路径：那本来就超出 root，
        模型读不到，但至少知道问题出在哪儿。
        """
        try:
            return Path(file).relative_to(self.root).as_posix()
        except ValueError:
            return file

    def _plan(
        self, system: str, source_root: str, build_dir: str | None
    ) -> list[str]:
        """把参数变成要依次执行的命令。

        命令是拼出来的，所以每个会进命令的字符串都先过一遍校验。
        """
        if system not in SYSTEMS:
            raise ToolError(
                f"不认识的构建系统 {system!r}，只能用：{'、'.join(SYSTEMS)}"
            )
        if not self._available(system):
            # 不查的话模型拿到的是 shell 那句「不是内部或外部命令」，
            # 看不出是名字写错了还是这台机器根本没装
            raise ToolError(
                f"这台机器上没有 {system}。装一个，"
                f"或者用 run_command 自己拼构建命令。"
            )

        source = resolve_shell_path(source_root, self.root, "源码目录")
        if not source.is_dir():
            raise ToolError(f"源码目录不存在：{source_root!r}")

        if system == "make":
            if build_dir is not None:
                # 不能默默忽略：模型会以为产物就在那个目录里
                raise ToolError(
                    "make 不用 build_dir——编到哪儿由 Makefile 自己决定，"
                    "去掉这个参数再试"
                )
            if source_root == ".":
                return ["make"]
            return [f"make -C {self._relative(source)}"]

        if system == MSVC_SYSTEM:
            if build_dir is not None:
                # 和 make 一个道理：.vcxproj 里写着产物去哪儿，改不了
                raise ToolError(
                    "msbuild 不用 build_dir——编到哪儿由工程文件自己决定，"
                    "去掉这个参数再试"
                )
            return [f"msbuild {self._relative(self._project(source))}"]

        target = resolve_shell_path(
            build_dir or DEFAULT_BUILD_DIR, self.root, "构建目录"
        )
        return [
            f"cmake -S {self._relative(source)} -B {self._relative(target)}",
            f"cmake --build {self._relative(target)}",
        ]

    def _available(self, system: str) -> bool:
        """这台机器上有没有这个构建系统。

        msbuild 和 cl.exe 一样藏在 VS 里，装了也不在 PATH 上，得多问一句。
        """
        if shutil.which(system):
            return True
        if system == MSVC_SYSTEM:
            return vsenv.find_program(system) is not None
        return False

    def _env(self, system: str) -> dict[str, str] | None:
        """跑这个构建系统要给子进程的环境。只有 msbuild 需要 VS 那一套。"""
        if system != MSVC_SYSTEM:
            return None
        return vsenv.vcvars_env()

    def _project(self, source: Path) -> Path:
        """在源码目录里找出要构建的工程文件。

        只有一个就肯定是它。一个都没有、或者有好几个，就得让模型自己说清楚——
        猜错那个比直接报错更浪费时间。
        找出来的名字也要过一遍路径校验：它是从磁盘上扫来的，
        不是模型给的，但一样会拼进命令里。
        """
        for suffix in _PROJECT_SUFFIXES:
            found = sorted(source.glob(f"*{suffix}"))
            if len(found) == 1:
                return resolve_shell_path(str(found[0]), self.root, "工程文件")
            if len(found) > 1:
                names = "、".join(item.name for item in found)
                raise ToolError(
                    f"{source.name}/ 下有多个 {suffix} 文件（{names}），"
                    f"msbuild 不知道该构建哪个。"
                    f"用 run_command 自己指定要构建的那一个。"
                )
        raise ToolError(
            f"{source.name}/ 下没有 {' 或 '.join(_PROJECT_SUFFIXES)} 工程文件，"
            f"msbuild 没有东西可构建"
        )

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
