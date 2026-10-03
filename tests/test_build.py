"""BuildTool 测试：CMake 构建、Make 构建、MSBuild 构建、构建失败、路径安全、Agent 调用。

这台机器上 cmake、make、msbuild 都不在 PATH 上，所以往 PATH 最前面塞几个
.bat 顶替它们。这样测的是**真的代码路径**：命令由工具拼出来、由 shell 去找
cmake，连 shutil.which 那道检查都真的走了一遍——不是把命令这件事绕过去测。

msbuild 比另两个多一层：它藏在 Visual Studio 里，命令要带着 vcvars 那套
环境跑。那部分由 VsEnvTest 假装，这里只钉「环境确实传下去了」。

末尾那段诊断（tools/diagnostics.py）由 *DiagnosticsTest 两组看着：假的那组
钉形状，BuildToolRealDiagnosticsTest 拿真 msbuild 跑一遍，把报出来的行号列号
戳回源码上核对。msbuild 的报错在标准输出、gcc 的在标准错误，两条都测了。

真的 msbuild 由 BuildToolRealTest 跑，机器上装了 VS 才跑，没有就跳过。

全部离线，不联网。
"""

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent
from config import PROJECT_ROOT
from tools import (
    BuildTool,
    ReadFileTool,
    RunCommandTool,
    Tool,
    ToolError,
    ToolRegistry,
    vsenv,
)
from tools.build import DEFAULT_BUILD_DIR, DEFAULT_BUILD_TIMEOUT, SYSTEMS

# 假 cmake / 假 make / 假 msbuild。第一个参数是「我是谁」（.bat 传进来的），
# 后面才是命令行。msbuild 那条路还会报一个环境记号，用来验 vcvars 有没有传下去。
FAKE_BUILD = '''\
"""假构建工具：够用的那部分行为。"""
import os
import sys
from pathlib import Path

tool, args = sys.argv[1], sys.argv[2:]
print(f"fake {tool} " + " ".join(args))

if tool == "msbuild":
    print("marker: " + os.environ.get("FAKE_VS_MARKER", "（没传进来）"))

if tool == "make":
    step = "make"
elif "--build" in args:
    step = "build"
elif tool == "msbuild":
    step = "msbuild"
else:
    step = "configure"

# 让假构建工具按剧本吐一段输出：{step}.OUT 走标准输出，{step}.ERR 走标准错误。
# 不同构建系统把报错打在哪个流上不一样（msbuild 打标准输出、gcc 打标准错误），
# 解析器得两条都认，所以两条都得能测。
for name, stream in ((f"{step}.OUT", sys.stdout), (f"{step}.ERR", sys.stderr)):
    extra = Path(name)
    if extra.exists():
        print(extra.read_text(encoding="utf-8"), end="", file=stream)

if Path(f"{step}.FAIL").exists():
    print(f"fake {tool}: {step} 失败：故意放的 {step}.FAIL", file=sys.stderr)
    sys.exit(1)

if step == "configure":
    target = Path(args[args.index("-B") + 1])
    target.mkdir(parents=True, exist_ok=True)
    (target / "CMakeCache.txt").write_text("假的缓存文件", encoding="utf-8")
else:
    print(f"fake {tool}: {step} 完成")
'''

BAT = '@echo off\n"{python}" "{script}" {name} %*\nexit /b %ERRORLEVEL%\n'


class BuildToolTestBase(unittest.TestCase):
    """搭一个假项目，再把两个假构建工具挂到 PATH 最前面。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name).resolve()

        self.root = base / "project"
        self.root.mkdir()

        script = base / "fake_build.py"
        script.write_text(FAKE_BUILD, encoding="utf-8")
        for name in ("cmake", "make", "msbuild"):
            (base / f"{name}.bat").write_text(
                BAT.format(python=sys.executable, script=script, name=name),
                encoding="utf-8",
            )

        # 用真的 PATH：shell 真的去找了一圈才找到它们，which 也是
        env = patch.dict(
            os.environ, {"PATH": f"{base}{os.pathsep}{os.environ['PATH']}"}
        )
        env.start()
        self.addCleanup(env.stop)

    def make_tool(self, **kwargs) -> BuildTool:
        return BuildTool(root=self.root, **kwargs)

    def build(self, **kwargs) -> str:
        return self.make_tool().run(**kwargs)

    def fail(self, step: str) -> None:
        """让假构建工具在某一步失败。文件放在 root 里，也就是命令的工作目录。"""
        (self.root / f"{step}.FAIL").write_text("", encoding="utf-8")

    def say(self, step: str, out: str = "", err: str = "") -> None:
        """让假构建工具在这一步多吐这么两段输出。"""
        for suffix, text in (("OUT", out), ("ERR", err)):
            if text:
                (self.root / f"{step}.{suffix}").write_text(
                    text, encoding="utf-8"
                )

    def block(self, result: str) -> str:
        """把末尾那段诊断摘出来。

        原始输出里本来就有那些话（那是构建工具自己报的），要和诊断段分开数，
        不然「只列了一遍」这种断言根本分不清是在数哪一段。
        """
        return result.partition("\n\n诊断（")[2]

    def assert_diagnosed(self, result: str, start: str) -> None:
        """诊断段里有这么一行。行首那两格缩进是段落里的，钉住它能保证
        匹配的不是原始输出里那一份。"""
        self.assertIn(f"\n  {start}", "\n" + self.block(result))

    def mkdir(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path


class BuildToolCMakeTest(BuildToolTestBase):
    def setUp(self):
        super().setUp()
        self.result = self.build(system="cmake")

    def test_it_runs_two_steps(self):
        self.assertIn("[第 1/2 步]", self.result)
        self.assertIn("[第 2/2 步]", self.result)

    def test_the_configure_command_is_the_documented_one(self):
        self.assertIn("命令：cmake -S . -B build", self.result)

    def test_the_build_command_is_the_documented_one(self):
        self.assertIn("命令：cmake --build build", self.result)

    def test_both_steps_succeed(self):
        self.assertEqual(self.result.count("退出码：0"), 2)

    def test_configure_runs_before_build(self):
        self.assertLess(
            self.result.index("cmake -S ."), self.result.index("cmake --build")
        )

    def test_the_build_step_only_works_because_configure_ran(self):
        # 假 cmake 只在 configure 那一步建出构建目录并写下缓存文件
        self.assertTrue((self.root / "build" / "CMakeCache.txt").is_file())

    def test_every_step_has_all_four_pieces(self):
        for piece in ("命令：", "退出码：", "标准输出：", "标准错误："):
            with self.subTest(piece=piece):
                self.assertEqual(self.result.count(piece), 2)

    def test_a_custom_build_dir_is_used(self):
        result = self.build(system="cmake", build_dir="out")

        self.assertIn("cmake -S . -B out", result)
        self.assertIn("cmake --build out", result)
        self.assertTrue((self.root / "out").is_dir())

    def test_a_custom_source_root_is_used(self):
        self.mkdir("src")

        result = self.build(system="cmake", source_root="src")

        self.assertIn("cmake -S src -B build", result)

    def test_paths_stay_relative_to_the_root(self):
        # 源码目录不是根目录时，构建目录的相对基准**仍然是根目录**，
        # 不是源码目录——不然 cmake -B src/out 会被理解成 src/src/out
        self.mkdir("src")

        result = self.build(system="cmake", source_root="src", build_dir="src/out")

        self.assertIn("cmake -S src -B src/out", result)
        self.assertTrue((self.root / "src" / "out").is_dir())

    def test_the_root_itself_is_allowed_as_source_root(self):
        # 默认值就是 "."，它归一化之后过不了路径白名单，得单独放行
        result = self.build(system="cmake", source_root=".")

        self.assertIn("cmake -S . -B build", result)


class BuildToolCMakeFailureTest(BuildToolTestBase):
    def test_a_failed_configure_stops_the_build(self):
        self.fail("configure")

        result = self.build(system="cmake")

        self.assertIn("[第 1/2 步]", result)
        self.assertNotIn("[第 2/2 步]", result)

    def test_it_says_the_rest_was_skipped(self):
        self.fail("configure")

        self.assertIn("剩下 1 步没有执行", self.build(system="cmake"))

    def test_a_failed_configure_reports_a_non_zero_exit_code(self):
        self.fail("configure")

        self.assertIn("退出码：1", self.build(system="cmake"))

    def test_a_failed_configure_keeps_the_error_text(self):
        self.fail("configure")

        self.assertIn(
            "configure 失败：故意放的 configure.FAIL", self.build(system="cmake")
        )

    def test_a_failed_configure_does_not_raise(self):
        self.fail("configure")

        self.assertIsInstance(self.build(system="cmake"), str)

    def test_a_failed_build_runs_after_a_good_configure(self):
        self.fail("build")

        result = self.build(system="cmake")

        self.assertIn("[第 1/2 步]", result)
        self.assertIn("[第 2/2 步]", result)
        self.assertIn("build 失败：故意放的 build.FAIL", result)

    def test_nothing_is_said_about_skipping_when_the_last_step_fails(self):
        self.fail("build")

        self.assertNotIn("没有执行", self.build(system="cmake"))


class BuildToolMakeTest(BuildToolTestBase):
    def setUp(self):
        super().setUp()
        self.result = self.build(system="make")

    def test_the_command_is_plain_make(self):
        self.assertIn("命令：make", self.result)

    def test_one_step_needs_no_step_header(self):
        self.assertNotIn("第 1/", self.result)

    def test_it_succeeds(self):
        self.assertIn("退出码：0", self.result)

    def test_a_source_root_becomes_dash_c(self):
        self.mkdir("src")

        result = self.build(system="make", source_root="src")

        self.assertIn("命令：make -C src", result)

    def test_build_dir_is_rejected(self):
        # 拒掉而不是默默忽略：不然模型会以为产物就在那个目录里
        with self.assertRaises(ToolError) as caught:
            self.build(system="make", build_dir="build")

        self.assertIn("make 不用 build_dir", str(caught.exception))

    def test_a_failed_make_is_reported(self):
        self.fail("make")

        result = self.build(system="make")

        self.assertIn("退出码：1", result)
        self.assertIn("make 失败：故意放的 make.FAIL", result)

    def test_a_failed_make_does_not_raise(self):
        self.fail("make")

        self.assertIsInstance(self.build(system="make"), str)


class BuildToolMSBuildTestBase(BuildToolTestBase):
    """msbuild 那条路多一层：命令藏在 Visual Studio 里，得带着 vcvars 的环境跑。

    真实的 vcvars_env() 里没有上面那个临时目录，假 msbuild 就不在它的 PATH 上，
    所以这里换成当前环境——形状一样（一份变量字典），内容换成能跑通的。
    """

    def setUp(self):
        super().setUp()
        vs = patch(
            "tools.build.vsenv.vcvars_env", side_effect=lambda: dict(os.environ)
        )
        vs.start()
        self.addCleanup(vs.stop)

    def project(self, name: str, text: str = "假的工程文件") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def said_by_msbuild(self, name: str, line: int, column: int, rest: str) -> str:
        """msbuild 报出来的一行，照它真实的样子：绝对路径、反斜杠，
        末尾还挂着「这条是哪个工程报的」。"""
        return f"{self.root}\\{name}({line},{column}): {rest} [{self.root}\\demo.vcxproj]\n"


class BuildToolMSBuildTest(BuildToolMSBuildTestBase):
    def setUp(self):
        super().setUp()
        self.project("demo.sln")
        self.result = self.build(system="msbuild")

    def test_one_step_needs_no_step_header(self):
        # msbuild 一步就够了，和 make 一样不用标「第 1/1 步」
        self.assertNotIn("第 1/", self.result)

    def test_the_command_names_the_solution(self):
        self.assertIn("命令：msbuild demo.sln", self.result)

    def test_it_succeeds(self):
        self.assertIn("退出码：0", self.result)

    def test_the_solution_is_found_in_a_subdirectory(self):
        fresh = tempfile.TemporaryDirectory()
        self.addCleanup(fresh.cleanup)
        base = Path(fresh.name).resolve()
        (base / "src").mkdir()
        (base / "src" / "app.sln").write_text("假的", encoding="utf-8")

        result = BuildTool(root=base).build(system="msbuild", source_root="src")

        self.assertIn("命令：msbuild src/app.sln", result)

    def test_the_environment_is_passed_to_the_build(self):
        # 环境没传下去的话，真的 msbuild 会找不到工具集。
        # 假 msbuild 把记号打出来，看见记号就说明送到了
        marker = patch.dict(os.environ, {"FAKE_VS_MARKER": "送到了"})
        marker.start()
        self.addCleanup(marker.stop)

        result = self.build(system="msbuild")

        self.assertIn("marker: 送到了", result)

    def test_a_failed_build_is_reported(self):
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assertIn("退出码：1", result)
        self.assertIn("msbuild 失败：故意放的 msbuild.FAIL", result)

    def test_a_failed_build_does_not_raise(self):
        self.fail("msbuild")

        self.assertIsInstance(self.build(system="msbuild"), str)


class BuildToolMSBuildProjectTest(BuildToolMSBuildTestBase):
    """工程文件是扫出来的，不是一个固定的名字——扫错了比报错浪费得多。"""

    def test_a_single_project_file_is_used(self):
        self.project("solo.vcxproj")

        result = self.build(system="msbuild")

        self.assertIn("命令：msbuild solo.vcxproj", result)

    def test_the_solution_wins_over_the_project(self):
        # 一个目录里既有 .sln 又有 .vcxproj 是常态：.sln 才是入口
        self.project("demo.sln")
        self.project("demo.vcxproj")

        result = self.build(system="msbuild")

        self.assertIn("命令：msbuild demo.sln", result)

    def test_several_solutions_are_refused(self):
        # 猜错那一个要跑完一整个失败的构建才知道，不如当场问清楚
        self.project("a.sln")
        self.project("b.sln")

        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild")

        self.assertIn("多个 .sln", str(caught.exception))
        self.assertIn("a.sln", str(caught.exception))

    def test_no_project_file_is_refused(self):
        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild")

        self.assertIn("没有 .sln 或 .vcxproj", str(caught.exception))

    def test_build_dir_is_rejected(self):
        # 和 make 一个道理：产物去哪儿由工程文件决定，工具改不了
        self.project("demo.sln")

        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild", build_dir="out")

        self.assertIn("msbuild 不用 build_dir", str(caught.exception))

    def test_a_missing_source_root_is_refused(self):
        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild", source_root="nope")

        self.assertIn("源码目录不存在", str(caught.exception))

    def test_nothing_runs_when_a_project_name_is_refused(self):
        self.project("demo.sln")
        self.project("other.sln")

        with self.assertRaises(ToolError):
            self.build(system="msbuild")

    def test_a_project_name_with_shell_characters_is_refused(self):
        # 工程文件的名字是从磁盘上扫来的，一样要过字符白名单：
        # 一个带 & 的文件名就能把后面的东西变成另一条命令
        self.project("a & whoami.sln")

        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild")

        self.assertIn("只能用字母", str(caught.exception))

    def test_source_root_still_cannot_escape_the_root(self):
        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild", source_root="../outside")

        self.assertIn("超出允许范围", str(caught.exception))


class BuildToolMSBuildAvailableTest(BuildToolMSBuildTestBase):
    def test_it_is_available_when_it_is_on_the_path(self):
        self.project("demo.sln")

        self.assertIn("退出码：0", self.build(system="msbuild"))

    def test_it_is_also_available_through_visual_studio(self):
        # 真实的场景就是这样：msbuild 不在 PATH 上，装了 VS 才有。
        # 这里把 PATH 断掉，只留 VS 那条路
        self.project("demo.sln")
        found = patch(
            "tools.build.vsenv.find_program",
            side_effect=lambda name: rf"C:\VS\{name}.exe",
        )
        found.start()
        self.addCleanup(found.stop)

        which = patch("tools.build.shutil.which", return_value=None)
        which.start()
        self.addCleanup(which.stop)

        tool = BuildTool(root=self.root)

        self.assertTrue(tool._available("msbuild"))

    def test_it_says_so_when_there_is_no_visual_studio(self):
        self.project("demo.sln")
        found = patch("tools.build.vsenv.find_program", return_value=None)
        found.start()
        self.addCleanup(found.stop)
        which = patch("tools.build.shutil.which", return_value=None)
        which.start()
        self.addCleanup(which.stop)

        with self.assertRaises(ToolError) as caught:
            self.build(system="msbuild")

        self.assertIn("这台机器上没有 msbuild", str(caught.exception))

    def test_cmake_and_make_do_not_ask_about_visual_studio(self):
        # 它们和 VS 没关系，不该为它们去起一次 vcvars（那一下要两秒多）
        for system in ("cmake", "make"):
            with self.subTest(system=system):
                with patch("tools.build.vsenv.vcvars_env") as asked:
                    self.build(system=system)

                asked.assert_not_called()


class BuildToolDiagnosticsTest(BuildToolTestBase):
    """cmake / make 那条路：gcc、clang 的报错打在标准错误上。

    和 msbuild 那条路（打在标准输出上）解出来的形状要一模一样。
    """

    GCC_ERROR = "src/a.cpp:12:5: error: 'x' was not declared in this scope\n"
    GCC_WARNING = (
        "src/a.cpp:3:5: warning: unused variable 'n' [-Wunused-variable]\n"
    )

    def test_an_error_on_stderr_is_parsed(self):
        self.say("make", err=self.GCC_ERROR)
        self.fail("make")

        result = self.build(system="make")

        self.assertIn("退出码：1", result)
        self.assert_diagnosed(result, "src/a.cpp:12:5: error:")

    def test_a_warning_from_a_successful_build_is_reported(self):
        # 构建过了但带 warning 也要附：warning 一样得改代码才能消掉
        self.say("build", out=self.GCC_WARNING)

        result = self.build(system="cmake")

        self.assertIn("退出码：0", result)
        self.assert_diagnosed(
            result, "src/a.cpp:3:5: warning -Wunused-variable:"
        )

    def test_a_clean_build_gets_no_diagnostics_block(self):
        # 没东西可报的时候一个字都不加——原来的输出就是原样
        for system in ("cmake", "make"):
            with self.subTest(system=system):
                self.assertNotIn("诊断（", self.build(system=system))

    def test_the_block_comes_after_the_step_summary(self):
        # 构建本身的输出一个字不动，诊断另起一段挂在最后
        self.fail("build")
        self.say("build", err=self.GCC_ERROR)

        result = self.build(system="cmake")

        self.assertIn("[第 2/2 步]", result)
        self.assertGreater(result.rindex("诊断（"), result.rindex("退出码：1"))

    def test_an_unparsable_failure_returns_the_raw_output_alone(self):
        # 构建失败了，但一段认得的诊断都没有：原文照旧，不硬凑一段空的
        self.fail("make")

        result = self.build(system="make")

        self.assertIn("make 失败：故意放的 make.FAIL", result)
        self.assertNotIn("诊断（", result)

    def test_the_original_output_is_left_alone(self):
        self.say("make", err=self.GCC_ERROR)
        self.fail("make")

        result = self.build(system="make")

        # 四件套一件不少，报错原文也还在，诊断是**追加**上去的
        for piece in ("命令：", "退出码：", "标准输出：", "标准错误："):
            with self.subTest(piece=piece):
                self.assertIn(piece, result)
        self.assertIn(self.GCC_ERROR.strip(), result)
        self.assertTrue(result.rstrip().endswith("error: 'x' was not declared in this scope"))


class BuildToolMSBuildDiagnosticsTest(BuildToolMSBuildTestBase):
    """msbuild 的输出解成结构化诊断。

    下面那些输出都是照着这台机器上真跑出来的样子写的：绝对路径、末尾挂着
    工程文件、每条诊断还会在结尾的「生成失败」汇总里再说一遍。
    """

    def setUp(self):
        super().setUp()
        self.project("demo.sln")

    def test_a_warning_from_a_successful_build_is_reported(self):
        self.say(
            "msbuild",
            out=self.said_by_msbuild(
                "a.cpp", 2, 3, "warning C4189: “x”: 局部变量已初始化但不使用"
            ),
        )

        result = self.build(system="msbuild")

        self.assertIn("退出码：0", result)
        self.assertIn("诊断（共 1 条：1 个 warning）", result)
        self.assert_diagnosed(result, "a.cpp:2:3: warning C4189:")

    def test_every_error_from_every_file_is_kept(self):
        # msbuild 会把整个工程编一遍再一起报，一个失败牵出一串——
        # 全留着，模型才知道一共要改几处
        self.say(
            "msbuild",
            out=(
                self.said_by_msbuild(
                    "a.cpp", 2, 3, "error C2065: “undeclared_a”: 未声明的标识符"
                )
                + self.said_by_msbuild(
                    "sub\\b.cpp", 7, 11, "error C2059: 语法错误:“;”"
                )
            ),
        )
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assertIn("诊断（共 2 条：2 个 error）", result)
        self.assert_diagnosed(result, "a.cpp:2:3: error C2065:")
        self.assert_diagnosed(result, "sub/b.cpp:7:11: error C2059:")

    def test_a_diagnostic_msbuild_repeats_is_only_listed_once(self):
        # msbuild 先把它打在构建日志里，再在结尾的「生成失败」汇总里重复一遍。
        # 不去重的话模型会以为真有两处一样的错，条数也是虚的
        line = self.said_by_msbuild(
            "a.cpp", 2, 3, "error C2065: “undeclared_a”: 未声明的标识符"
        )
        self.say("msbuild", out=line + line)
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assertIn("诊断（共 1 条：1 个 error）", result)
        self.assertEqual(self.block(result).count("a.cpp:2:3"), 1)

    def test_two_different_errors_on_one_line_are_both_kept(self):
        # 去重比的是整条诊断，不是位置：同一行上两条不同的错不能并成一条
        self.say(
            "msbuild",
            out=(
                self.said_by_msbuild("a.cpp", 2, 3, "error C2065: 甲")
                + self.said_by_msbuild("a.cpp", 2, 3, "error C2143: 乙")
            ),
        )
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assertIn("诊断（共 2 条：2 个 error）", result)

    def test_an_absolute_path_is_rewritten_relative_to_the_root(self):
        # msbuild 报的是绝对路径，而 read_file 只认相对 root 的路径
        # （`C:/...` 里那个冒号连路径白名单都过不了），所以这里要换掉
        self.say(
            "msbuild",
            out=self.said_by_msbuild(
                "a.cpp", 2, 3, "error C2065: “undeclared_a”: 未声明的标识符"
            ),
        )
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assert_diagnosed(result, "a.cpp:2:3: error C2065:")

    def test_a_path_outside_the_root_is_left_as_it_is(self):
        # MSVC 自带的头文件在项目外面，换不成相对的。留着绝对路径，
        # 至少知道问题出在哪儿
        report = self.make_tool()._report(
            "C:\\Program Files\\Microsoft Visual Studio\\VC\\include\\xmemory(12,3):"
            " error C2039: “x”: 不是“std”的成员\n"
        )

        self.assertEqual(len(report.diagnostics), 1)
        self.assertEqual(
            report.diagnostics[0].file,
            "C:/Program Files/Microsoft Visual Studio/VC/include/xmemory",
        )

    def test_a_line_that_is_not_a_diagnostic_does_not_stop_the_others(self):
        # 构建日志里夹着大段不是诊断的话。认不出来的原样放着，
        # 一条都不许挡着后面那些真诊断
        self.say(
            "msbuild",
            out=(
                "  正在创建目录“out\\”。\n"
                "  main.cpp\n"
                + self.said_by_msbuild(
                    "a.cpp", 2, 3, "error C2065: “undeclared_a”: 未声明的标识符"
                )
                + "  正在生成代码...\n"
            ),
        )
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assertIn("诊断（共 1 条：1 个 error）", result)
        self.assert_diagnosed(result, "a.cpp:2:3: error C2065:")

    def test_a_project_level_error_is_not_a_diagnostic(self):
        # MSB1009 这种工程级的错误没有位置可指，解不成诊断。
        # 不能为了凑一条就瞎猜文件行号——原文就在上面，模型自己看得到
        self.say("msbuild", out="MSBUILD : error MSB1009: 项目文件不存在。\n")
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assertIn("MSBUILD : error MSB1009", result)
        self.assertNotIn("诊断（", result)

    def test_a_project_level_error_is_kept_in_other(self):
        # 「解不出来」和「解出来是别人的」不是一回事：不丢，收在 other 里。
        # 这条是解析器的约定，这里钉住 BuildTool 也没把它吃掉
        report = self.make_tool()._report(
            "MSBUILD : error MSB1009: 项目文件不存在。\n开关:nope.vcxproj\n"
        )

        self.assertEqual(report.diagnostics, ())
        self.assertIn("MSBUILD : error MSB1009: 项目文件不存在。", report.other)

    def test_a_project_file_error_keeps_its_position(self):
        # MSB4025 指的位置是工程文件自己的第 1 行——那也是个能读的位置，
        # 照样解成诊断，模型拿着就能去 read_file 看那行
        self.say(
            "msbuild",
            out=(
                f"{self.root}\\broken.vcxproj(1,21): error MSB4025: "
                "未能加载项目文件。\n"
            ),
        )
        self.fail("msbuild")

        result = self.build(system="msbuild")

        self.assert_diagnosed(result, "broken.vcxproj:1:21: error MSB4025:")


class BuildToolRealTest(unittest.TestCase):
    """真的叫一次 msbuild。

    上面那些假 msbuild 验的是「命令拼得对不对、环境有没有传下去」。
    这一条验的是另一件事：拼出来的那条命令，真的 msbuild 认不认。

    这台机器上没装 Visual Studio 就跳过。
    """

    def test_a_real_project_builds(self):
        vsenv.reset_cache()
        if vsenv.find_program("msbuild") is None:
            self.skipTest("这台机器上没找到 Visual Studio 的 msbuild")

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "main.cpp").write_text(
            "#include <iostream>\n"
            "int main(){ std::cout << \"msbuild ok\" << std::endl; return 0; }\n",
            encoding="utf-8",
        )
        (root / "demo.vcxproj").write_text(_VCXPROJ, encoding="utf-8")

        result = BuildTool(root=root).build(system="msbuild")

        self.assertIn("退出码：0", result)
        # 退出码 0 也可能是「什么都没做就成功了」。产物真的躺在那里才算数
        self.assertTrue((root / "out" / "demo.exe").is_file(), result)


# 一个能真的编出东西来的最小 vcxproj。
# OutDir 写死成 out\，不然产物落在哪由 MSBuild 的默认值决定，没法断言。
_VCXPROJ = """<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{8A0E3E4B-6D2E-4C3B-9F5B-1C2D3E4F5A6B}</ProjectGuid>
    <RootNamespace>demo</RootNamespace>
  </PropertyGroup>
  <PropertyGroup>
    <Configuration>Release</Configuration>
    <Platform>x64</Platform>
    <OutDir>$(ProjectDir)out\\</OutDir>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <PropertyGroup Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <PlatformToolset>v143</PlatformToolset>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <ItemGroup>
    <ClCompile Include="main.cpp" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>
"""


class BuildToolRealDiagnosticsTest(unittest.TestCase):
    """真的用 msbuild 编一个编不过的工程，再回头看诊断里的位置对不对得上源码。

    上面那些假的验的是「解出来是什么形状」，这一条验的是另一件事：
    真 msbuild 报出来的那一行，抽出来的文件、行号、列号**就是源码里的那一处**。
    所以这里不拿 diagnostics 去解自己的输出——用被测的东西验它自己等于没验——
    而是自己写一条正则，再把列号戳回源码那一行上。

    这台机器上没装 Visual Studio 就跳过。
    """

    # 第 2 行那个分号是故意的：cl 会指着它说「语法错误」
    SOURCE = "int main() {\n    int answer = ;\n    return answer;\n}\n"

    def test_a_real_error_points_at_the_real_character(self):
        vsenv.reset_cache()
        if vsenv.find_program("msbuild") is None:
            self.skipTest("这台机器上没找到 Visual Studio 的 msbuild")

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "main.cpp").write_text(self.SOURCE, encoding="utf-8")
        (root / "demo.vcxproj").write_text(_VCXPROJ, encoding="utf-8")

        result = BuildTool(root=root).build(system="msbuild")

        self.assertIn("退出码：1", result)
        found = re.search(
            r"^\s+(?P<file>\S+):(?P<line>\d+):(?P<column>\d+): error C2059:",
            result,
            re.MULTILINE,
        )
        self.assertIsNotNone(found, result)

        # 路径是相对 root 的：绝对路径过不了路径白名单，read_file 读不了
        self.assertEqual(found["file"], "main.cpp")

        # 行号、列号戳回源码，得正好落在那个人为放的分号上
        number = int(found["line"])
        column = int(found["column"])
        line = self.SOURCE.split("\n")[number - 1]
        self.assertEqual(line[column - 1], ";", f"第 {number} 行第 {column} 列不是分号")

        # 同一个位置交给 read_file 也读得到——诊断里的行号就是它数的那一套
        shown = ReadFileTool(root=root).execute(found["file"])
        self.assertRegex(shown, rf"(?m)^\s*{number}\|    int answer = ;$")


class BuildToolGuardTest(BuildToolTestBase):
    def test_an_unknown_system_is_rejected(self):
        # ninja / bazel 是刻意不收的：这一阶段只加 VS 自带的那个，
        # 不引入新的构建系统
        for raw in ("ninja", "bazel", "xmake", "cmake; whoami"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.build(system=raw)

                self.assertIn("不认识的构建系统", str(caught.exception))

    def test_a_missing_system_is_reported_clearly(self):
        with patch("tools.build.shutil.which", return_value=None):
            with self.assertRaises(ToolError) as caught:
                self.build(system="cmake")

        self.assertIn("这台机器上没有 cmake", str(caught.exception))

    def test_source_root_outside_the_root_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.build(system="cmake", source_root="../outside")

        self.assertIn("超出允许范围", str(caught.exception))

    def test_an_absolute_source_root_is_rejected(self):
        with self.assertRaises(ToolError):
            self.build(system="cmake", source_root=str(PROJECT_ROOT))

    def test_build_dir_outside_the_root_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.build(system="cmake", build_dir="../outside")

        self.assertIn("超出允许范围", str(caught.exception))

    def test_shell_characters_in_source_root_are_rejected(self):
        for raw in ("src; whoami", "src & whoami", "src | more", "src`whoami`"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.build(system="cmake", source_root=raw)

                self.assertIn("只能用字母", str(caught.exception))

    def test_shell_characters_in_build_dir_are_rejected(self):
        error = self.build_catching(build_dir="out & whoami")

        self.assertIn("只能用字母", error)

    def build_catching(self, **kwargs) -> str:
        with self.assertRaises(ToolError) as caught:
            self.build(system="cmake", **kwargs)
        return str(caught.exception)

    def test_a_missing_source_root_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.build(system="cmake", source_root="nope")

        self.assertIn("源码目录不存在", str(caught.exception))

    def test_nothing_is_built_when_a_path_is_rejected(self):
        with self.assertRaises(ToolError):
            self.build(system="cmake", build_dir="../outside")

        self.assertFalse((self.root / "build").exists())

    def test_command_is_not_a_parameter(self):
        # 命令由工具内部生成，模型给不了 command
        self.assertNotIn("command", BuildTool.parameters)


class BuildToolShapeTest(unittest.TestCase):
    def test_it_is_a_tool(self):
        self.assertIsInstance(BuildTool(root=PROJECT_ROOT), Tool)
        self.assertEqual(BuildTool.name, "build")

    def test_it_reuses_run_command(self):
        # 这一层不许自己再写一遍 subprocess / 超时 / 输出处理
        self.assertIsInstance(
            BuildTool(root=PROJECT_ROOT).command_tool, RunCommandTool
        )

    def test_the_boundary_is_shared_with_run_command(self):
        tool = BuildTool(root=PROJECT_ROOT)

        self.assertEqual(tool.command_tool.root, tool.root)

    def test_schema_shape(self):
        parameters = BuildTool(root=PROJECT_ROOT).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(
            list(parameters["properties"]),
            ["system", "source_root", "build_dir", "timeout"],
        )
        self.assertEqual(parameters["required"], ["system"])
        self.assertEqual(parameters["properties"]["system"]["type"], "string")
        self.assertEqual(
            parameters["properties"]["source_root"]["type"], "string"
        )
        self.assertEqual(parameters["properties"]["build_dir"]["type"], "string")
        self.assertEqual(parameters["properties"]["timeout"]["type"], "number")

    def test_the_known_systems_are_exactly_these_three(self):
        # 钉住这个清单：多一个就是把新的构建系统引了进来，
        # 那得是单独决定的事，不该顺手发生
        self.assertEqual(SYSTEMS, ("cmake", "make", "msbuild"))

    def test_the_system_parameter_names_them(self):
        parameter = BuildTool(root=PROJECT_ROOT).to_schema()["function"][
            "parameters"
        ]["properties"]["system"]

        self.assertIn("msbuild", parameter["description"])

    def test_default_root_is_project_root(self):
        self.assertEqual(BuildTool().root, PROJECT_ROOT)

    def test_default_build_dir_is_build(self):
        self.assertEqual(DEFAULT_BUILD_DIR, "build")

    def test_default_timeout_is_sane(self):
        # 构建是最慢的一类命令，默认值不能比跑测试还紧
        self.assertGreaterEqual(DEFAULT_BUILD_TIMEOUT, 120)

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            BuildTool(root=Path("C:/"))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            BuildTool(root=Path("C:/definitely/not/a/real/dir/xyz"))


class StubClient:
    """按剧本回话的假客户端。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


def tool_call_reply(name: str, arguments: str, call_id: str = "c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def final_reply(content: str = "好的"):
    return {"role": "assistant", "content": content}


class BuildToolAgentTest(BuildToolTestBase):
    """走一遍完整的 Agent 调用链：模型要求构建 → 结果回灌。"""

    def build_agent(self, replies, **kwargs) -> Agent:
        return Agent(
            client=StubClient(replies),
            tools=ToolRegistry([self.make_tool(**kwargs)]),
        )

    def test_agent_can_build_with_cmake(self):
        agent = self.build_agent(
            [
                tool_call_reply("build", json.dumps({"system": "cmake"})),
                final_reply("构建完了"),
            ]
        )

        answer = agent.run("构建一下")

        self.assertEqual(answer, "构建完了")
        self.assertIn("[第 2/2 步]", agent.messages[3]["content"])
        self.assertIn("退出码：0", agent.messages[3]["content"])

    def test_agent_can_build_with_make(self):
        agent = self.build_agent(
            [
                tool_call_reply("build", json.dumps({"system": "make"})),
                final_reply("构建完了"),
            ]
        )

        agent.run("用 make 构建")

        self.assertIn("命令：make", agent.messages[3]["content"])

    def test_a_failed_build_does_not_break_the_turn(self):
        self.fail("build")
        agent = self.build_agent(
            [
                tool_call_reply("build", json.dumps({"system": "cmake"})),
                final_reply("构建失败了"),
            ]
        )

        answer = agent.run("构建一下")

        self.assertEqual(answer, "构建失败了")
        self.assertIn("build 失败", agent.messages[3]["content"])
        # 构建系统自己退出了非 0，但工具调用本身是成功的
        self.assertTrue(agent.get_trace()[0].success)

    def test_a_bad_system_does_not_break_the_turn(self):
        agent = self.build_agent(
            [
                tool_call_reply("build", json.dumps({"system": "ninja"})),
                final_reply("换个构建系统"),
            ]
        )

        answer = agent.run("用 ninja 构建")

        self.assertEqual(answer, "换个构建系统")
        self.assertIn("不认识的构建系统", agent.messages[3]["content"])
        self.assertFalse(agent.get_trace()[0].success)

    def test_the_call_is_recorded_in_the_trace(self):
        agent = self.build_agent(
            [
                tool_call_reply("build", json.dumps({"system": "make"})),
                final_reply("好了"),
            ]
        )

        agent.run("构建一下")

        events = agent.get_trace()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tool, "build")
        self.assertTrue(events[0].success)

    def test_it_is_registered_in_main(self):
        # main.build_agent() 会去读 API Key，这里把客户端换掉再装配
        import main

        with patch("agent.agent.LLMClient", StubClient):
            agent = main.build_agent()

        self.assertIn("build", [tool.name for tool in agent.tools.list()])


if __name__ == "__main__":
    unittest.main()
