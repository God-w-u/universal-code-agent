"""CompileTool 测试：编译成功、编译失败、错误信息、退出码、Agent 调用。

g++ 和 clang++ 这台机器上没有（`shutil.which` 找不到），所以真正跑编译器的
那几条用一个「假编译器」顶替：一个只模仿 g++ 对本工具重要那部分行为的
Python 脚本。它验的是「命令拼得对不对、结果怎么处理」，不是「g++ 能不能编译」。
假 cl 是同样的路子，多了一个 .bat 壳——cl 这条路的命令确实要过 shell 找。

除了假的，还有两条**真的**（CompileToolRealTest）：机器上装了 VS 才跑，
没有就跳过。假编译器不会因为你把 /std: 写成 /std- 就报错，真的会。

全部离线，不联网。
"""

import json
import os
import re
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from agent import Agent
from config import PROJECT_ROOT
from tools import (
    CompileTool,
    ReadFileTool,
    RunCommandTool,
    Tool,
    ToolError,
    ToolRegistry,
    diagnostics,
    vsenv,
)
from tools.compile import (
    DEFAULT_COMPILE_TIMEOUT,
    _is_output_flag,
    _to_msvc,
    find_compiler,
)

# 假编译器：接口和 g++ 一样。源码里有 #error 就按真实的方式报错退出 1，
# 否则写出产物退出 0，并把自己收到的参数原样报出来——测试靠这行确认
# flags 真的变成了独立的参数，而不是粘在别的词上。
FAKE_COMPILER = '''\
"""假编译器。"""
import sys
from pathlib import Path

args = sys.argv[1:]
# 源文件是 -o 前面那一个：命令里 flags 在最前、"-o 输出" 永远在最后
output_at = args.index("-o")
source = args[output_at - 1]
output = args[output_at + 1]

print("argv: " + " ".join(args))

text = Path(source).read_text(encoding="utf-8")

if "#error" in text:
    print(f"{source}:3:2: error: #error 这条错误是故意写错的", file=sys.stderr)
    print("compilation terminated.", file=sys.stderr)
    sys.exit(1)

if "#warning" in text:
    # 编过了但带警告。这条也走 stderr，和真的 g++ 一样
    print(f"{source}:2:9: warning: 未使用的变量 ‘x’ [-Wunused-variable]", file=sys.stderr)

Path(output).write_text("假的可执行文件", encoding="utf-8")
print(f"compiled {source} -> {output}")
'''

GOOD_SOURCE = "int main() { return 0; }\n"
BAD_SOURCE = "#include <iostream>\nint main() {\n#error 这条错误是故意写错的\n}\n"
WARN_SOURCE = (
    "int main() {\n"  # 1
    "  int x;\n"  # 2 —— 假 g++ 说这一行有警告
    "#warning 看到这一行就报一条警告，然后照常编过\n"  # 3
    "  return 0;\n"  # 4
    "}\n"  # 5
)

# 假 cl.exe：参数写法是 MSVC 那一套。产物写作 /Fe: 后面那个名字，
# 不补 .exe——真的 cl 会补，但那是它的行为，这里要钉的是这个工具的行为。
FAKE_CL = '''\
"""假 cl.exe。"""
import sys
from pathlib import Path

args = sys.argv[1:]
print("argv: " + " ".join(args))
# cl 能不能用，全看 vcvars 铺的那份环境有没有传进来。留个记号好验
import os
print("marker: " + os.environ.get("FAKE_VS_MARKER", "（没传进来）"))

# 源文件是唯一那个不以 / 或 - 开头的参数，产物写在 /Fe: 后面
source = next(a for a in args if not a.startswith(("/", "-")))
output = next(a for a in args if a.startswith("/Fe:"))[4:]

text = Path(source).read_text(encoding="utf-8")

if "#error" in text:
    # 真的 cl 就是打在标准输出里的，不是标准错误
    print(f"{source}(3): error C2059: 语法错误（这条是故意写错的）")
    sys.exit(2)

if "#warning" in text:
    # 警告也走标准输出，也一样编得过
    print(f"{source}(2): warning C4101: “x”: 未引用的局部变量")

Path(output).write_text("假的可执行文件", encoding="utf-8")
print(f"compiled {source} -> {output}")
'''

# 假 cl 的 .bat 壳。cl 这条路命令要真的过 shell 才找得到它，
# 和 build 那边的假 cmake / 假 make 是同一套做法。
CL_BAT = '@echo off\n"{python}" "{script}" %*\nexit /b %ERRORLEVEL%\n'


def parts(result: str) -> tuple[str, str, str]:
    """把结果拆成 (头部, 标准输出, 标准错误) 三段。

    末尾那段诊断（tools/diagnostics.py）是后来加的第四段，拆之前先摘掉——
    不摘的话它会被算进标准错误，那些「标准错误里就只有编译器的原话」
    的断言就变味了。
    """
    head, _, rest = result.partition("\n\n诊断（")[0].partition("\n\n标准输出：\n")
    stdout, _, stderr = rest.partition("\n\n标准错误：\n")
    return head, stdout, stderr


def _installed(*names):
    """假装这台机器上装了这几个编译器（别的都没有）。

    这里必须把 cl 那条路也堵上：它不走 shutil.which，而是去问 VS。
    只挡 shutil.which 的话，装了 VS 的机器上「一个都没装」根本不成立。
    """
    stack = ExitStack()
    stack.enter_context(
        patch(
            "tools.compile.shutil.which",
            side_effect=lambda name: f"/usr/bin/{name}" if name in names else None,
        )
    )
    stack.enter_context(patch("tools.compile.vsenv.find_program", return_value=None))
    return stack


class CompileToolTestBase(unittest.TestCase):
    """搭一个假项目：临时目录 + 一个假编译器。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

        script = self.root / "fake_gpp.py"
        script.write_text(FAKE_COMPILER, encoding="utf-8")
        self.compiler = f'"{sys.executable}" "{script}"'

    def make_tool(self, **kwargs) -> CompileTool:
        return CompileTool(root=self.root, compiler=self.compiler, **kwargs)

    def write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def compile(self, **kwargs) -> str:
        return self.make_tool().run(**kwargs)


class CompileToolSuccessTest(CompileToolTestBase):
    def setUp(self):
        super().setUp()
        self.write("main.cpp", GOOD_SOURCE)
        self.result = self.compile(source="main.cpp")
        self.head, self.stdout, self.stderr = parts(self.result)

    def test_a_good_file_compiles(self):
        self.assertIn("退出码：0", self.head)

    def test_zero_exit_code_is_not_marked_as_a_failure(self):
        self.assertNotIn("非 0", self.head)

    def test_the_command_is_generated_for_you(self):
        # 模型给的是源文件名，命令是工具拼的
        self.assertIn("main.cpp -o main", self.head)

    def test_the_compiler_output_is_kept(self):
        self.assertIn("compiled main.cpp", self.stdout)

    def test_all_four_pieces_are_present(self):
        self.assertIn("命令：", self.head)
        self.assertIn("退出码：", self.head)
        self.assertIn("标准输出：", self.result)
        self.assertIn("标准错误：", self.result)

    def test_the_output_file_is_created(self):
        self.assertTrue((self.root / "main").is_file())

    def test_a_custom_output_name_is_used(self):
        self.write("app.cpp", GOOD_SOURCE)

        result = self.compile(source="app.cpp", output="tool")

        self.assertIn("app.cpp -o tool", result)
        self.assertTrue((self.root / "tool").is_file())

    def test_the_output_keeps_the_source_directory(self):
        # src/app.cpp -> src/app，不是丢到根目录去
        self.write("src/app.cpp", GOOD_SOURCE)

        result = self.compile(source="src/app.cpp")

        self.assertIn("src/app.cpp -o src/app", result)
        self.assertTrue((self.root / "src" / "app").is_file())

    def test_a_dot_slash_prefix_is_normalized_away(self):
        # 命令里出现的是校验过的相对路径，不是模型原样递进来的那一份
        result = self.compile(source="./main.cpp")

        self.assertIn("main.cpp -o main", result)
        self.assertNotIn("./main.cpp", result)


class CompileToolFailureTest(CompileToolTestBase):
    def setUp(self):
        super().setUp()
        self.write("broken.cpp", BAD_SOURCE)
        self.result = self.compile(source="broken.cpp")
        self.head, self.stdout, self.stderr = parts(self.result)

    def test_a_broken_file_reports_a_non_zero_exit_code(self):
        self.assertIn("退出码：1", self.head)

    def test_a_failed_compile_does_not_raise(self):
        self.assertIsInstance(self.result, str)

    def test_failure_is_stated_in_words(self):
        self.assertIn("非 0，命令失败", self.head)

    def test_the_compiler_error_is_kept(self):
        # 模型要靠这段文本才知道错在哪一行，不能吞掉
        self.assertIn("error: #error 这条错误是故意写错的", self.stderr)

    def test_the_error_keeps_the_file_and_line(self):
        self.assertIn("broken.cpp:3:2", self.stderr)

    def test_no_output_file_is_produced(self):
        self.assertFalse((self.root / "broken").exists())


class CompileToolCompilerTest(CompileToolTestBase):
    """编译器：这次调用指定的 > 装配时指定的 > 自动找。"""

    def setUp(self):
        super().setUp()
        self.write("main.cpp", GOOD_SOURCE)

    def installed(self, *names):
        """假装这几个编译器装在这台机器上。

        不这么假装的话，断言就得看这台机器实际装了什么——本机有 VS 的 cl，
        而 CI 或者别人的机器上未必。

        两张补丁缺一不可：shutil.which 管 g++ / clang++，
        vsenv.find_program 管藏在 VS 里的 cl——只挡前者的话，
        「一个都没装」的场景在这台机器上根本不成立。
        """
        return _installed(*names)

    def test_the_constructor_compiler_is_the_default(self):
        result = self.compile(source="main.cpp")

        self.assertIn(sys.executable, result)  # 装配时钉进去的假编译器
        self.assertNotIn("clang++", result)

    def test_the_requested_compiler_wins(self):
        # 装配时钉的是假编译器，模型点名要 clang++：这次调用说了算。
        # 命令里出现 clang++ 就说明假编译器没被用上
        with self.installed("clang++", "g++"):
            result = self.compile(source="main.cpp", compiler="clang++")

        self.assertIn("clang++ main.cpp -o main", result)

    def test_gpp_can_be_requested_too(self):
        with self.installed("clang++", "g++"):
            result = self.compile(source="main.cpp", compiler="g++")

        self.assertIn("g++ main.cpp -o main", result)

    def test_a_requested_compiler_still_gets_the_flags(self):
        with self.installed("clang++"):
            result = self.compile(source="main.cpp", compiler="clang++", flags="-O2")

        self.assertIn("clang++ -O2 main.cpp -o main", result)

    def test_a_missing_compiler_is_reported_clearly(self):
        # 比 shell 那句「不是内部或外部命令」有用：说清楚是没装，不是名字写错
        with self.installed("g++"):
            with self.assertRaises(ToolError) as caught:
                self.compile(source="main.cpp", compiler="clang++")

        self.assertIn("没有它", str(caught.exception))
        self.assertIn("装了的是 g++", str(caught.exception))

    def test_the_hint_says_so_when_nothing_is_installed(self):
        with self.installed():
            with self.assertRaises(ToolError) as caught:
                self.compile(source="main.cpp", compiler="g++")

        self.assertIn("一个都没装", str(caught.exception))

    def test_an_unknown_compiler_is_rejected(self):
        # 编译器名字也是拼进命令的，所以只能是清单里那几个
        for raw in ("clang", "nvcc", "g++; whoami", "../../evil"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.compile(source="main.cpp", compiler=raw)

                self.assertIn("不认识的编译器", str(caught.exception))

    def test_a_bad_compiler_does_not_run_anything(self):
        with self.assertRaises(ToolError):
            self.compile(source="main.cpp", compiler="rm -rf /")


class CompileToolFlagsTest(CompileToolTestBase):
    def setUp(self):
        super().setUp()
        self.write("main.cpp", GOOD_SOURCE)

    def argv(self, result: str) -> str:
        """假编译器报出来的「它实际收到的参数」。"""
        return parts(result)[1]

    def test_flags_are_passed_through(self):
        result = self.compile(source="main.cpp", flags="-std=c++11")

        self.assertIn("-std=c++11 main.cpp -o main", result)

    def test_the_compiler_receives_them_as_separate_arguments(self):
        result = self.compile(source="main.cpp", flags="-std=c++11 -O2 -Wall")

        self.assertIn("argv: -std=c++11 -O2 -Wall main.cpp -o main", self.argv(result))

    def test_no_flags_means_no_extra_spaces(self):
        self.assertIn("argv: main.cpp -o main", self.argv(self.compile(source="main.cpp")))

    def test_extra_whitespace_is_normalized(self):
        result = self.compile(source="main.cpp", flags="  -O2   -Wall  ")

        self.assertIn("argv: -O2 -Wall main.cpp -o main", self.argv(result))

    def test_flags_still_produce_the_output(self):
        self.compile(source="main.cpp", flags="-std=c++11")

        self.assertTrue((self.root / "main").is_file())

    def test_flags_work_with_a_custom_output_name(self):
        result = self.compile(source="main.cpp", output="tool", flags="-O2")

        self.assertIn("argv: -O2 main.cpp -o tool", self.argv(result))

    def test_uppercase_o_is_not_confused_with_lowercase_o(self):
        # -O2 是优化级别，-o 才是输出，只差一个大小写
        result = self.compile(source="main.cpp", flags="-O2")

        self.assertIn("argv: -O2 main.cpp -o main", self.argv(result))

    def test_shell_characters_in_flags_are_rejected(self):
        for raw in (
            "-O2; whoami",
            "-O2 & whoami",
            "-O2 | more",
            "-O2 > out.txt",
            "-O2 `whoami`",
            "-O2 $(whoami)",
            '-std="c++11"',
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.compile(source="main.cpp", flags=raw)

                self.assertIn("编译参数里不能有", str(caught.exception))

    def test_output_flags_are_rejected(self):
        # 输出到哪儿是 output 参数的事，不从 flags 里再指一遍
        for raw in ("-o other", "-oother", "--output=other", "-MF deps.txt"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.compile(source="main.cpp", flags=raw)

                self.assertIn("输出到哪儿由 output 参数决定", str(caught.exception))

    def test_a_rejected_flag_does_not_compile_anything(self):
        with self.assertRaises(ToolError):
            self.compile(source="main.cpp", flags="-o ../outside")

        self.assertFalse((self.root / "main").exists())


class CompileToolPathTest(CompileToolTestBase):
    """路径参数是拼进命令的，这一组盯的是「拼进去的东西能不能改变命令的意思」。"""

    def setUp(self):
        super().setUp()
        self.write("main.cpp", GOOD_SOURCE)

    def test_source_outside_root_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.compile(source="../outside.cpp")

        self.assertIn("超出允许范围", str(caught.exception))

    def test_absolute_source_is_rejected(self):
        with self.assertRaises(ToolError):
            self.compile(source=str(PROJECT_ROOT / "main.py"))

    def test_output_outside_root_is_rejected(self):
        with self.assertRaises(ToolError):
            self.compile(source="main.cpp", output="../outside")

    def test_shell_characters_in_source_are_rejected(self):
        for raw in ("main.cpp; whoami", "main.cpp & whoami", "main.cpp | more"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.compile(source=raw)

                self.assertIn("只能用字母", str(caught.exception))

    def test_shell_characters_in_output_are_rejected(self):
        # 输出文件还不存在，挡不住它的是「文件不存在」，只能是这层校验
        with self.assertRaises(ToolError) as caught:
            self.compile(source="main.cpp", output="app & whoami")

        self.assertIn("只能用字母", str(caught.exception))

    def test_backticks_are_rejected(self):
        with self.assertRaises(ToolError):
            self.compile(source="main.cpp", output="app`whoami`")

    def test_a_missing_source_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.compile(source="nope.cpp")

        self.assertIn("源文件不存在", str(caught.exception))

    def test_an_empty_source_is_rejected(self):
        with self.assertRaises(ToolError):
            self.compile(source="")

    def test_command_is_not_a_parameter(self):
        # 编译命令由工具内部生成，模型给不了 command
        self.assertNotIn("command", CompileTool.parameters)

    def test_the_safety_rule_still_allows_normal_paths(self):
        self.write("src/deep/my_app-2.cpp", GOOD_SOURCE)

        result = self.compile(source="src/deep/my_app-2.cpp")

        self.assertIn("退出码：0", result)


class CompileToolShapeTest(unittest.TestCase):
    def test_it_is_a_tool(self):
        self.assertIsInstance(CompileTool(root=PROJECT_ROOT), Tool)
        self.assertEqual(CompileTool.name, "compile")

    def test_it_reuses_run_command(self):
        # 这一层不许自己再写一遍 subprocess / 超时 / 输出处理
        self.assertIsInstance(
            CompileTool(root=PROJECT_ROOT).command_tool, RunCommandTool
        )

    def test_the_boundary_is_shared_with_run_command(self):
        tool = CompileTool(root=PROJECT_ROOT)

        self.assertEqual(tool.command_tool.root, tool.root)

    def test_schema_shape(self):
        parameters = CompileTool(root=PROJECT_ROOT).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(
            list(parameters["properties"]),
            ["source", "output", "compiler", "flags", "timeout"],
        )
        self.assertEqual(parameters["required"], ["source"])
        for name, type_name in (
            ("source", "string"),
            ("output", "string"),
            ("compiler", "string"),
            ("flags", "string"),
            ("timeout", "number"),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    parameters["properties"][name]["type"], type_name
                )

    def test_default_root_is_project_root(self):
        self.assertEqual(CompileTool().root, PROJECT_ROOT)

    def test_default_timeout_is_sane(self):
        # 编译比跑一条普通命令慢，默认值不能被压到 run_command 那个量级
        self.assertGreaterEqual(DEFAULT_COMPILE_TIMEOUT, 30)

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            CompileTool(root=Path("C:/"))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            CompileTool(root=Path("C:/definitely/not/a/real/dir/xyz"))

    def test_no_compiler_raises_a_clear_error(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "main.cpp").write_text(GOOD_SOURCE, encoding="utf-8")

        with patch("tools.compile.find_compiler", return_value=None):
            with self.assertRaises(ToolError) as caught:
                CompileTool(root=root).run(source="main.cpp")

        self.assertIn("找不到 C++ 编译器", str(caught.exception))

    def test_an_explicit_compiler_beats_detection(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "main.cpp").write_text(GOOD_SOURCE, encoding="utf-8")

        with patch("tools.compile.find_compiler", return_value="clang++"):
            result = CompileTool(root=root, compiler="g++").run(source="main.cpp")

        self.assertIn("g++ main.cpp", result)

    def test_the_detected_compiler_is_used(self):
        # 这条不假装能编译：机器上没装 g++ 时，shell 只会回一句「找不到命令」，
        # 那也是一个正常结果。它要钉住的是「自动找到的名字真的被拼进命令了」。
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "main.cpp").write_text(GOOD_SOURCE, encoding="utf-8")

        with patch("tools.compile.find_compiler", return_value="g++"):
            result = CompileTool(root=root).run(source="main.cpp")

        self.assertIn("g++ main.cpp", result)

    def test_find_compiler_only_returns_known_names(self):
        # 结果取决于这台机器装了什么，所以只钉「不会返回别的东西」。
        # 编译器名字是拼进命令的，多一个能返回的名字就多一条缝
        self.assertIn(find_compiler(), (None, "g++", "clang++", "cl"))

    def test_cl_is_only_reached_after_the_others(self):
        # 顺序是有意排的：用 cl 得先铺一遍 vcvars（这台机器上实测 2.5 秒），
        # 机器上真装了 g++ 的话没理由付这个成本
        with _installed("g++", "clang++"):
            self.assertEqual(find_compiler(), "g++")

        with _installed("clang++"):
            self.assertEqual(find_compiler(), "clang++")

    def test_the_error_names_every_compiler_it_looked_for(self):
        with _installed():
            with self.assertRaises(ToolError) as caught:
                CompileTool(root=PROJECT_ROOT).run(source="main.py")

        self.assertIn("cl", str(caught.exception))

    def test_a_compiler_name_cannot_be_a_path(self):
        # 编译器名字是直接拼进命令的。它只能是清单里那三个死名字，
        # 路径、带参数的写法、别的可执行文件都得挡在外面
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "main.cpp").write_text(GOOD_SOURCE, encoding="utf-8")

        for raw in (
            "cl.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\...\cl.exe",
            "cl /c calc",
            "cl&&whoami",
            "cl;whoami",
            "./cl",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    CompileTool(root=root).run(source="main.cpp", compiler=raw)

                self.assertIn("不认识的编译器", str(caught.exception))


class MSVCFlagTranslationTest(unittest.TestCase):
    """g++ 写法的参数怎么翻成 cl 的写法。

    这一层是纯字符串，不用编译器也能测——而且必须单独测：
    翻错了 cl 不会报错，只会打一句 warning 然后**照常编**，
    是那种「看起来成功了，其实参数没生效」的错。
    """

    def translate(self, flags: str) -> str:
        return _to_msvc(flags)

    def test_the_standard_is_renamed(self):
        self.assertEqual(self.translate("-std=c++17"), "/std:c++17")

    def test_cpp11_becomes_cpp14(self):
        # cl 没有 C++11 模式，c++14 是它的最低档，向下兼容 C++11。
        # 不翻的话 cl 会打一句「未知选项」然后把整个 -std 忽略掉
        self.assertEqual(self.translate("-std=c++11"), "/std:c++14")

    def test_the_other_std_spellings_are_normalized(self):
        for given, expected in (
            ("-std=c++1z", "/std:c++17"),
            ("-std=c++2a", "/std:c++20"),
            ("-std=c++2b", "/std:c++latest"),
            ("-std=c++latest", "/std:c++latest"),
        ):
            with self.subTest(given=given):
                self.assertEqual(self.translate(given), expected)

    def test_an_unknown_std_is_rejected(self):
        # 宁可不编也不猜：cl 认不出 /std:xxx 时只打个 warning 就跳过
        for given in ("-std=c++03", "-std=gnu++17", "-std=c99"):
            with self.subTest(given=given):
                with self.assertRaises(ToolError) as caught:
                    self.translate(given)

                self.assertIn("cl 没有", str(caught.exception))

    def test_optimisation_levels(self):
        for given, expected in (
            ("-O0", "/Od"),
            ("-O1", "/O1"),
            ("-O2", "/O2"),
            ("-O3", "/O2"),  # cl 没有 /O3
            ("-Os", "/O1"),  # cl 的 /O1 就是「优先编得小」
        ):
            with self.subTest(given=given):
                self.assertEqual(self.translate(given), expected)

    def test_warning_flags(self):
        # 对应关系不是一对一的：-Wall 和 -Wextra 都落在 /W4 上，
        # 因为 cl 的 /Wall 是「把所有警告全打开」，噪音大到没法用
        for given in ("-Wall", "-Wextra"):
            with self.subTest(given=given):
                self.assertEqual(self.translate(given), "/W4")

    def test_debug_info_and_silence(self):
        self.assertEqual(self.translate("-g"), "/Zi")
        self.assertEqual(self.translate("-w"), "/w")

    def test_defines_and_include_paths(self):
        self.assertEqual(self.translate("-DDEBUG=1"), "/DDEBUG=1")
        self.assertEqual(self.translate("-Iinclude"), "/Iinclude")
        self.assertEqual(self.translate("-DNAME"), "/DNAME")

    def test_several_flags_together(self):
        self.assertEqual(
            self.translate("-std=c++17 -O2 -Wall -DNDEBUG -Iinclude"),
            "/std:c++17 /O2 /W4 /DNDEBUG /Iinclude",
        )

    def test_a_flag_it_does_not_know_is_rejected_not_guessed(self):
        # 悄悄丢掉一个参数会让模型以为它生效了，那是更坏的失败
        for given in ("-pthread", "-fsanitize=address", "-march=native", "-lz", "-c"):
            with self.subTest(given=given):
                with self.assertRaises(ToolError) as caught:
                    self.translate(given)

                self.assertIn("不知道怎么翻", str(caught.exception))

    def test_one_bad_flag_rejects_the_whole_thing(self):
        # 一半生效一半不生效是最难查的状态
        with self.assertRaises(ToolError):
            self.translate("-O2 -pthread")


class MSVCOutputFlagTest(unittest.TestCase):
    """产物写到哪儿是 output 参数的事，两条路都堵上。"""

    def test_msvc_output_flags_are_recognized(self):
        for token in ("/Fe:main", "/FO:main.obj", "/Fd:main.pdb", "/Fa:main.asm"):
            with self.subTest(token=token):
                self.assertTrue(_is_output_flag(token))

    def test_msvc_flags_are_matched_case_insensitively(self):
        # cl 的参数不分大小写，/FO 和 /fo 说的是同一件事
        self.assertTrue(_is_output_flag("/fo:main.obj"))

    def test_the_gcc_ones_still_work(self):
        for token in ("-o", "-oother", "--output=x", "-MFdeps.txt"):
            with self.subTest(token=token):
                self.assertTrue(_is_output_flag(token))

    def test_normal_flags_are_not_mistaken_for_output_flags(self):
        for token in ("-O2", "/O2", "/std:c++17", "/W4", "-Wall", "/EHsc"):
            with self.subTest(token=token):
                self.assertFalse(_is_output_flag(token))


class CompileToolMSVCTestBase(CompileToolTestBase):
    """把假 cl 挂到 PATH 上，再让工具以为 VS 环境就是这个环境。

    真实的 cl 不在 PATH 上，是 vcvars 把它铺进去的。这里用同样的形状：
    假 cl 放在一个临时目录里，让 vcvars_env() 返回的 PATH 里有它。
    """

    def setUp(self):
        super().setUp()

        script = self.root / "fake_cl.py"
        script.write_text(FAKE_CL, encoding="utf-8")
        bat = self.root / "cl.bat"
        bat.write_text(
            CL_BAT.format(python=sys.executable, script=script), encoding="utf-8"
        )

        # 工具自己的 which 找的是这台机器上的真 PATH，所以真的把它加上去
        env = patch.dict(os.environ, {"PATH": f"{self.root}{os.pathsep}{os.environ['PATH']}"})
        env.start()
        self.addCleanup(env.stop)

        # 子进程拿到的那份环境：真实的 vcvars_env() 里没有这个临时目录，
        # 所以这里换成当前环境——形状一样（一份变量字典），内容换成能跑通的
        vs = patch("tools.compile.vsenv.vcvars_env", side_effect=lambda: dict(os.environ))
        vs.start()
        self.addCleanup(vs.stop)

    def compile_cl(self, **kwargs) -> str:
        return self.make_tool().run(compiler="cl", **kwargs)


class CompileToolMSVCCommandTest(CompileToolMSVCTestBase):
    """命令长什么样。假 cl 会把它收到的参数原样报出来。"""

    def setUp(self):
        super().setUp()
        self.write("main.cpp", GOOD_SOURCE)
        self.result = self.compile_cl(source="main.cpp")
        self.head, self.stdout, self.stderr = parts(self.result)

    def test_it_compiles(self):
        self.assertIn("退出码：0", self.head)

    def test_the_program_is_cl(self):
        self.assertIn("命令：cl ", self.head)

    def test_the_output_flag_is_the_msvc_one(self):
        # g++ 是 -o，cl 是 /Fe:，这条是这一阶段的核心
        self.assertIn("/Fe:main", self.head)
        self.assertNotIn("-o main", self.head)

    def test_the_banner_and_exceptions_are_switched_on(self):
        # /nologo 去横幅；/EHsc 开异常——g++ 默认就开，cl 不写会差一截
        self.assertIn("/nologo", self.head)
        self.assertIn("/EHsc", self.head)

    def test_utf8_sources_are_switched_on(self):
        # 这条是实测逼出来的：cl 默认按系统代码页读源码，
        # 中文 Windows 上 UTF-8 的中文源码会被读歪，报出来的错完全看不出
        # 真正的原因是编码。/utf-8 让 cl 和 g++ 对同一份源码的理解一致。
        self.assertIn("/utf-8", self.head)

    def test_columns_are_switched_on(self):
        # 默认 cl 只报到行（`a.cpp(3): error`），带上这个才是 `a.cpp(3,3)`,
        # 和 g++ 的 `a.cpp:3:3:` 对得上——诊断里的 column 全靠它
        self.assertIn("/diagnostics:column", self.head)

    def test_the_output_file_is_created(self):
        self.assertTrue((self.root / "main").is_file())

    def test_the_output_keeps_the_source_directory(self):
        self.write("src/app.cpp", GOOD_SOURCE)

        result = self.compile_cl(source="src/app.cpp")

        self.assertIn("/Fe:src/app", result)
        self.assertTrue((self.root / "src" / "app").is_file())

    def test_msvc_gets_one_argument_per_token(self):
        result = self.compile_cl(source="main.cpp", flags="-std=c++17 -O2")

        self.assertIn(
            "argv: /nologo /EHsc /utf-8 /diagnostics:column "
            "/std:c++17 /O2 main.cpp /Fe:main",
            parts(result)[1],
        )

    def test_the_flags_are_translated_on_the_way_in(self):
        result = self.compile_cl(source="main.cpp", flags="-std=c++11 -Wall")

        self.assertIn("/std:c++14", result)
        self.assertIn("/W4", result)
        self.assertNotIn("-std=c++11", result)

    def test_the_compiler_gets_the_vs_environment(self):
        # 环境得真的传到子进程里。不传的话，真的 cl 会因为找不到头文件
        # 报一堆看不懂的错——那是这一整条路最容易断的地方。
        # 假 cl 会把它收到的那个记号报出来，看见记号就说明环境送到了
        marker = patch.dict(os.environ, {"FAKE_VS_MARKER": "送到了"})
        marker.start()
        self.addCleanup(marker.stop)

        result = self.compile_cl(source="main.cpp")

        self.assertIn("marker: 送到了", parts(result)[1])


class CompileToolMSVCFailureTest(CompileToolMSVCTestBase):
    def setUp(self):
        super().setUp()
        self.write("broken.cpp", BAD_SOURCE)
        self.result = self.compile_cl(source="broken.cpp")
        self.head, self.stdout, self.stderr = parts(self.result)

    def test_a_broken_file_reports_the_msvc_exit_code(self):
        # cl 编译错误退的是 2，不是 1
        self.assertIn("退出码：2", self.head)

    def test_a_failed_compile_does_not_raise(self):
        self.assertIsInstance(self.result, str)

    def test_the_error_is_readable(self):
        self.assertIn("error C2059", self.stdout)

    def test_the_error_is_kept_even_though_it_is_on_stdout(self):
        # 真的 cl 把报错打在标准输出里。只扫标准错误的话，
        # 模型会拿到一句「退出码 2」加一片空白
        self.assertIn("error C2059", self.result)
        self.assertEqual(self.stderr.strip(), "（空）")

    def test_no_output_file_is_produced(self):
        self.assertFalse((self.root / "broken").exists())

    def test_an_untranslatable_flag_does_not_reach_the_compiler(self):
        with self.assertRaises(ToolError) as caught:
            self.compile_cl(source="broken.cpp", flags="-pthread")

        self.assertIn("不知道怎么翻", str(caught.exception))


class CompileToolMSVCGuardTest(CompileToolMSVCTestBase):
    """安全检查在 cl 这条路上一样得过。"""

    def setUp(self):
        super().setUp()
        self.write("main.cpp", GOOD_SOURCE)

    def test_output_flags_in_the_msvc_spelling_are_rejected(self):
        for raw in ("/Fe:other", "/FO:main.obj", "-o other"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.compile_cl(source="main.cpp", flags=raw)

                self.assertIn("输出到哪儿由 output 参数决定", str(caught.exception))

    def test_shell_characters_are_still_rejected(self):
        for raw in ("-O2; whoami", "-O2 & whoami", '-D"x"', "-O2 | more"):
            with self.subTest(raw=raw):
                with self.assertRaises(ToolError) as caught:
                    self.compile_cl(source="main.cpp", flags=raw)

                self.assertIn("编译参数里不能有", str(caught.exception))

    def test_paths_still_cannot_escape_the_root(self):
        with self.assertRaises(ToolError):
            self.compile_cl(source="../outside.cpp")

    def test_the_source_must_still_exist(self):
        with self.assertRaises(ToolError) as caught:
            self.compile_cl(source="nope.cpp")

        self.assertIn("源文件不存在", str(caught.exception))

    def test_cl_is_useful_as_a_configured_default(self):
        # 装配时就指定 cl：这次调用不用再点名
        result = CompileTool(root=self.root, compiler="cl").run(source="main.cpp")

        self.assertIn("命令：cl ", result)

    def test_a_machine_without_visual_studio_says_so(self):
        with _installed("g++"):
            with self.assertRaises(ToolError) as caught:
                self.make_tool().run(source="main.cpp", compiler="cl")

        self.assertIn("没有它", str(caught.exception))
        self.assertIn("装了的是 g++", str(caught.exception))


class CompileToolDiagnosticsTest(CompileToolTestBase):
    """编译结果末尾那段解析好的诊断（tools/diagnostics.py）。

    解析器自己认得对不对，tests/test_diagnostics.py 已经钉过了。
    这里钉的是另一件事：那段东西**有没有接上来**，以及接上来之后
    原始的编译输出有没有被动过。
    """

    def block(self, result: str) -> str:
        """结果末尾那段诊断；没有就返回空串。"""
        _, separator, tail = result.partition("\n\n诊断（")
        return f"诊断（{tail}" if separator else ""

    def test_a_failed_compile_gets_a_diagnostics_block(self):
        self.write("broken.cpp", BAD_SOURCE)

        self.assertIn("诊断（", self.compile(source="broken.cpp"))

    def test_the_block_names_the_file_and_line(self):
        self.write("broken.cpp", BAD_SOURCE)

        block = self.block(self.compile(source="broken.cpp"))

        self.assertIn("broken.cpp:3:2: error:", block)

    def test_the_block_is_one_line_per_diagnostic(self):
        self.write("broken.cpp", BAD_SOURCE)

        block = self.block(self.compile(source="broken.cpp"))

        self.assertEqual(len(block.strip().splitlines()), 2)  # 标题 + 一条

    def test_the_original_output_survives_untouched(self):
        # 要求是「不要破坏原始编译输出」：诊断是**另起一段附在后面**，
        # 前面那段一个字符都不能变——模型仍然要能读到编译器报的原文
        self.write("broken.cpp", BAD_SOURCE)

        result = self.compile(source="broken.cpp")
        head, stdout, stderr = parts(result.partition("\n\n诊断（")[0])

        self.assertIn("退出码：1", head)
        self.assertIn("compilation terminated.", stderr)
        self.assertIn("broken.cpp:3:2: error: #error", stderr)
        self.assertIn("argv: ", stdout)

    def test_the_block_comes_last(self):
        self.write("broken.cpp", BAD_SOURCE)

        result = self.compile(source="broken.cpp")

        self.assertGreater(result.index("诊断（"), result.index("标准错误："))

    def test_a_clean_compile_gets_no_block(self):
        # 一切正常时不该多出一段废话
        self.write("good.cpp", GOOD_SOURCE)

        self.assertEqual(self.block(self.compile(source="good.cpp")), "")

    def test_a_warning_gets_a_block_even_though_it_compiled(self):
        # 退出码是 0，但警告一样是要改代码才能消掉的
        self.write("warn.cpp", WARN_SOURCE)

        result = self.compile(source="warn.cpp")

        self.assertIn("退出码：0", result)
        self.assertIn("warn.cpp:2:9: warning -Wunused-variable:", self.block(result))

    def test_the_count_line_says_how_many_of_each(self):
        self.write("warn.cpp", WARN_SOURCE)

        self.assertIn("1 个 warning", self.block(self.compile(source="warn.cpp")))

    def test_the_line_number_works_with_read_file(self):
        # 整条链路上最要紧的一步：这段里报的行号，拿去 read_file 读出来的
        # 必须是出问题的那一行，不能差一
        self.write("broken.cpp", BAD_SOURCE)

        block = self.block(self.compile(source="broken.cpp"))
        number = int(re.search(r"broken\.cpp:(\d+):", block).group(1))

        text = ReadFileTool(root=self.root).run(path="broken.cpp")

        self.assertIn("#error 这条错误是故意写错的", text.splitlines()[number - 1])

    def test_an_unrecognized_output_has_no_block(self):
        # 认不出来就什么都不附——原始输出还在，模型自己看
        tool = self.make_tool()

        result = tool._with_diagnostics("命令：x\n退出码：1\n\n标准输出：\n哎哟\n")

        self.assertNotIn("诊断（", result)

    def test_a_note_alone_has_no_block(self):
        tool = self.make_tool()

        result = tool._with_diagnostics("a.cpp:1:1: note: 'x' declared here\n")

        self.assertNotIn("诊断（", result)

    def test_the_text_it_is_given_is_returned_unchanged(self):
        tool = self.make_tool()
        original = "命令：x\n退出码：1\n\n标准输出：\n一切正常\n"

        self.assertEqual(tool._with_diagnostics(original), original)


class CompileToolMSVCDiagnosticsTest(CompileToolMSVCTestBase):
    """同一条链路，走 cl 的格式。

    这一组要钉的主要是：cl 把错误打在标准输出里，诊断照样要出来。
    """

    def test_an_error_from_stdout_is_parsed(self):
        # 只解析标准错误的话，MSVC 这条路一条诊断都出不来
        self.write("broken.cpp", BAD_SOURCE)

        result = self.compile_cl(source="broken.cpp")
        _, stdout, _ = parts(result)

        self.assertIn("error C2059", stdout)  # 原文确实在标准输出里
        self.assertIn("broken.cpp:3: error C2059:", result)  # 解析出来了

    def test_the_block_is_appended_after_the_output(self):
        self.write("broken.cpp", BAD_SOURCE)

        result = self.compile_cl(source="broken.cpp")

        self.assertGreater(result.index("诊断（"), result.index("标准错误："))

    def test_a_warning_from_a_successful_compile_is_parsed(self):
        self.write("warn.cpp", WARN_SOURCE)

        result = self.compile_cl(source="warn.cpp")

        self.assertIn("退出码：0", result)
        self.assertIn("warn.cpp:2: warning C4101:", result)


class CompileToolRealTest(unittest.TestCase):
    """真的叫一次 cl.exe。

    上面那些假 cl 验的是「命令拼得对不对」。这一条验的是另一件事：
    拼出来的命令 cl 认不认——假编译器不会因为你把 /std: 写成 /std- 就报错。

    这台机器上没装 Visual Studio 就跳过。
    """

    @classmethod
    def setUpClass(cls):
        vsenv.reset_cache()
        if vsenv.find_vcvars() is None:
            raise unittest.SkipTest("这台机器上没找到 Visual Studio")

        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name).resolve()

    def write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def test_a_real_program_compiles_and_runs(self):
        # 带中文、带 C++17 头文件。中文那部分是刻意的：cl 默认按系统代码页
        # 读源码，中文 Windows 上会直接把 UTF-8 的源码读歪
        self.write(
            "hello.cpp",
            "#include <iostream>\n"
            "#include <vector>\n"
            "#include <string>\n"
            "#include <filesystem>\n"
            "#include <optional>\n"
            "int main() {\n"
            '    std::vector<std::string> v = {"cl", "真的", "编过了"};\n'
            "    std::optional<int> n = 7;\n"
            "    for (const auto& s : v) std::cout << s << \" \";\n"
            "    std::cout << *n << std::endl;\n"
            "    return 0;\n"
            "}\n",
        )

        result = CompileTool(root=self.root).run(
            source="hello.cpp", flags="-std=c++17 -O2 -Wall"
        )

        self.assertIn("退出码：0", result)
        exe = self.root / "hello.exe"
        self.assertTrue(exe.is_file(), result)

        import subprocess

        run = subprocess.run([str(exe)], capture_output=True)
        self.assertEqual(run.returncode, 0)
        self.assertIn("cl 真的 编过了 7", run.stdout.decode("utf-8", "replace"))

    def test_a_real_error_comes_back_with_the_file_and_line(self):
        self.write("bad.cpp", "int main() {\n    int x = ;\n    return 0;\n}\n")

        result = CompileTool(root=self.root).run(source="bad.cpp")

        self.assertIn("退出码：2", result)
        self.assertIn("error C", result)
        # 真的 cl 带上 /diagnostics:column 之后报到列：`bad.cpp(2,13)`
        self.assertIn("bad.cpp(2,13)", result)

    def test_a_real_error_is_parsed_into_a_diagnostic(self):
        # 真跑一遍，确认这条路从 cl 的 stdout 一直通到结构化的那一行
        source_line = "    int x = ;"
        self.write("bad.cpp", f"int main() {{\n{source_line}\n    return 0;\n}}\n")

        result = CompileTool(root=self.root).run(source="bad.cpp")
        block = diagnostics.render(diagnostics.parse(result))

        self.assertIn("bad.cpp:2:13: error C2059:", block)
        # 报的那个列上真的是那个分号（列从 1 数起）
        self.assertEqual(source_line[13 - 1], ";")

    def test_the_compiler_is_found_without_being_on_the_path(self):
        # 关键的一条：cl 不在 PATH 上，工具也得自己找得到它。
        # 上面所有用例都在假装它找得到，这条是真的。
        found = find_compiler()

        self.assertIn(found, ("g++", "clang++", "cl"))


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


class CompileToolAgentTest(CompileToolTestBase):
    """走一遍完整的 Agent 调用链：模型要求编译 → 结果回灌。"""

    def build(self, replies):
        return Agent(
            client=StubClient(replies),
            tools=ToolRegistry([self.make_tool()]),
        )

    def test_agent_can_compile_a_file(self):
        self.write("main.cpp", GOOD_SOURCE)
        agent = self.build(
            [
                tool_call_reply("compile", json.dumps({"source": "main.cpp"})),
                final_reply("编译好了"),
            ]
        )

        answer = agent.run("编译 main.cpp")

        self.assertEqual(answer, "编译好了")
        self.assertIn("退出码：0", agent.messages[3]["content"])

    def test_agent_can_pass_flags(self):
        self.write("main.cpp", GOOD_SOURCE)
        agent = self.build(
            [
                tool_call_reply(
                    "compile",
                    json.dumps({"source": "main.cpp", "flags": "-std=c++11 -O2"}),
                ),
                final_reply("编译好了"),
            ]
        )

        agent.run("用 c++11 编译 main.cpp")

        self.assertIn(
            "-std=c++11 -O2 main.cpp -o main", agent.messages[3]["content"]
        )

    def test_agent_can_pick_a_compiler(self):
        self.write("main.cpp", GOOD_SOURCE)
        agent = self.build(
            [
                tool_call_reply(
                    "compile",
                    json.dumps({"source": "main.cpp", "compiler": "clang++"}),
                ),
                final_reply("好了"),
            ]
        )

        # 本机一个编译器都没装，先假装 clang++ 在
        with patch("tools.compile.shutil.which", return_value="/usr/bin/clang++"):
            agent.run("用 clang++ 编译")

        self.assertIn("clang++ main.cpp -o main", agent.messages[3]["content"])

    def test_bad_flags_do_not_break_the_turn(self):
        self.write("main.cpp", GOOD_SOURCE)
        agent = self.build(
            [
                tool_call_reply(
                    "compile",
                    json.dumps({"source": "main.cpp", "flags": "-O2; whoami"}),
                ),
                final_reply("参数有问题"),
            ]
        )

        answer = agent.run("带个怪参数编译")

        self.assertEqual(answer, "参数有问题")
        self.assertIn("编译参数里不能有", agent.messages[3]["content"])
        self.assertFalse(agent.get_trace()[0].success)

    def test_a_failed_compile_does_not_break_the_turn(self):
        self.write("broken.cpp", BAD_SOURCE)
        agent = self.build(
            [
                tool_call_reply("compile", json.dumps({"source": "broken.cpp"})),
                final_reply("编译报错了"),
            ]
        )

        answer = agent.run("编译 broken.cpp")

        self.assertEqual(answer, "编译报错了")
        self.assertIn("退出码：1", agent.messages[3]["content"])
        self.assertIn("error:", agent.messages[3]["content"])
        self.assertTrue(agent.get_trace()[0].success)

    def test_a_bad_path_does_not_break_the_turn(self):
        self.write("main.cpp", GOOD_SOURCE)
        agent = self.build(
            [
                tool_call_reply(
                    "compile",
                    json.dumps({"source": "main.cpp", "output": "../out"}),
                ),
                final_reply("换个名字"),
            ]
        )

        answer = agent.run("编译到项目外面去")

        self.assertEqual(answer, "换个名字")
        self.assertIn("超出允许范围", agent.messages[3]["content"])
        self.assertFalse(agent.get_trace()[0].success)

    def test_the_call_is_recorded_in_the_trace(self):
        self.write("main.cpp", GOOD_SOURCE)
        agent = self.build(
            [
                tool_call_reply("compile", json.dumps({"source": "main.cpp"})),
                final_reply("好了"),
            ]
        )

        agent.run("编译 main.cpp")

        events = agent.get_trace()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tool, "compile")
        self.assertTrue(events[0].success)

    def test_it_is_registered_in_main(self):
        # main.build_agent() 默认会去读 LLM 配置（API Key / 模型名）；把
        # 客户端整个换成替身，装配不需要任何真实配置，也不会联网
        import main

        with patch("main.LLMClient", StubClient):
            agent = main.build_agent()

        self.assertIn("compile", [tool.name for tool in agent.tools.list()])


if __name__ == "__main__":
    unittest.main()
