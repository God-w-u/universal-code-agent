"""Verifier 测试：改完自动验一次、和上次的诊断对一遍、验不了的时候也说明白。

这件事的价值全在「别让模型以为它验过了」，所以这里测的最多的不是「验成功长
什么样」，而是那些**不能报成通过**的情形：验证用的工具自己没跑起来、工程里
没有构建系统、改的是头文件、改的是 README、改的是 Python 却只有编译可用。

跑工具的假替身和别处不一样：假的 compile / build / run_tests 不真跑命令，
直接返回排好的文本。验证器干的活就是「跑一下、读结果、跟上次比」，
把结果换成排好的文本，测的正好是它自己那部分。

最后一条是**真的**（装了编译器才跑）：真编译一次、真改、自动重编通过。

全部离线，不联网。
"""

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent, Verifier, compare
from agent.verify import (
    BUILD,
    COMPILE,
    EDIT,
    RUN_TESTS,
    _build_system,
    _kind,
    _outcome,
    _raw,
)
from tools import CompileTool, EditFileTool, Tool, ToolExecutor, ToolRegistry
from tools import diagnostics
from tools.compile import find_compiler


class FakeVerifyTool(Tool):
    """假的 compile / build / run_tests：不真跑，按剧本返回结果文本。

    声明了 source / system / source_root / timeout 这几个参数，是为了让
    验证器挑出来的调用真的能走进来——参数名对不上，校验那一关就把它挡了。
    """

    description = "假的验证工具"
    parameters = {
        "source": {"type": "string", "required": False},
        "flags": {"type": "string", "required": False},
        "system": {"type": "string", "required": False},
        "source_root": {"type": "string", "required": False},
        "timeout": {"type": "number", "required": False},
    }

    def __init__(self, name, replies, root=None):
        self.name = name
        self._replies = list(replies)
        self.root = root
        self.calls = []  # 每次收到的参数

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        # 最后一条用完就一直重复，省得每条测试都要数着排
        return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


def ran(command, code=0, out="", err=""):
    """排一段 run_command 会返回的文本。

    头两行的样子是照着 tools/run_command._render 写的——验证器就是从那里
    读「过没过」的，自己另编一种写法，测的就不是真东西了。
    """
    head = f"命令：{command}\n退出码：{code}"
    if code != 0:
        head += "（非 0，命令失败）"
    return f"{head}\n\n标准输出：\n{out or '（空）'}\n\n标准错误：\n{err or '（空）'}"


def appended(text):
    """照 compile / build 的做法，在末尾附一段解析好的诊断。"""
    report = diagnostics.parse(text)
    return f"{text}\n\n{diagnostics.render(report)}"


C2059 = "main.cpp(2,18): error C2059: 语法错误:“;”"
GCC_ERROR = "a.cpp:2:18: error: expected ';' before '}'"


class VerifyTestBase(unittest.TestCase):
    """一个临时项目目录。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def write(self, name, text, encoding="utf-8"):
        # newline="" 是必须的：Windows 上文本模式会把 \n 翻成 \r\n
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding, newline="")
        return path

    def verifier(self, *tools):
        """拿着这几个假工具装一个验证器。"""
        registry = ToolRegistry(list(tools))
        return Verifier(registry, ToolExecutor(registry), root=self.root)

    def fake(self, name, replies):
        return FakeVerifyTool(name, replies, root=self.root)


class KindTest(unittest.TestCase):
    """按后缀分类：只有三类文件谈得上「编译 / 跑测试」。"""

    def test_c_and_cpp_sources(self):
        for name in ("a.c", "a.cc", "a.cpp", "a.cxx"):
            with self.subTest(name=name):
                self.assertEqual(_kind(name), "source")

    def test_headers(self):
        for name in ("a.h", "a.hh", "a.hpp", "a.hxx"):
            with self.subTest(name=name):
                self.assertEqual(_kind(name), "header")

    def test_python(self):
        self.assertEqual(_kind("src/a.py"), "python")

    def test_anything_else(self):
        for name in ("README.md", "notes.txt", "Makefile", "CMakeLists.txt"):
            with self.subTest(name=name):
                self.assertEqual(_kind(name), "other")

    def test_the_suffix_is_matched_case_insensitively(self):
        self.assertEqual(_kind("MAIN.CPP"), "source")
        self.assertEqual(_kind("A.H"), "header")


class BuildSystemTest(VerifyTestBase):
    """工程里有没有构建系统、是哪一套。"""

    def test_cmake(self):
        self.write("CMakeLists.txt", "project(x)\n")

        self.assertEqual(_build_system(self.root, "a.cpp"), ("cmake", "."))

    def test_make(self):
        self.write("Makefile", "all:\n")

        self.assertEqual(_build_system(self.root, "a.cpp"), ("make", "."))

    def test_gnu_makefile(self):
        self.write("GNUmakefile", "all:\n")

        self.assertEqual(_build_system(self.root, "a.cpp"), ("make", "."))

    def test_msbuild(self):
        self.write("app.sln", "Microsoft Visual Studio Solution File\n")

        self.assertEqual(_build_system(self.root, "a.cpp"), ("msbuild", "."))

    def test_a_vcxproj_alone_is_enough(self):
        self.write("app.vcxproj", "<Project />\n")

        self.assertEqual(_build_system(self.root, "a.cpp"), ("msbuild", "."))

    def test_cmake_wins_over_make(self):
        self.write("CMakeLists.txt", "project(x)\n")
        self.write("Makefile", "all:\n")

        self.assertEqual(_build_system(self.root, "a.cpp")[0], "cmake")

    def test_the_closest_one_wins(self):
        # 改的是 src/ 里的文件，src/ 里也有构建文件，就用它——
        # 工程根上那个是给别的子工程用的
        self.write("CMakeLists.txt", "project(big)\n")
        self.write("src/CMakeLists.txt", "project(small)\n")

        self.assertEqual(_build_system(self.root, "src/a.cpp"), ("cmake", "src"))

    def test_the_root_one_is_used_for_a_file_at_the_root(self):
        self.write("CMakeLists.txt", "project(x)\n")
        self.write("src/CMakeLists.txt", "project(y)\n")

        self.assertEqual(_build_system(self.root, "a.cpp"), ("cmake", "."))

    def test_nothing_found(self):
        self.write("a.cpp", "int main(){}\n")

        self.assertIsNone(_build_system(self.root, "a.cpp"))

    def test_it_does_not_look_above_the_root(self):
        # root 外面（临时目录的上一层）的构建文件不算：工具的活动范围就在
        # root 里，拿外面的构建系统去构建，等于跑到项目外面去动东西
        self.write("CMakeLists.txt", "project(外层的)\n")
        inner = self.root / "inner"
        inner.mkdir()
        self.write("inner/a.cpp", "int main(){}\n")

        self.assertIsNone(_build_system(inner, "a.cpp"))


class CompareTest(unittest.TestCase):
    """两批诊断怎么对。"""

    def report(self, *lines):
        return diagnostics.parse("\n".join(lines))

    def test_the_same_error_is_still_there(self):
        diff = compare(self.report(C2059), self.report(C2059))

        self.assertEqual(len(diff.still), 1)
        self.assertEqual(diff.gone, ())
        self.assertEqual(diff.added, ())

    def test_an_error_that_is_gone(self):
        diff = compare(self.report(C2059), self.report())

        self.assertEqual(len(diff.gone), 1)
        self.assertEqual(diff.still, ())
        self.assertEqual(diff.added, ())

    def test_a_brand_new_error(self):
        diff = compare(self.report(), self.report(C2059))

        self.assertEqual(len(diff.added), 1)
        self.assertEqual(diff.gone, ())
        self.assertEqual(diff.still, ())

    def test_a_moved_error_is_still_the_same_error(self):
        # 上面删掉一行，同一条错从第 5 行挪到第 4 行。要是按行列号比，
        # 就会说成「旧的没了、又来一条新的」——两句话都是假的
        before = self.report("a.cpp(5,3): error C2065: “x”: 未声明的标识符")
        after = self.report("a.cpp(4,3): error C2065: “x”: 未声明的标识符")

        diff = compare(before, after)

        self.assertEqual(len(diff.still), 1)
        self.assertEqual(diff.gone, ())
        self.assertEqual(diff.added, ())

    def test_a_changed_message_is_a_different_error(self):
        before = self.report("a.cpp(2,1): error C2065: “x”: 未声明的标识符")
        after = self.report("a.cpp(2,1): error C2065: “y”: 未声明的标识符")

        diff = compare(before, after)

        self.assertEqual(len(diff.gone), 1)
        self.assertEqual(len(diff.added), 1)

    def test_a_changed_code_is_a_different_error(self):
        before = self.report("a.cpp(2,1): error C2065: 未声明的标识符")
        after = self.report("a.cpp(2,1): error C2086: 未声明的标识符")

        diff = compare(before, after)

        self.assertEqual(len(diff.gone), 1)
        self.assertEqual(len(diff.added), 1)

    def test_a_warning_is_not_the_same_as_an_error(self):
        before = self.report("a.cpp(2,1): error C4101: 未引用的变量")
        after = self.report("a.cpp(2,1): warning C4101: 未引用的变量")

        diff = compare(before, after)

        self.assertEqual(len(diff.gone), 1)
        self.assertEqual(len(diff.added), 1)

    def test_another_file_is_a_different_error(self):
        before = self.report("a.cpp(2,1): error C2065: 未声明的标识符")
        after = self.report("b.cpp(2,1): error C2065: 未声明的标识符")

        diff = compare(before, after)

        self.assertEqual(len(diff.gone), 1)
        self.assertEqual(len(diff.added), 1)

    def test_the_same_error_twice_counts_once(self):
        # 同一条错在文件里出现两次（比如两个函数里各犯一次），
        # 改掉一个之后应当还剩一个，不能整批算没
        before = self.report(C2059, "main.cpp(9,1): " + C2059.split(": ", 1)[1])

        diff = compare(before, self.report(C2059))

        self.assertEqual(len(diff.still), 1)
        self.assertEqual(len(diff.gone), 1)

    def test_several_changes_at_once(self):
        before = self.report(C2059, "a.cpp(7,1): warning C4101: 未引用的变量 x")
        after = self.report("b.cpp(3,3): error C2065: “y”: 未声明的标识符")

        diff = compare(before, after)

        self.assertEqual(len(diff.gone), 2)
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(diff.still, ())


class OutcomeTest(unittest.TestCase):
    """过没过，是从 run_command 那段文本的头两行里读出来的。"""

    def test_a_zero_exit_code_passes(self):
        self.assertEqual(_outcome(ran("g++ a.cpp", 0)), (True, "通过（退出码 0）"))

    def test_a_non_zero_exit_code_fails(self):
        ok, text = _outcome(ran("g++ a.cpp", 2))

        self.assertFalse(ok)
        self.assertIn("退出码 2", text)

    def test_a_two_step_build_that_passed(self):
        text = ran("cmake -S . -B build") + "\n\n" + ran("cmake --build build")

        self.assertTrue(_outcome(text)[0])

    def test_a_two_step_build_whose_second_step_failed(self):
        # 只看第一个退出码的话，这里会报成「通过」——配置成功了、编译没过，
        # 那是这个阶段最不能出的一种错
        text = ran("cmake -S . -B build") + "\n\n" + ran("cmake --build build", 1)

        ok, verdict = _outcome(text)

        self.assertFalse(ok)
        self.assertIn("退出码 1", verdict)

    def test_a_timeout_fails(self):
        text = "命令：g++ a.cpp -o a\n结果：超时，已强制终止（超过 60 秒仍未结束）\n"

        ok, verdict = _outcome(text)

        self.assertFalse(ok)
        self.assertIn("超时", verdict)

    def test_no_exit_code_at_all_is_not_a_pass(self):
        # 认不出退出码就不敢说通过：命令找不到、编译器吐了一堆别的东西，
        # 都可能一条结构化诊断都不留，那种「通过」是假的
        ok, verdict = _outcome("sh: g++: command not found\n")

        self.assertFalse(ok)
        self.assertIn("没在输出里找到退出码", verdict)

    def test_a_stray_exit_code_in_the_output_is_not_counted(self):
        # 「退出码：」必须出现在 run_command 自己写的那个位置上，
        # 编译输出里正好有一行这样的文字不算
        text = ran("g++ a.cpp", 0, out="退出码：1")

        self.assertTrue(_outcome(text)[0])


class RawTest(unittest.TestCase):
    def test_the_appended_report_is_cut_off(self):
        text = appended(GCC_ERROR)

        self.assertEqual(_raw(text), GCC_ERROR)
        # 留着的话会解出两条一模一样的诊断
        self.assertEqual(len(diagnostics.parse(_raw(text)).diagnostics), 1)

    def test_text_without_a_report_is_untouched(self):
        self.assertEqual(_raw("随便一段文字"), "随便一段文字")


class CheckTest(VerifyTestBase):
    """改完之后跑哪个工具。"""

    def test_a_cpp_file_without_a_build_system_is_compiled(self):
        compile_tool = self.fake(COMPILE, [ran("g++ a.cpp", 0)])
        verifier = self.verifier(compile_tool)

        verifier.check("a.cpp")

        self.assertEqual(compile_tool.calls, [{"source": "a.cpp"}])

    def test_a_cpp_file_in_a_cmake_project_is_built(self):
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(BUILD, [ran("cmake --build build", 0)])
        verifier = self.verifier(build_tool, self.fake(COMPILE, [ran("x", 0)]))

        verifier.check("src/a.cpp")

        self.assertEqual(
            build_tool.calls, [{"system": "cmake", "source_root": "."}]
        )

    def test_a_python_file_runs_the_tests(self):
        test_tool = self.fake(RUN_TESTS, [ran("python -m unittest", 0)])
        verifier = self.verifier(test_tool)

        verifier.check("src/a.py")

        self.assertEqual(test_tool.calls, [{}])

    def test_the_same_compile_is_repeated_with_the_same_arguments(self):
        # 模型多半是先编译看到错才来改的。原样重跑那一条（连 flags 一起），
        # 前后才比得出来
        compile_tool = self.fake(COMPILE, [ran("g++ -Wall a.cpp", 2)])
        verifier = self.verifier(compile_tool)
        verifier.note(COMPILE, {"source": "a.cpp", "flags": "-Wall"}, ran("g++ -Wall a.cpp", 2))

        verifier.check("a.cpp")

        self.assertEqual(
            compile_tool.calls, [{"source": "a.cpp", "flags": "-Wall"}]
        )

    def test_last_time_it_compiled_another_file(self):
        # 上次编的是别的文件，不能拿它顶：那几个 flags 和那个文件
        # 跟这次改的不是一回事，改走规则重新挑
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(BUILD, [ran("cmake --build build", 0)])
        verifier = self.verifier(build_tool)
        verifier.note(COMPILE, {"source": "other.cpp"}, ran("g++ other.cpp", 0))

        verifier.check("a.cpp")

        self.assertEqual(len(build_tool.calls), 1)

    def test_a_build_covers_a_header_change(self):
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(BUILD, [ran("cmake --build build", 0)])
        verifier = self.verifier(build_tool)
        verifier.note(BUILD, {"system": "cmake"}, ran("cmake --build build", 0))

        verifier.check("src/a.h")

        self.assertEqual(build_tool.calls, [{"system": "cmake"}])

    def test_a_test_run_does_not_cover_a_cpp_change(self):
        # 改了个 .cpp 却只拿「测试过没过」当验证，等于没验——
        # 那是最糟的一种：假通过。这里必须改走编译
        compile_tool = self.fake(COMPILE, [ran("g++ a.cpp", 0)])
        test_tool = self.fake(RUN_TESTS, [ran("python -m unittest", 0)])
        verifier = self.verifier(compile_tool, test_tool)
        verifier.note(RUN_TESTS, {}, ran("python -m unittest", 0))

        verifier.check("a.cpp")

        self.assertEqual(test_tool.calls, [])
        self.assertEqual(compile_tool.calls, [{"source": "a.cpp"}])

    def test_a_header_without_a_build_system_is_not_compiled_on_its_own(self):
        # 头文件单独编出来的是「没有 main」这类链接错误，不是真问题
        compile_tool = self.fake(COMPILE, [ran("g++ a.h", 1)])
        verifier = self.verifier(compile_tool)

        text = verifier.check("a.h")

        self.assertEqual(compile_tool.calls, [])
        self.assertIn("头文件", text)

    def test_the_header_message_points_at_the_source_that_was_compiled(self):
        compile_tool = self.fake(COMPILE, [ran("g++ a.cpp", 0)])
        verifier = self.verifier(compile_tool)
        verifier.note(COMPILE, {"source": "a.cpp"}, ran("g++ a.cpp", 0))

        self.assertIn("a.cpp", verifier.check("a.h"))

    def test_nothing_to_run_for_a_readme(self):
        text = self.verifier(self.fake(COMPILE, [ran("x", 0)])).check("README.md")

        self.assertIn("没有验证过", text)

    def test_a_missing_tool_is_said_out_loud(self):
        text = self.verifier().check("a.py")

        self.assertIn(RUN_TESTS, text)
        self.assertIn("没有验证过", text)

    def test_an_empty_path_is_ignored(self):
        verifier = self.verifier(self.fake(COMPILE, [ran("x", 0)]))

        self.assertEqual(verifier.check(None), "")
        self.assertEqual(verifier.check("   "), "")

    def test_a_verifier_that_cannot_run_is_reported_as_such(self):
        # 工具自己没跑起来（没装编译器……）：得说清楚是「验不了」，
        # 不是「改错了」——不然模型会去改本来是对的代码
        class Broken(FakeVerifyTool):
            def execute(self, **kwargs):
                raise RuntimeError("这台机器上没有 C++ 编译器")

        verifier = self.verifier(Broken(COMPILE, [""], root=self.root))

        text = verifier.check("a.cpp")

        self.assertIn("没跑成", text)
        self.assertIn("没有 C++ 编译器", text)

    def test_the_result_of_a_check_becomes_the_next_baseline(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ a.cpp", 2, err=C2059),
                ran("g++ a.cpp", 0),
            ],
        )
        verifier = self.verifier(compile_tool)

        first = verifier.check("a.cpp")
        second = verifier.check("a.cpp")

        # 第一次没有基线，第二次跟第一次比：那条错算「已消失」
        self.assertIn("没有可比较的老诊断", first)
        self.assertIn("已消失", second)


class RenderTest(VerifyTestBase):
    """给模型看的那段话。"""

    def test_a_pass_says_so(self):
        compile_tool = self.fake(COMPILE, [ran("g++ a.cpp", 0)])
        verifier = self.verifier(compile_tool)
        verifier.note(COMPILE, {"source": "a.cpp"}, ran("g++ a.cpp", 2, err=C2059))

        text = verifier.check("a.cpp")

        self.assertIn("验证（compile a.cpp）：通过（退出码 0）", text)
        self.assertIn("之前 1 个 error；现在没有 error 或 warning", text)
        self.assertIn("已消失", text)
        self.assertIn("C2059", text)

    def test_a_failure_lists_the_new_diagnostics(self):
        compile_tool = self.fake(
            COMPILE, [ran("g++ a.cpp", 2, err="a.cpp:4:9: error: 'x' was not declared")]
        )
        verifier = self.verifier(compile_tool)
        verifier.note(COMPILE, {"source": "a.cpp"}, ran("g++ a.cpp", 2, err=GCC_ERROR))

        text = verifier.check("a.cpp")

        self.assertIn("没通过（退出码 2）", text)
        self.assertIn("已消失", text)  # 原来那条
        self.assertIn("新增", text)  # 新冒出来的

    def test_an_unparsable_failure_shows_the_raw_output(self):
        # msbuild 的「项目文件不存在」没有行列号，进不了结构化诊断。
        # 只回一句「没通过」，模型就没法接着查了
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(
            BUILD, [ran("msbuild app.sln", 1, out="MSB1009: 项目文件不存在。")]
        )
        verifier = self.verifier(build_tool)

        text = verifier.check("a.cpp")

        self.assertIn("MSB1009", text)

    def test_the_raw_output_is_clipped(self):
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(BUILD, [ran("msbuild app.sln", 1, out="废话" * 2000)])
        verifier = self.verifier(build_tool)

        text = verifier.check("a.cpp")

        self.assertIn("只给了开头", text)
        self.assertLess(len(text), 2000)

    def test_the_appended_report_is_not_counted_twice(self):
        # compile 的输出末尾已经附了一段解析好的诊断，验证器必须认出那是
        # 同一批诊断的另一种写法。再解一遍会凭空多出一条一模一样的，
        # 「已消失 / 仍在 / 新增」就全乱了
        compile_tool = self.fake(
            COMPILE, [appended(ran("g++ a.cpp", 2, err=GCC_ERROR))]
        )
        verifier = self.verifier(compile_tool)

        text = verifier.check("a.cpp")

        self.assertEqual(text.count("expected ';' before '}'"), 1)

    def test_a_first_check_has_nothing_to_compare_against(self):
        compile_tool = self.fake(COMPILE, [ran("g++ a.cpp", 0)])
        verifier = self.verifier(compile_tool)

        self.assertIn("没有可比较的老诊断", verifier.check("a.cpp"))

    def test_the_verification_names_the_build_system(self):
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(BUILD, [ran("cmake --build build", 0)])
        verifier = self.verifier(build_tool)

        self.assertIn("验证（build cmake）", verifier.check("a.cpp"))


class StubClient:
    """按剧本回话的假客户端。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


def tool_call_reply(name, arguments, call_id="c1"):
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


def final_reply(content="好的"):
    return {"role": "assistant", "content": content}


class AgentVerifyTest(VerifyTestBase):
    """走一遍完整的 Agent：改完 -> 自动验证 -> 结果回灌。"""

    def setUp(self):
        super().setUp()
        self.write("main.cpp", "int main() {\n    int answer = 42\n}\n")

    def build_agent(self, replies, *tools):
        return Agent(
            client=StubClient(replies),
            tools=ToolRegistry(
                [EditFileTool(root=self.root), *tools]
            ),
        )

    def edit(self, **kwargs):
        return tool_call_reply(
            EDIT,
            json.dumps({"path": "main.cpp", "new_text": "    int answer = 42;", **kwargs}),
        )

    def test_a_pass_is_appended_to_the_edit_result(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:1: error: expected ';'"),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(old_text="    int answer = 42"),
                final_reply("修好了"),
            ],
            compile_tool,
        )

        answer = agent.run("编译不过，改一下")

        # 修复过的回答前面多一句摘要（7.6），模型自己的话原样跟在后面
        self.assertTrue(answer.startswith("修复 1 轮；第 1 轮编译通过；最终成功。\n\n"))
        self.assertTrue(answer.endswith("修好了"))
        edit_message = agent.messages[5]["content"]
        self.assertIn("结果：成功", edit_message)
        self.assertIn("验证（compile main.cpp）：通过（退出码 0）", edit_message)
        self.assertIn("已消失", edit_message)

    def test_a_still_failing_compile_is_reported(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:1: error: expected ';'"),
                ran("g++ main.cpp", 2, err="main.cpp:3:9: error: 'x' was not declared"),
            ],
        )
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(start_line=2),
                final_reply("还有别的错"),
                # 验证没过就想收尾，会被自动修复催一句（7.5）；这是被催之后的
                # 回话——又只回话没动手，于是自动修复停下并报状态
                final_reply("我改不动"),
            ],
            compile_tool,
        )

        agent.run("编译不过，改一下")

        edit_message = agent.messages[5]["content"]
        self.assertIn("没通过（退出码 2）", edit_message)
        self.assertIn("新增", edit_message)

    def test_a_failing_build_carries_its_diagnostics(self):
        self.write("CMakeLists.txt", "project(x)\n")
        build_tool = self.fake(
            BUILD,
            [
                ran("cmake --build build", 1, out="main.cpp(2,18): error C2059: 语法错误:“;”"),
            ],
        )
        agent = self.build_agent(
            [
                self.edit(old_text="    int answer = 42"),
                final_reply("看看结果"),
                # 同上：构建没过就收尾会被催一句，这是被催之后的回话
                final_reply("改不动"),
            ],
            build_tool,
        )

        agent.run("改一下")

        edit_message = agent.messages[3]["content"]
        self.assertIn("验证（build cmake）：没通过（退出码 1）", edit_message)
        self.assertIn("C2059", edit_message)

    def test_the_verification_shows_up_in_the_trace(self):
        compile_tool = self.fake(COMPILE, [ran("g++ main.cpp", 0)])
        agent = self.build_agent(
            [self.edit(old_text="    int answer = 42"), final_reply("好了")],
            compile_tool,
        )

        agent.run("改一下")

        events = agent.get_trace()
        self.assertEqual([event.tool for event in events], [EDIT, COMPILE])
        self.assertTrue(all(event.success for event in events))
        self.assertEqual(events[1].arguments, {"source": "main.cpp"})

    def test_the_verification_runs_once_per_edit(self):
        # 一次编辑只触发一次验证：验不过也不会自己再改（再改是模型下一轮的事，
        # 由 RepairLoop 盯着，见 tests/test_repair.py）
        compile_tool = self.fake(COMPILE, [ran("g++ main.cpp", 2)])
        agent = self.build_agent(
            [
                self.edit(old_text="    int answer = 42"),
                final_reply("好"),
                # 验证没过就收尾，模型会被催一句；这是被催之后的回话
                final_reply("好，我知道了"),
            ],
            compile_tool,
        )

        agent.run("改一下")

        self.assertEqual(len(compile_tool.calls), 1)
        self.assertEqual(len(agent.get_trace()), 2)

    def test_a_refused_edit_is_not_verified(self):
        compile_tool = self.fake(COMPILE, [ran("g++ main.cpp", 0)])
        agent = self.build_agent(
            [
                self.edit(old_text="这段原文根本不在文件里"),
                final_reply("没改成"),
            ],
            compile_tool,
        )

        agent.run("改一下")

        self.assertEqual(compile_tool.calls, [])
        self.assertIn("找不到这段原文", agent.messages[3]["content"])

    def test_a_first_edit_without_any_previous_compile_says_so(self):
        compile_tool = self.fake(COMPILE, [ran("g++ main.cpp", 0)])
        agent = self.build_agent(
            [self.edit(old_text="    int answer = 42"), final_reply("好")],
            compile_tool,
        )

        agent.run("改一下")

        self.assertIn("没有可比较的老诊断", agent.messages[3]["content"])

    def test_an_agent_without_verification_tools_says_so_and_stays_quiet(self):
        # 只装了 edit_file 的 Agent：验不了，那就说明白，但别去调不存在的工具
        agent = Agent(
            client=StubClient(
                [self.edit(old_text="    int answer = 42"), final_reply("好")]
            ),
            tools=ToolRegistry([EditFileTool(root=self.root)]),
        )

        agent.run("改一下")

        self.assertIn("没有验证过", agent.messages[3]["content"])
        self.assertEqual([event.tool for event in agent.get_trace()], [EDIT])

    def test_a_broken_verification_does_not_break_the_turn(self):
        class Broken(FakeVerifyTool):
            def execute(self, **kwargs):
                raise RuntimeError("这台机器上没有 C++ 编译器")

        agent = self.build_agent(
            [self.edit(old_text="    int answer = 42"), final_reply("知道了")],
            Broken(COMPILE, [""], root=self.root),
        )

        answer = agent.run("改一下")

        self.assertEqual(answer, "知道了")
        self.assertIn("没跑成", agent.messages[3]["content"])
        # 改是改成什么样就是什么样，不自动回滚
        self.assertIn("int answer = 42;", (self.root / "main.cpp").read_text("utf-8"))


class RealVerifyTest(VerifyTestBase):
    """真的编译一遍：编不过 -> 改 -> 自动重编，退出码 0。

    这台机器上没装编译器就跳过：假编译器验不了「真的编译器怎么报错」。
    """

    BAD = (
        "#include <iostream>\n"
        "int main() {\n"
        "    int answer = 42\n"  # 少了分号
        "    std::cout << answer << std::endl;\n"
        "    return 0;\n"
        "}\n"
    )

    def setUp(self):
        super().setUp()
        if find_compiler() is None:
            raise unittest.SkipTest("这台机器上没有 C++ 编译器")
        self.write("main.cpp", self.BAD)

    def test_a_real_compile_failure_is_fixed_and_verified(self):
        compile_tool = CompileTool(root=self.root)
        agent = Agent(
            client=StubClient(
                [
                    tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                    tool_call_reply(
                        EDIT,
                        json.dumps(
                            {
                                "path": "main.cpp",
                                "old_text": "    int answer = 42",
                                "new_text": "    int answer = 42;",
                            }
                        ),
                    ),
                    final_reply("修好了"),
                ]
            ),
            tools=ToolRegistry([compile_tool, EditFileTool(root=self.root)]),
        )

        answer = agent.run("main.cpp 编译不过，改一下")
        self.assertTrue(answer.endswith("修好了"))  # 摘要之后还是模型自己的话

        # 第一次编译：真的报了错，而且诊断能指到文件上
        first = agent.messages[3]["content"]
        self.assertIn("退出码", first)
        report = diagnostics.parse(first)
        self.assertTrue(report.has_problems)
        self.assertEqual(report.diagnostics[0].file, "main.cpp")

        # 改动之后自动重编：这次过了
        edit_message = agent.messages[5]["content"]
        self.assertIn("验证（compile main.cpp）：通过（退出码 0）", edit_message)

        # 再自己跑一遍确认：真的编得过
        final = compile_tool.run(source="main.cpp")
        self.assertIn("退出码：0", final)
        self.assertIn("int answer = 42;", (self.root / "main.cpp").read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
