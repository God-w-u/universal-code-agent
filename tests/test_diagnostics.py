"""tools/diagnostics.py 测试：把编译器的输出读成结构化诊断。

这里钉的是「认不认得出来、认得对不对」，不涉及真实编译——
真编译在 tests/test_compile.py 的 CompileToolRealTest 里。

样本都是照着三个编译器真实的输出抄的：MSVC 和 GCC 那几张是这台机器上
跑出来的原文，连出错的那个源文件都一起搬来了（下面的 SOURCE），
所以行号能不能对上 read_file 是真的验过，不是凑出来的。
Clang 这台机器上没装编译器（只有 clang-format），用的是它公开文档里的格式。

全部离线，不联网。
"""

import re
import tempfile
import unittest
from pathlib import Path

from tools import diagnostics
from tools.diagnostics import Report, parse
from tools.read_file import ReadFileTool
from tools.textfile import split_lines

# --- 真实样本 ---------------------------------------------------------------

# MSVC 那几张来自这台机器上
# `cl /nologo /EHsc /utf-8 /W4 /diagnostics:column m2.cpp /Fe:m2` 的原文。
# 注意它打在标准输出里，不在标准错误里
MSVC_UNDECLARED = 'm2.cpp(3,3): error C2065: “undeclared_thing”: 未声明的标识符'
MSVC_SYNTAX = 'm2.cpp(4,11): error C2059: 语法错误:“;”'
MSVC_WARNING = 'm2.cpp(2,7): warning C4101: “unused”: 未引用的局部变量'
# 不带 /diagnostics:column 时 cl 只给行号，不给列
MSVC_NO_COLUMN = 'm2.cpp(3): error C2065: “undeclared_thing”: 未声明的标识符'

MSVC_FATAL = (
    "msvc.cpp(1): fatal error C1083: 无法打开包括文件: “nope.h”: "
    "No such file or directory"
)

# 上面那条 cl 命令编的就是这个文件，一个字符没改
M2_CPP = (
    "int main(){\n"  # 1
    "  int unused;\n"  # 2
    "  undeclared_thing = 1;\n"  # 3
    "  int x = ;\n"  # 4
    "  return 0;\n"  # 5
    "}\n"  # 6
)

# GCC：这台机器上 `g++ -c bad.cpp` 的原文（MinGW GCC 7.3.0）
GCC_ERROR = "bad.cpp:2:11: error: expected primary-expression before ';' token"
GCC_WARNING = "warn.cpp:2:7: warning: unused variable 'unused' [-Wunused-variable]"

# Clang：格式和 GCC 同源，差异在措辞和 -W 名字上
CLANG_ERROR = "hello.cpp:3:5: error: use of undeclared identifier 'x'"
CLANG_WARNING = (
    "hello.cpp:7:1: warning: control reaches end of non-void function "
    "[-Wreturn-type]"
)


class MSVCTest(unittest.TestCase):
    def test_an_error_is_parsed(self):
        item = parse(MSVC_UNDECLARED).diagnostics[0]

        self.assertEqual(item.file, "m2.cpp")
        self.assertEqual(item.line, 3)
        self.assertEqual(item.level, "error")
        self.assertEqual(item.code, "C2065")

    def test_the_message_is_kept_whole(self):
        # 消息里本来就有冒号（MSVC 的中文消息是「“x”: 未声明的标识符」），
        # 切字段只能切到第一个，后面整段都算消息
        item = parse(MSVC_UNDECLARED).diagnostics[0]

        self.assertEqual(item.message, "“undeclared_thing”: 未声明的标识符")

    def test_a_column_is_parsed(self):
        item = parse(MSVC_SYNTAX).diagnostics[0]

        self.assertEqual(item.line, 4)
        self.assertEqual(item.column, 11)

    def test_a_line_without_a_column_has_no_column(self):
        # 默认 cl 只报行号。这时候 column 是 None 而不是 0——
        # 0 是个真实的列号，混起来模型会去第 0 列找
        item = parse(MSVC_NO_COLUMN).diagnostics[0]

        self.assertEqual(item.line, 3)
        self.assertIsNone(item.column)

    def test_a_warning_is_not_an_error(self):
        item = parse(MSVC_WARNING).diagnostics[0]

        self.assertEqual(item.level, "warning")
        self.assertEqual(item.code, "C4101")

    def test_a_fatal_error_counts_as_an_error(self):
        # fatal 和普通 error 的区别（编译当场停下）在错误码里写着（C1083），
        # 级别上不再分一层
        item = parse(MSVC_FATAL).diagnostics[0]

        self.assertEqual(item.level, "error")
        self.assertEqual(item.code, "C1083")

    def test_a_drive_letter_survives(self):
        item = parse(
            r"c:\build\src\main.cpp(3,3): error C2065: “x”: 未声明的标识符"
        ).diagnostics[0]

        self.assertEqual(item.file, "c:/build/src/main.cpp")
        self.assertEqual(item.code, "C2065")

    def test_a_linker_error_without_a_location_is_not_a_diagnostic(self):
        # 链接错误长这样：`main.obj : error LNK2019: 无法解析的外部符号`——
        # 没有行号。诊断是有位置的东西，没位置的这条不进 diagnostics，
        # 但它在 other 里原样躺着，模型照样看得到
        report = parse("main.obj : error LNK2019: 无法解析的外部符号 _main\n")

        self.assertEqual(report.diagnostics, ())
        self.assertEqual(len(report.other), 1)


class GccClangTest(unittest.TestCase):
    def test_a_gcc_error_is_parsed(self):
        item = parse(GCC_ERROR).diagnostics[0]

        self.assertEqual(item.file, "bad.cpp")
        self.assertEqual(item.line, 2)
        self.assertEqual(item.column, 11)
        self.assertEqual(item.level, "error")
        self.assertEqual(
            item.message, "expected primary-expression before ';' token"
        )

    def test_a_gcc_warning_keeps_its_switch_as_the_code(self):
        # [-Wunused-variable] 是「哪个开关报的」，抽出来当 code，
        # 消息里就只剩人话
        item = parse(GCC_WARNING).diagnostics[0]

        self.assertEqual(item.level, "warning")
        self.assertEqual(item.code, "-Wunused-variable")
        self.assertEqual(item.message, "unused variable 'unused'")

    def test_a_clang_error_is_parsed(self):
        item = parse(CLANG_ERROR).diagnostics[0]

        self.assertEqual(item.file, "hello.cpp")
        self.assertEqual(item.line, 3)
        self.assertEqual(item.column, 5)
        self.assertEqual(item.level, "error")
        self.assertEqual(item.message, "use of undeclared identifier 'x'")

    def test_a_clang_warning_is_parsed(self):
        item = parse(CLANG_WARNING).diagnostics[0]

        self.assertEqual(item.level, "warning")
        self.assertEqual(item.code, "-Wreturn-type")

    def test_a_line_without_a_column_is_parsed(self):
        item = parse("main.cpp:9: error: 'x' was not declared").diagnostics[0]

        self.assertEqual(item.line, 9)
        self.assertIsNone(item.column)

    def test_a_bracket_that_is_not_a_switch_is_not_taken_as_a_code(self):
        # 模板诊断末尾挂着 [with T = int]，那是参数说明不是开关。
        # 当成 code 抽走的话，消息就断在半句话上
        item = parse("a.cpp:1:1: error: no match for call [with T = int]").diagnostics[0]

        self.assertIsNone(item.code)
        self.assertEqual(item.message, "no match for call [with T = int]")

    def test_a_note_is_kept(self):
        item = parse("a.cpp:1:5: note: 'x' declared here").diagnostics[0]

        self.assertEqual(item.level, "note")

    def test_a_gcc_fatal_error_counts_as_an_error(self):
        # GCC 也用它：`fatal error: iostream: No such file or directory`
        item = parse(
            "a.cpp:1:10: fatal error: iostream: No such file or directory"
        ).diagnostics[0]

        self.assertEqual(item.level, "error")
        self.assertEqual(item.line, 1)


class PathTest(unittest.TestCase):
    def test_a_relative_path_stays_as_written(self):
        # 命令里的源文件本来就是相对 root 的，编译器原样回显——
        # 所以这个字段直接就是 read_file 想要的那个路径
        item = parse("src/app/main.cpp:3:1: error: boom").diagnostics[0]

        self.assertEqual(item.file, "src/app/main.cpp")

    def test_a_windows_absolute_path_keeps_its_drive_letter(self):
        # 盘符后面也有一个冒号，切错了就会变成一个叫 "C" 的文件
        item = parse(r"C:\project\src\main.cpp:12:5: error: boom").diagnostics[0]

        self.assertEqual(item.file, "C:/project/src/main.cpp")
        self.assertEqual(item.line, 12)
        self.assertEqual(item.column, 5)

    def test_backslashes_become_slashes(self):
        # 和 list_files / search_code 返回的路径一个写法，模型不用先判断
        # 「这条路是哪个工具给的」才知道要不要转斜杠
        item = parse(r"src\app\main.cpp:3:1: error: boom").diagnostics[0]

        self.assertEqual(item.file, "src/app/main.cpp")

    def test_a_windows_path_without_a_column_is_parsed(self):
        item = parse(r"D:\a\b\main.cpp:7: error: boom").diagnostics[0]

        self.assertEqual(item.file, "D:/a/b/main.cpp")
        self.assertEqual(item.line, 7)

    def test_a_path_that_looks_like_a_drive_is_not_one(self):
        # `src:/a.cpp` 这种少见但合法：冒号前面是一个字母，看着像盘符。
        # 它仍然要能被认出来，只是把 `src:` 当成盘符的一部分
        item = parse("a:b.cpp:5:1: error: boom").diagnostics[0]

        self.assertEqual(item.line, 5)
        self.assertEqual(item.column, 1)

    def test_the_same_line_parses_the_same_under_any_prefix(self):
        # 解析器不认识「项目」这个概念：同一行诊断换个目录前缀，
        # 除了路径本身，别的字段一个都不能变。这条挡的是
        # 「往解析器里塞项目路径」那种写法
        results = [
            parse(f"{prefix}:12:5: error: boom").diagnostics[0]
            for prefix in (
                "main.cpp",
                "src/main.cpp",
                "/home/someone/project/main.cpp",
                r"C:\somewhere\else\main.cpp",
            )
        ]

        self.assertEqual(
            [
                (item.line, item.column, item.level, item.message)
                for item in results
            ],
            [(12, 5, "error", "boom")] * 4,
        )


class ManyTest(unittest.TestCase):
    # 这台机器上 `cl /W4 /diagnostics:column m2.cpp` 的原文，照抄
    OUTPUT = (
        "m2.cpp\n"
        f"{MSVC_UNDECLARED}\n"
        f"{MSVC_SYNTAX}\n"
        f"{MSVC_WARNING}\n"
    )

    def test_every_diagnostic_is_found(self):
        self.assertEqual(len(parse(self.OUTPUT).diagnostics), 3)

    def test_they_stay_in_the_order_the_compiler_printed_them(self):
        # 不按级别重排：note 跟在它解释的那条 error 后面，
        # 重排会把这条线索扯断
        lines = [item.line for item in parse(self.OUTPUT).diagnostics]

        self.assertEqual(lines, [3, 4, 2])

    def test_each_one_keeps_its_own_level(self):
        levels = [item.level for item in parse(self.OUTPUT).diagnostics]

        self.assertEqual(levels, ["error", "error", "warning"])

    def test_each_one_keeps_its_own_code(self):
        codes = [item.code for item in parse(self.OUTPUT).diagnostics]

        self.assertEqual(codes, ["C2065", "C2059", "C4101"])

    def test_has_problems_is_true(self):
        self.assertTrue(parse(self.OUTPUT).has_problems)

    def test_mixed_compilers_in_one_run_are_both_parsed(self):
        # make 串起好几个工具链时，一次输出里混着两种格式是正常的
        report = parse(f"{GCC_ERROR}\n{MSVC_UNDECLARED}\n")

        self.assertEqual(
            [item.file for item in report.diagnostics], ["bad.cpp", "m2.cpp"]
        )


class OtherTest(unittest.TestCase):
    def test_unparsed_lines_are_kept_verbatim(self):
        # 编译器还会打源文件上下文、In file included from、命令行回显。
        # 它们有用，只是不是诊断——不能因为认不出来就丢掉
        text = (
            "bad.cpp: In function 'int main()':\n"
            f"{GCC_ERROR}\n"
            "   int x = ;\n"
            "           ^\n"
        )

        report = parse(text)

        self.assertEqual(len(report.diagnostics), 1)
        self.assertEqual(
            report.other,
            ("bad.cpp: In function 'int main()':", "   int x = ;", "           ^"),
        )

    def test_the_caret_line_is_not_mistaken_for_a_diagnostic(self):
        report = parse("           ^\n")

        self.assertEqual(report.diagnostics, ())
        self.assertEqual(report.other, ("           ^",))

    def test_output_with_nothing_recognizable_is_all_other(self):
        text = "正在生成代码...\n已完成生成项目“demo.vcxproj”的操作。\n"

        report = parse(text)

        self.assertEqual(report.diagnostics, ())
        self.assertFalse(report.has_problems)
        self.assertEqual(len(report.other), 2)

    def test_an_empty_text_gives_an_empty_report(self):
        self.assertEqual(parse(""), Report())

    def test_the_run_command_header_is_not_a_diagnostic(self):
        # 解析的是 run_command 渲染过的那段文本，开头的「命令：/退出码：」
        # 会一起进来。它们不能被当成诊断——注意「（非 0，命令失败）」
        # 里那个 0 是全角括号包着的，不是 MSVC 那种 (12)
        report = parse("命令：cl /nologo m.cpp\n退出码：2（非 0，命令失败）\n")

        self.assertEqual(report.diagnostics, ())

    def test_crlf_output_is_handled(self):
        # cl 打在 Windows 控制台上的是 CRLF
        item = parse(f"{MSVC_UNDECLARED}\r\n").diagnostics[0]

        self.assertEqual(item.line, 3)
        self.assertEqual(item.message, "“undeclared_thing”: 未声明的标识符")

    def test_the_only_thing_lost_is_which_stream_it_came_from(self):
        # 解析器丢了「这条是 stdout 还是 stderr」这一点，
        # 因为对模型没差别：cl 打在 stdout、gcc 打在 stderr，
        # 而 run_command 两路都收，原文也还在
        report = parse(f"标准输出：\n{MSVC_UNDECLARED}\n\n标准错误：\n（空）\n")

        self.assertEqual(len(report.diagnostics), 1)


class RenderTest(unittest.TestCase):
    def test_it_renders_every_diagnostic(self):
        text = diagnostics.render(parse(f"{MSVC_SYNTAX}\n{MSVC_WARNING}\n"))

        self.assertIn("m2.cpp:4:11: error C2059: 语法错误:“;”", text)
        self.assertIn("m2.cpp:2:7: warning C4101: “unused”: 未引用的局部变量", text)

    def test_it_always_renders_gcc_style(self):
        # 不管原来是哪家报的，模型看到的形状是一样的——
        # 而且行号就挂在前缀里，可以整行拿去 read_file 核对
        text = diagnostics.render(parse(MSVC_UNDECLARED))

        self.assertIn("m2.cpp:3:3: error C2065:", text)

    def test_it_counts_the_levels(self):
        text = diagnostics.render(
            parse(f"{MSVC_UNDECLARED}\n{MSVC_WARNING}\n{GCC_ERROR}\n")
        )

        self.assertIn("共 3 条", text)
        self.assertIn("2 个 error", text)
        self.assertIn("1 个 warning", text)

    def test_an_unrecognized_output_renders_as_nothing(self):
        self.assertEqual(diagnostics.render(parse("一切正常\n")), "")

    def test_a_diagnostic_without_a_column_renders_without_one(self):
        text = diagnostics.render(parse(MSVC_NO_COLUMN))

        self.assertIn("m2.cpp:3: error C2065:", text)
        self.assertNotIn("m2.cpp:3:0", text)

    def test_notes_are_shown_when_there_is_a_real_problem(self):
        # note 单独出现时不值得给模型看，但它跟着 error 一起来时是要看的：
        # 它说的是「这个标识符是在那儿声明的」
        report = parse(f"{GCC_ERROR}\nbad.cpp:9:5: note: 'x' declared here\n")

        self.assertTrue(report.has_problems)
        self.assertIn("declared here", diagnostics.render(report))

    def test_notes_alone_are_not_a_problem(self):
        report = parse("bad.cpp:9:5: note: 'x' declared here\n")

        self.assertFalse(report.has_problems)


class AgreesWithReadFileTest(unittest.TestCase):
    """解析出来的行号必须能直接喂给 read_file。

    这条是最要紧的一条：编译器说第 3 行有问题，模型拿 3 去 read_file，
    读到的就必须是出问题的那一行。两边只要有一边数错了（差一、
    或者把 \\x0c 也当换行），模型就会对着相邻的一行改代码，
    还查不出为什么没用。

    这里用的 MSVC 输出是这台机器上真跑出来的，M2_CPP 就是当时编的那个文件。
    """

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "m2.cpp").write_text(M2_CPP, encoding="utf-8")

    def read_line(self, number: int) -> str:
        """read_file 眼里的第 number 行。"""
        text = ReadFileTool(root=self.root).run(path="m2.cpp")
        for line in split_lines(text):
            # read_file 会把行号右对齐，行数多的时候前面补空格
            hit = re.match(rf"\s*{number}\|(.*)$", line)
            if hit:
                return hit.group(1)
        raise AssertionError(f"read_file 没有给出第 {number} 行：\n{text}")

    def test_a_msvc_error_points_at_the_line_it_says(self):
        item = parse(MSVC_UNDECLARED).diagnostics[0]

        self.assertIn("undeclared_thing", self.read_line(item.line))

    def test_another_msvc_error_points_at_its_own_line(self):
        item = parse(MSVC_SYNTAX).diagnostics[0]

        self.assertIn("int x = ;", self.read_line(item.line))

    def test_an_msvc_warning_points_at_its_own_line(self):
        item = parse(MSVC_WARNING).diagnostics[0]

        self.assertIn("int unused;", self.read_line(item.line))

    def test_the_last_line_is_readable(self):
        # 差一错误最常在这里露出来：末尾那个换行会不会被算成一行
        item = parse("m2.cpp:6:1: error: boom").diagnostics[0]

        self.assertEqual(self.read_line(item.line).strip(), "}")

    def test_the_first_line_is_readable(self):
        item = parse("m2.cpp:1:1: error: boom").diagnostics[0]

        self.assertEqual(self.read_line(item.line), "int main(){")

    def test_a_column_lands_on_the_character_the_compiler_meant(self):
        # 列号是 1 起算的字符位置。第 4 行是 `  int x = ;`，第 11 个字符
        # 正好是那个分号——cl 报的也是「语法错误:“;”」。两边对得上，
        # 说明 column 没有差一，也没有被前面的空格数错
        item = parse(MSVC_SYNTAX).diagnostics[0]
        line = self.read_line(item.line)

        self.assertEqual(line[item.column - 1], ";")
        self.assertEqual(item.message, "语法错误:“;”")

    def test_the_line_number_is_not_shifted_by_a_bom(self):
        # 带 BOM 的文件：BOM 不该被算成第一行的内容，更不该多出一行
        (self.root / "bom.cpp").write_bytes(
            b"\xef\xbb\xbf" + "int main(){\n  bad();\n}\n".encode("utf-8")
        )

        text = ReadFileTool(root=self.root).run(path="bom.cpp")
        lines = split_lines(text)

        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
