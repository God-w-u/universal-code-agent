"""CodeSummaryTool 测试：Python 的类/函数/async 提取、C/C++ 的结构提取、
坏文件跳过、路径安全、输出限制。

全部在临时目录里跑，不联网。
"""

import tempfile
import unittest
from pathlib import Path

from config import PROJECT_ROOT
from tools import (
    CodeSummaryTool,
    ReadFileTool,
    SearchCodeTool,
    Tool,
    ToolError,
)


def section(result: str, path: str) -> list[str]:
    """取出某个文件那段的结构行（去掉缩进）；路径行本身不算。"""
    rows: list[str] = []
    current: str | None = None
    for line in result.split("\n"):
        if line and not line.startswith(" "):
            current = line[:-1] if line.endswith(":") else line
            continue
        if current == path and line.strip():
            rows.append(line.strip())
    return rows


def block(result: str, path: str) -> list[str]:
    """和 section 一样，但**留着缩进**——C/C++ 那条要看层级。"""
    head = f"{path}:\n"
    start = result.index(head) + len(head)
    stop = result.find("\n\n", start)
    return (result[start:] if stop < 0 else result[start:stop]).split("\n")


class CodeSummaryToolTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str) -> Path:
        """精确写入 UTF-8 字节，不做换行转换，保证跨平台一致。"""
        return self.make_bytes(relative, content.encode("utf-8"))

    def make_bytes(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def summary(self, **kwargs) -> str:
        return CodeSummaryTool(root=self.root).run(**kwargs)


class CodeSummaryToolStructureTest(CodeSummaryToolTestBase):
    def test_extracts_class(self):
        self.make("m.py", "class Alpha:\n    pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["class Alpha"])

    def test_extracts_methods_of_a_class(self):
        self.make("m.py", "class Alpha:\n    def run(self):\n        pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["class Alpha", "def run"])

    def test_extracts_module_level_function(self):
        self.make("m.py", "def top():\n    pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["def top"])

    def test_extracts_async_function(self):
        self.make("m.py", "async def fetch():\n    pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["async def fetch"])

    def test_extracts_async_method(self):
        self.make("m.py", "class A:\n    async def go(self):\n        pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["class A", "async def go"])

    def test_nesting_is_shown_by_indentation(self):
        self.make("m.py", "class Outer:\n    def method(self):\n        pass\n")

        self.assertIn("\n  class Outer\n    def method", self.summary())

    def test_nested_class(self):
        self.make("m.py", "class Outer:\n    class Inner:\n        pass\n")

        self.assertEqual(
            section(self.summary(), "m.py"), ["class Outer", "class Inner"]
        )

    def test_definitions_inside_if_are_found(self):
        self.make("m.py", 'if __name__ == "__main__":\n    def main():\n        pass\n')

        self.assertEqual(section(self.summary(), "m.py"), ["def main"])

    def test_definitions_inside_try_are_found(self):
        self.make(
            "m.py",
            "try:\n    import fast\nexcept ImportError:\n    def slow():\n        pass\n",
        )

        self.assertEqual(section(self.summary(), "m.py"), ["def slow"])

    def test_definitions_inside_with_are_found(self):
        self.make("m.py", "with open('x') as handle:\n    def helper():\n        pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["def helper"])

    def test_definition_inside_a_loop_keeps_its_depth(self):
        self.make("m.py", "for name in 'ab':\n    def made():\n        pass\n")

        self.assertEqual(section(self.summary(), "m.py"), ["def made"])

    def test_container_does_not_add_a_level(self):
        self.make("m.py", "if True:\n    class A:\n        def m(self):\n            pass\n")

        # if 本身不是定义，所以不多缩进一层
        self.assertIn("\n  class A\n    def m", self.summary())

    def test_class_body_statement_is_not_a_definition(self):
        self.make("m.py", "class A:\n    VALUE = 1\n")

        self.assertEqual(section(self.summary(), "m.py"), ["class A"])

    def test_empty_file_is_marked(self):
        self.make("empty.py", "")

        self.assertIn("（没有类或函数）", self.summary())

    def test_file_with_only_imports_is_marked(self):
        self.make("m.py", "import os\n")

        self.assertIn("（没有类或函数）", self.summary())

    def test_files_are_ordered_by_path(self):
        self.make("b.py", "def bee():\n    pass\n")
        self.make("a.py", "def ay():\n    pass\n")

        result = self.summary()

        self.assertLess(result.index("a.py:"), result.index("b.py:"))

    def test_subdirectory_argument_limits_scope(self):
        self.make("pkg/a.py", "def ay():\n    pass\n")
        self.make("other/b.py", "def bee():\n    pass\n")

        result = self.summary(path="pkg")

        self.assertIn("pkg/a.py:", result)
        self.assertNotIn("other/", result)

    def test_relative_paths_are_relative_to_root(self):
        self.make("pkg/a.py", "def ay():\n    pass\n")

        self.assertIn("pkg/a.py:", self.summary(path="pkg"))

    def test_it_is_a_tool(self):
        tool = CodeSummaryTool(root=self.root)

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "code_summary")
        self.assertTrue(tool.description)


class CodeSummaryToolSkipTest(CodeSummaryToolTestBase):
    """坏文件只影响自己，不能拖垮整次分析。"""

    def test_non_python_files_are_skipped(self):
        self.make("notes.txt", "class NotReal:\n    pass\n")
        self.make("m.py", "def real():\n    pass\n")

        result = self.summary()

        self.assertNotIn("notes.txt", result)
        self.assertIn("m.py:", result)

    def test_syntax_error_file_is_skipped_but_others_survive(self):
        self.make("broken.py", "def (:\n")
        self.make("good.py", "def fine():\n    pass\n")

        result = self.summary()

        self.assertIn("语法错误", result)
        self.assertEqual(section(result, "good.py"), ["def fine"])

    def test_syntax_error_does_not_raise(self):
        self.make("broken.py", "class A(:\n")

        self.summary()

    def test_undecodable_python_file_is_skipped(self):
        self.make_bytes("weird.py", b"\xff\xff\xff\xff")
        self.make("good.py", "def fine():\n    pass\n")

        result = self.summary()

        self.assertIn("不是文本文件", result)
        self.assertEqual(section(result, "good.py"), ["def fine"])

    def test_binary_python_file_is_skipped(self):
        self.make_bytes("bin.py", b"\x00\x01\x02")

        self.assertIn("已跳过", self.summary())

    def test_noise_directories_are_skipped(self):
        self.make("keep.py", "def keep():\n    pass\n")
        for noise in (".git", ".venv", "__pycache__", "node_modules", "build"):
            self.make(f"{noise}/junk.py", "def junk():\n    pass\n")

        result = self.summary()

        self.assertIn("keep.py:", result)
        self.assertNotIn("junk", result)

    def test_no_source_files_reports_clearly(self):
        self.make("readme.md", "# hi\n")

        self.assertIn("没有可分析的源文件", self.summary())

    def test_empty_directory_reports_no_source_files(self):
        self.assertIn("没有可分析的源文件", self.summary())


class CodeSummaryToolCFileTest(CodeSummaryToolTestBase):
    """.c / .h / .cpp / .hpp 四种后缀都要认，而且每条都带行号。"""

    def test_c_file(self):
        self.make("m.c", "#include <stdio.h>\n\nint main(void) {\n    return 0;\n}\n")

        self.assertEqual(
            section(self.summary(), "m.c"), ["1|#include <stdio.h>", "3|main(void)"]
        )

    def test_h_file(self):
        self.make("m.h", "#ifndef M_H\n#define M_H\nvoid init(void);\n#endif\n")

        # 预处理指令里只认 include，别的都不报
        self.assertEqual(section(self.summary(), "m.h"), ["3|init(void)"])

    def test_cpp_file(self):
        self.make("m.cpp", "namespace app {\nclass A {\npublic:\n    A();\n};\n}\n")

        self.assertEqual(
            section(self.summary(), "m.cpp"),
            ["1|namespace app", "2|class A", "4|A()"],
        )

    def test_hpp_file(self):
        self.make("m.hpp", "struct Point {\n    int x, y;\n};\n")

        self.assertEqual(section(self.summary(), "m.hpp"), ["1|struct Point"])

    def test_all_four_suffixes_are_summarised_together(self):
        for name in ("a.c", "b.h", "c.cpp", "d.hpp"):
            self.make(name, "void f(void);\n")

        result = self.summary()

        for name in ("a.c:", "b.h:", "c.cpp:", "d.hpp:"):
            with self.subTest(path=name):
                self.assertIn(name, result)

    def test_c_and_python_live_side_by_side(self):
        self.make("m.py", "def f():\n    pass\n")
        self.make("m.cpp", "void f(void);\n")

        result = self.summary()

        # Python 那条还是老样子（不带行号），C++ 那条带行号
        self.assertEqual(section(result, "m.py"), ["def f"])
        self.assertEqual(section(result, "m.cpp"), ["1|f(void)"])

    def test_every_kind_is_reported(self):
        self.make(
            "m.hpp",
            '#include <vector>\n'
            "\n"
            "namespace app {\n"
            "class A {\n"
            "    void run();\n"
            "};\n"
            "struct B {\n"
            "};\n"
            "union U {\n"
            "};\n"
            "enum class Color { Red };\n"
            "void free_function(int x);\n"
            "}\n",
        )

        self.assertEqual(
            section(self.summary(), "m.hpp"),
            [
                "1|#include <vector>",
                "3|namespace app",
                "4|class A",
                "5|run()",
                "7|struct B",
                "9|union U",
                "11|enum class Color",
                "12|free_function(int x)",
            ],
        )

    def test_nesting_is_shown_by_indentation(self):
        self.make("m.hpp", "namespace app {\nclass A {\n    void run();\n};\n}\n")

        self.assertEqual(
            block(self.summary(), "m.hpp"),
            ["  1|namespace app", "    2|class A", "      3|run()"],
        )

    def test_several_functions_are_all_listed(self):
        self.make(
            "m.cpp",
            "int add(int a, int b);\n"
            "int sub(int a, int b);\n"
            "static int helper(void) {\n"
            "    return mul(2, 3);\n"  # 函数体里的调用不算结构
            "}\n",
        )

        self.assertEqual(
            section(self.summary(), "m.cpp"),
            ["1|add(int a, int b)", "2|sub(int a, int b)", "3|helper(void)"],
        )


class CodeSummaryToolCLineNumberTest(CodeSummaryToolTestBase):
    """行号得和 read_file / search_code 是同一个数，不然模型核对不上。"""

    def test_it_agrees_with_read_file(self):
        self.make("m.cpp", "#include <vector>\n\nclass Widget {\n};\n")

        rows = section(self.summary(), "m.cpp")
        self.assertTrue(rows)

        read = ReadFileTool(root=self.root).run(path="m.cpp").split("\n")
        for row in rows:
            number, _, content = row.partition("|")
            line = read[int(number) - 1]
            # 行号是同一个数，而且那一行的原文确实以报出来的名字开头
            # （行尾的 { 不会出现在摘要里，摘要写的是「什么东西在第几行」）
            self.assertEqual(line.partition("|")[0], number)
            self.assertTrue(
                line.partition("|")[2].strip().startswith(content.strip()),
                f"第 {number} 行是 {line!r}，和报出来的 {content!r} 对不上",
            )

    def test_it_agrees_with_search_code(self):
        self.make("m.h", "// 头部\nvoid init(void);\n")

        self.assertIn(
            "m.h:2: void init(void);",
            SearchCodeTool(root=self.root).run(keyword="init"),
        )
        self.assertEqual(section(self.summary(), "m.h"), ["2|init(void)"])

    def test_width_follows_the_whole_file(self):
        # 宽度按整个文件的行数算，和 read_file 一个规矩（120 行 → 3 位）
        self.make(
            "m.cpp",
            "".join(f"// 第 {index} 行\n" for index in range(1, 120)) + "void f(void);\n",
        )

        self.assertIn("  120|f(void)", self.summary())

    def test_crlf_and_form_feed_follow_the_same_rule(self):
        # 换页符不换行、CRLF 只算一个换行——和 read_file / search_code 同一套
        self.make_bytes("m.cpp", b"void a(void);\r\n\x0c\r\nvoid b(void);\r\n")

        self.assertEqual(
            section(self.summary(), "m.cpp"), ["1|a(void)", "3|b(void)"]
        )


class CodeSummaryToolCWeirdTest(CodeSummaryToolTestBase):
    """畸形输入只能让结果少一点，不能拖垮整次分析。"""

    def test_a_file_with_no_structure_is_marked(self):
        self.make("m.c", "/* 只有注释 */\n")

        self.assertIn("没有可提取的结构", self.summary())

    def test_weird_syntax_does_not_raise(self):
        self.make("weird.cpp", "template <\nstruct {\n)))]]}}\nint ((((;\n")

        self.summary()  # 不抛就算过

    def test_a_binary_cpp_file_is_skipped(self):
        self.make_bytes("bin.cpp", b"\x00\x01\x02")
        self.make("good.cpp", "void f(void);\n")

        result = self.summary()

        self.assertIn("不是文本文件", result)
        self.assertEqual(section(result, "good.cpp"), ["1|f(void)"])

    def test_an_oversized_cpp_file_is_skipped(self):
        class TinyTool(CodeSummaryTool):
            max_file_bytes = 8

        self.make("big.cpp", "void " + "a" * 100 + "(void);\n")

        self.assertIn("文件过大", TinyTool(root=self.root).run())


class CodeSummaryToolPathTest(CodeSummaryToolTestBase):
    def test_escaping_root_is_rejected(self):
        with self.assertRaises(ToolError):
            self.summary(path="../..")

    def test_absolute_path_outside_root_is_rejected(self):
        with self.assertRaises(ToolError):
            self.summary(path="C:/Windows")

    def test_file_argument_is_rejected(self):
        self.make("m.py", "def f():\n    pass\n")

        with self.assertRaises(ToolError):
            self.summary(path="m.py")


class CodeSummaryToolLimitTest(CodeSummaryToolTestBase):
    def test_file_limit_is_reported(self):
        class TinyTool(CodeSummaryTool):
            max_files = 2

        for index in range(4):
            self.make(f"m{index}.py", f"def f{index}():\n    pass\n")

        result = TinyTool(root=self.root).run()

        self.assertIn("只分析了前 2 个", result)
        self.assertEqual(len(section(result, "m3.py")), 0)

    def test_no_file_limit_note_when_under(self):
        self.make("m.py", "def f():\n    pass\n")

        self.assertNotIn("只分析了前", self.summary())

    def test_output_limit_is_reported(self):
        class TinyTool(CodeSummaryTool):
            max_output_chars = 60

        for index in range(20):
            self.make(f"m{index:02d}.py", f"def function_number_{index}():\n    pass\n")

        result = TinyTool(root=self.root).run()

        self.assertIn("只显示了前", result)

    def test_output_limit_actually_bounds_the_text(self):
        class TinyTool(CodeSummaryTool):
            max_output_chars = 60

        for index in range(20):
            self.make(f"m{index:02d}.py", f"def function_number_{index}():\n    pass\n")

        result = TinyTool(root=self.root).run()
        body = result.split("…（")[0].rstrip()

        self.assertLessEqual(len(body), 60)

    def test_single_oversized_block_is_hard_truncated(self):
        class TinyTool(CodeSummaryTool):
            max_output_chars = 40

        self.make("m.py", "".join(f"def f{index}():\n    pass\n" for index in range(30)))

        result = TinyTool(root=self.root).run()

        self.assertLessEqual(len(result.split("…（")[0].rstrip()), 40)
        self.assertIn("只返回了开头", result)

    def test_oversized_file_is_skipped(self):
        class TinyTool(CodeSummaryTool):
            max_file_bytes = 8

        self.make("big.py", "def " + "a" * 100 + "():\n    pass\n")

        self.assertIn("文件过大", TinyTool(root=self.root).run())

    def test_limits_are_actually_bounded(self):
        self.assertLessEqual(CodeSummaryTool.max_files, 1000)
        self.assertLessEqual(CodeSummaryTool.max_output_chars, 50_000)
        self.assertLessEqual(CodeSummaryTool.max_file_bytes, 4 * 1024 * 1024)


class CodeSummaryToolRootTest(unittest.TestCase):
    def test_default_root_is_project_root(self):
        self.assertEqual(CodeSummaryTool().root, PROJECT_ROOT)

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            CodeSummaryTool(root=Path("C:/"))

    def test_system_directory_as_root_rejected(self):
        for path in ("C:/Windows", "C:/Program Files", "/etc"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    CodeSummaryTool(root=Path(path))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            CodeSummaryTool(root=Path("C:/definitely/not/a/real/dir/xyz"))


class CodeSummaryToolRealProjectTest(unittest.TestCase):
    def test_summarizes_this_project(self):
        result = CodeSummaryTool().run()

        self.assertIn("agent/agent.py:", result)
        self.assertIn("class Agent", result)
        self.assertIn("def _run_loop", result)

    def test_reports_classes_from_other_modules(self):
        # 指定目录而不是整个项目：文件是按路径排的，tools/ 排最后，
        # 项目一大就会被输出上限截掉，这个断言会跟着误报
        result = CodeSummaryTool().run(path="tools")

        self.assertIn("class ToolRegistry", result)
        self.assertIn("class ReadFileTool", result)

    def test_no_results_come_from_noise_directories(self):
        result = CodeSummaryTool().run()

        self.assertNotIn(".venv", result)
        self.assertNotIn("__pycache__", result)

    def test_subdirectory_argument_works(self):
        result = CodeSummaryTool().run(path="tools")

        self.assertIn("tools/calculator.py:", result)
        self.assertNotIn("agent/agent.py:", result)

    def test_output_stays_under_the_limit(self):
        result = CodeSummaryTool().run()

        self.assertLessEqual(len(result), CodeSummaryTool.max_output_chars + 200)


if __name__ == "__main__":
    unittest.main()
