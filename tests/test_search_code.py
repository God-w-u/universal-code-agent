"""SearchCodeTool 测试：正常搜索、大小写、行号、数量限制、噪声目录、二进制文件。

全部在临时目录里跑，不联网。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import PROJECT_ROOT
from tools import (
    ReadFileTool,
    SearchCodeTool,
    Tool,
    ToolError,
    ToolExecutor,
    ToolRegistry,
)


def parse(result: str) -> list[tuple[str, int, str]]:
    """把结果解析成 (相对路径, 行号, 内容)。

    末尾的说明行（"…（匹配过多…）"、"（没有找到…）"）会被跳过。
    """
    rows = []
    for line in result.splitlines():
        if line.startswith("…（") or line.startswith("（"):
            continue
        path, number, content = line.split(":", 2)
        rows.append((path, int(number), content.removeprefix(" ")))
    return rows


class SearchCodeToolTestBase(unittest.TestCase):
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

    def search(self, **kwargs) -> str:
        return SearchCodeTool(root=self.root).run(**kwargs)

    def rows(self, **kwargs) -> list[tuple[str, int, str]]:
        return parse(self.search(**kwargs))


class SearchCodeToolSensitiveTest(SearchCodeToolTestBase):
    """搜密钥文件：命中了也不许返回（13 阶段回归）。

    一句 `search_code "OPENAI_API_KEY"` 就能把密钥原文送进模型上下文，
    送出去就收不回来了——所以这些文件根本不进遍历（tools/tree.py）。
    """

    SECRET = "sk-真钥匙不该出现在这里"

    def test_a_secret_file_is_never_searched(self):
        self.make("main.py", "KEY = '占位'\n")
        self.make(".env", f"OPENAI_API_KEY={self.SECRET}\n")

        result = self.search(keyword="OPENAI_API_KEY")

        self.assertEqual(parse(result), [])
        self.assertNotIn(self.SECRET, result)
        self.assertNotIn("OPENAI_API_KEY=", result)

    def test_searching_for_the_value_itself_finds_nothing(self):
        self.make(".env.local", f"TOKEN={self.SECRET}\n")
        self.make("deploy/id_rsa", f"-----BEGIN {self.SECRET}-----\n")

        result = self.search(keyword=self.SECRET)

        # 「没找到」那句会把关键词原样带回显（那是**提问**，不是文件内容），
        # 所以只钉「没有一行来自密钥文件」
        self.assertEqual(parse(result), [])
        self.assertNotIn(f"TOKEN={self.SECRET}", result)
        self.assertNotIn("deploy/id_rsa", result)

    def test_the_public_template_is_still_searched(self):
        self.make(".env.example", "OPENAI_API_KEY=\n")

        self.assertEqual(
            self.rows(keyword="OPENAI_API_KEY"), [(".env.example", 1, "OPENAI_API_KEY=")]
        )


class SearchCodeToolBasicTest(SearchCodeToolTestBase):
    def test_reports_path_line_number_and_content(self):
        self.make("tools/calculator.py", '"""doc"""\n\ndef f():\n    return result\n')

        self.assertEqual(
            self.rows(keyword="return result"),
            [("tools/calculator.py", 4, "return result")],
        )

    def test_matches_are_ordered_by_path(self):
        self.make("b.py", "hit one\n")
        self.make("a.py", "hit two\n")

        self.assertEqual(
            self.rows(keyword="hit"), [("a.py", 1, "hit two"), ("b.py", 1, "hit one")]
        )

    def test_multiple_matches_in_one_file(self):
        self.make("m.py", "alpha\nbeta\nalpha again\n")

        self.assertEqual(
            self.rows(keyword="alpha"),
            [("m.py", 1, "alpha"), ("m.py", 3, "alpha again")],
        )

    def test_matches_inside_subdirectories(self):
        self.make("a/b/c.py", "needle\n")

        self.assertEqual(self.rows(keyword="needle"), [("a/b/c.py", 1, "needle")])

    def test_leading_indentation_is_stripped(self):
        self.make("m.py", "def f():\n        deeply = 1\n")

        self.assertEqual(self.rows(keyword="deeply"), [("m.py", 2, "deeply = 1")])

    def test_substring_match_not_word_match(self):
        self.make("m.py", "concatenate\n")

        self.assertEqual(self.rows(keyword="cat"), [("m.py", 1, "concatenate")])

    def test_no_match_reports_clearly(self):
        self.make("m.py", "alpha\n")

        self.assertIn("没有找到", self.search(keyword="不存在的东西"))

    def test_empty_project_reports_no_match(self):
        self.assertIn("没有找到", self.search(keyword="x"))

    def test_it_is_a_tool(self):
        tool = SearchCodeTool(root=self.root)

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "search_code")
        self.assertTrue(tool.description)


class SearchCodeToolLineNumberTest(SearchCodeToolTestBase):
    def test_line_numbers_are_one_based(self):
        self.make("m.py", "first\nsecond\n")

        self.assertEqual(self.rows(keyword="first")[0][1], 1)

    def test_crlf_file_reports_right_line_and_clean_content(self):
        self.make_bytes("crlf.py", b"aaa\r\nbbb\r\nccc\r\n")

        self.assertEqual(self.rows(keyword="bbb"), [("crlf.py", 2, "bbb")])

    def test_line_numbers_agree_with_read_file(self):
        self.make("m.py", "alpha\nbeta\ngamma\ndelta\n")

        path, number, content = self.rows(keyword="gamma")[0]
        line = ReadFileTool(root=self.root).run(path=path).split("\n")[number - 1]
        numbered, _, text = line.partition("|")

        # 不光是内容对得上，read_file 编的号也得是同一个
        self.assertEqual(int(numbered), number)
        self.assertEqual(text.strip(), content)

    def test_form_feed_does_not_shift_line_numbers(self):
        # splitlines() 会把 \x0c 也当换行，那样行号就和编译器数出来的对不上了
        self.make("m.py", "aaa\x0cbbb\nccc\n")

        self.assertEqual(self.rows(keyword="ccc"), [("m.py", 2, "ccc")])


class SearchCodeToolCaseTest(SearchCodeToolTestBase):
    def setUp(self):
        super().setUp()
        self.make("m.py", "# TODO: fix\n")

    def test_case_insensitive_by_default(self):
        self.assertEqual(self.rows(keyword="todo"), [("m.py", 1, "# TODO: fix")])

    def test_keyword_case_is_irrelevant_when_insensitive(self):
        self.assertEqual(self.rows(keyword="ToDo"), [("m.py", 1, "# TODO: fix")])

    def test_case_sensitive_when_requested(self):
        self.assertIn("没有找到", self.search(keyword="todo", case_sensitive=True))

    def test_case_sensitive_matches_exact_case(self):
        self.assertEqual(
            self.rows(keyword="TODO", case_sensitive=True),
            [("m.py", 1, "# TODO: fix")],
        )


class SearchCodeToolIgnoreTest(SearchCodeToolTestBase):
    def test_noise_directories_are_skipped(self):
        self.make("keep.py", "needle\n")
        for noise in (".git", ".venv", "__pycache__", "node_modules", "build"):
            self.make(f"{noise}/junk.py", "needle\n")

        self.assertEqual(self.rows(keyword="needle"), [("keep.py", 1, "needle")])

    def test_directory_merely_resembling_a_noise_name_is_searched(self):
        self.make("build_tools/make.py", "needle\n")

        self.assertEqual(
            self.rows(keyword="needle"), [("build_tools/make.py", 1, "needle")]
        )


class SearchCodeToolBinaryTest(SearchCodeToolTestBase):
    """解不开的文件要跳过，不能拖垮整次搜索。"""

    def test_binary_file_is_skipped(self):
        self.make_bytes("bin.dat", b"\x00needle\x01\x02")

        self.assertIn("没有找到", self.search(keyword="needle"))

    def test_undecodable_file_does_not_raise(self):
        self.make_bytes("weird.dat", b"\xff\xff\xff\xff")

        self.assertIn("没有找到", self.search(keyword="needle"))

    def test_one_bad_file_does_not_break_the_search(self):
        self.make_bytes("bin.dat", b"\x00needle\x00")
        self.make("good.py", "needle\n")

        self.assertEqual(self.rows(keyword="needle"), [("good.py", 1, "needle")])

    def test_gbk_file_is_searched(self):
        self.make_bytes("gbk.py", "中文注释 needle\n".encode("gbk"))

        self.assertEqual(self.rows(keyword="needle"), [("gbk.py", 1, "中文注释 needle")])


class SearchCodeToolLimitTest(SearchCodeToolTestBase):
    def test_match_limit_truncates_and_says_so(self):
        class TinyLimitTool(SearchCodeTool):
            max_matches = 3

        self.make("m.py", "".join(f"hit {index}\n" for index in range(5)))

        result = TinyLimitTool(root=self.root).run(keyword="hit")

        self.assertEqual(len(parse(result)), 3)
        self.assertIn("只显示前 3 条", result)

    def test_no_note_when_under_the_limit(self):
        self.make("m.py", "hit\n")

        self.assertNotIn("只显示前", self.search(keyword="hit"))

    def test_big_file_truncation_is_reported(self):
        class SmallFileTool(SearchCodeTool):
            max_file_bytes = 8

        self.make("big.py", "x" * 100)

        result = SmallFileTool(root=self.root).run(keyword="x")

        self.assertIn("只搜索了开头部分", result)

    def test_limits_are_actually_bounded(self):
        self.assertLessEqual(SearchCodeTool.max_matches, 500)
        self.assertLessEqual(SearchCodeTool.max_file_bytes, 4 * 1024 * 1024)


class SearchCodeToolArgumentTest(SearchCodeToolTestBase):
    def test_schema_shape(self):
        parameters = SearchCodeTool(root=self.root).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(
            list(parameters["properties"]), ["keyword", "path", "case_sensitive"]
        )
        self.assertEqual(parameters["required"], ["keyword"])

    def test_path_is_optional(self):
        schema = SearchCodeTool(root=self.root).to_schema()["function"]["parameters"]
        self.assertEqual(schema["properties"]["path"]["type"], "string")

    def test_wrong_path_type_rejected(self):
        with self.assertRaises(ToolError):
            self.search(keyword="x", path=123)

    def test_missing_keyword_rejected(self):
        with self.assertRaises(ToolError):
            SearchCodeTool(root=self.root).run()

    def test_empty_keyword_rejected(self):
        with self.assertRaises(ToolError):
            self.search(keyword="")

    def test_wrong_keyword_type_rejected(self):
        with self.assertRaises(ToolError):
            self.search(keyword=123)

    def test_wrong_case_sensitive_type_rejected(self):
        with self.assertRaises(ToolError):
            self.search(keyword="x", case_sensitive="yes")


class SearchCodeToolRootTest(unittest.TestCase):
    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            SearchCodeTool(root=Path("C:/"))

    def test_system_directory_as_root_rejected(self):
        for path in ("C:/Windows", "C:/Program Files", "/etc"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    SearchCodeTool(root=Path(path))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            SearchCodeTool(root=Path("C:/definitely/not/a/real/dir/xyz"))

    def test_default_root_is_project_root(self):
        self.assertEqual(SearchCodeTool().root, PROJECT_ROOT)


class SearchCodeToolRealProjectTest(unittest.TestCase):
    """直接搜本项目，确认结果能接上 read_file。"""

    def test_finds_something_in_this_project(self):
        paths = [path for path, _, _ in parse(SearchCodeTool().run(keyword="class Agent"))]

        self.assertIn("agent/agent.py", paths)

    def test_result_paths_can_be_fed_to_read_file(self):
        rows = parse(SearchCodeTool().run(keyword="DEFAULT_SYSTEM_PROMPT"))
        self.assertTrue(rows)

        path, number, content = rows[0]
        line = ReadFileTool().run(path=path).split("\n")[number - 1]
        numbered, _, text = line.partition("|")

        self.assertEqual(int(numbered), number)
        self.assertEqual(text.strip(), content)

    def test_no_results_come_from_noise_directories(self):
        paths = [path for path, _, _ in parse(SearchCodeTool().run(keyword="import"))]

        self.assertTrue(paths)
        for path in paths:
            self.assertFalse(path.startswith(".venv/"))
            self.assertNotIn("__pycache__", path)


class SearchCodeToolPathTest(SearchCodeToolTestBase):
    """path 参数：把搜索限定在某个子目录里。"""

    def setUp(self):
        super().setUp()
        self.make("src/network/socket.py", "import socket\n")
        self.make("src/db/store.py", "socket = None\n")
        self.make("docs/notes.md", "socket 说明\n")

    def test_default_searches_the_whole_project(self):
        self.assertEqual(len(self.rows(keyword="socket")), 3)

    def test_path_limits_the_search(self):
        rows = self.rows(keyword="socket", path="src/network")

        self.assertEqual([row[0] for row in rows], ["src/network/socket.py"])

    def test_path_can_point_at_a_nested_directory(self):
        rows = self.rows(keyword="socket", path="src/db")

        self.assertEqual([row[0] for row in rows], ["src/db/store.py"])

    def test_dot_is_the_whole_project(self):
        self.assertEqual(len(self.rows(keyword="socket", path=".")), 3)

    def test_paths_stay_relative_to_root(self):
        # 限定子目录后路径也必须相对 root——相对搜索目录的话，喂给 read_file 会找不到
        rows = self.rows(keyword="socket", path="src/network")

        self.assertEqual(rows[0][0], "src/network/socket.py")

    def test_results_can_still_be_fed_to_read_file(self):
        rows = self.rows(keyword="socket", path="src/network")

        text = ReadFileTool(root=self.root).run(path=rows[0][0])

        self.assertIn("import socket", text)

    def test_a_file_path_is_not_a_directory(self):
        with self.assertRaises(ToolError) as ctx:
            self.search(keyword="socket", path="src/network/socket.py")

        self.assertIn("不是目录", str(ctx.exception))

    def test_missing_directory_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.search(keyword="socket", path="src/nope")

        self.assertIn("不是目录", str(ctx.exception))

    def test_parent_escape_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.search(keyword="socket", path="../..")

        self.assertIn("超出允许范围", str(ctx.exception))

    def test_absolute_path_outside_root_rejected(self):
        with self.assertRaises(ToolError):
            self.search(keyword="socket", path=str(self.root.parent))

    def test_directory_without_matches_reports_clearly(self):
        self.make("src/util/helper.py", "pass\n")

        self.assertIn("没有找到", self.search(keyword="socket", path="src/util"))


class SearchCodeToolOSErrorTest(SearchCodeToolTestBase):
    """底层遍历抛 OSError 时要转成 ToolError，不能裸抛给 Agent。"""

    def test_oserror_becomes_toolerror(self):
        self.make("a.py", "x = 1\n")

        with patch("tools.search_code.iter_files", side_effect=OSError("拒绝访问")):
            with self.assertRaises(ToolError) as ctx:
                self.search(keyword="x")

        self.assertIn("读取目录失败", str(ctx.exception))
        self.assertIn("拒绝访问", str(ctx.exception))

    def test_permission_error_becomes_toolerror(self):
        with patch(
            "tools.search_code.iter_files", side_effect=PermissionError("没有权限")
        ):
            with self.assertRaises(ToolError) as ctx:
                self.search(keyword="x")

        self.assertIn("读取目录失败", str(ctx.exception))

    def test_oserror_does_not_leak_out(self):
        # 明确钉住：抛出来的必须是 ToolError，不是 OSError
        with patch("tools.search_code.iter_files", side_effect=OSError("坏了")):
            try:
                self.search(keyword="x")
            except ToolError:
                pass
            except OSError as exc:  # pragma: no cover - 只有修错了才会走到
                self.fail(f"OSError 漏出来了：{exc}")
            else:  # pragma: no cover
                self.fail("没有抛 ToolError")

    def test_the_search_failure_reaches_the_model_as_an_error(self):
        # 走一遍完整的执行层：ToolError → ToolResult(success=False) → 回灌文本
        executor = ToolExecutor(ToolRegistry([SearchCodeTool(root=self.root)]))

        with patch("tools.search_code.iter_files", side_effect=OSError("拒绝访问")):
            result = executor.execute("search_code", keyword="x")

        self.assertFalse(result.success)
        self.assertTrue(result.as_observation().startswith("错误："))


if __name__ == "__main__":
    unittest.main()
