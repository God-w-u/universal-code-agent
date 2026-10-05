"""ReadFileTool 测试：正常读取、编码、行号、截断、非法路径与越界。

全部在临时目录里跑，不联网。
"""

import tempfile
import unittest
from pathlib import Path

from config import PROJECT_ROOT
from tools import ReadFileTool, Tool, ToolError


class ReadFileToolTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str = "内容") -> Path:
        """精确写入 UTF-8 字节，不做换行转换。

        用 write_text 的话 Windows 会把 \\n 变成 \\r\\n，
        而工具是如实返回文件内容的，测试就会平白无故地在平台间不一致。
        """
        return self.make_bytes(relative, content.encode("utf-8"))

    def make_bytes(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read(self, **kwargs) -> str:
        return ReadFileTool(root=self.root).run(**kwargs)


class ReadFileToolBasicTest(ReadFileToolTestBase):
    def test_reads_file_content(self):
        self.make("main.py", "print('hello')\n")

        self.assertEqual(self.read(path="main.py"), "1|print('hello')")

    def test_reads_file_in_subdirectory(self):
        self.make("agent/agent.py", "class Agent: ...")

        self.assertEqual(self.read(path="agent/agent.py"), "1|class Agent: ...")

    def test_preserves_chinese_and_indentation(self):
        content = "第一行\n第二行\n\t缩进\n"

        self.make("cn.py", content)

        self.assertEqual(
            self.read(path="cn.py"), "1|第一行\n2|第二行\n3|\t缩进"
        )

    def test_empty_file_returns_empty_string(self):
        self.make("empty.py", "")

        self.assertEqual(self.read(path="empty.py"), "")

    def test_it_is_a_tool(self):
        tool = ReadFileTool(root=self.root)

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "read_file")
        self.assertTrue(tool.description)

    def test_path_is_required_in_schema(self):
        parameters = ReadFileTool(root=self.root).to_schema()["function"]["parameters"]

        self.assertEqual(list(parameters["properties"]), ["path"])
        self.assertEqual(parameters["required"], ["path"])

    def test_missing_path_argument_rejected(self):
        with self.assertRaises(ToolError):
            ReadFileTool(root=self.root).run()


class ReadFileToolLineNumberTest(ReadFileToolTestBase):
    """行号本身：从 1 开始、空行占位、宽度对齐、怎么数出来的。

    行号是给模型拿去引用位置的，错一行就等于把人指到隔壁去了。
    """

    def test_line_numbers_start_at_one(self):
        self.make("m.py", "first\nsecond\nthird\n")

        self.assertEqual(self.read(path="m.py"), "1|first\n2|second\n3|third")

    def test_a_missing_trailing_newline_still_counts_that_line(self):
        # 文件末尾没有换行符，最后那行也是一行
        self.make("m.py", "a\nb")

        self.assertEqual(self.read(path="m.py"), "1|a\n2|b")

    def test_blank_lines_are_counted(self):
        # 空行也要占一个行号，不然它后面的行号全会往前错一位
        self.make("m.py", "a\n\nb\n")

        self.assertEqual(self.read(path="m.py"), "1|a\n2|\n3|b")

    def test_a_file_that_is_only_a_newline(self):
        self.make("m.py", "\n")

        self.assertEqual(self.read(path="m.py"), "1|")

    def test_a_single_line_file_needs_no_padding(self):
        self.make("m.py", "only = 1\n")

        self.assertEqual(self.read(path="m.py"), "1|only = 1")

    def test_crlf_is_normalized(self):
        # \r\n 只能算一个换行，行尾也不该留一个看不见的 \r
        self.make_bytes("crlf.py", b"a\r\nb\r\n")

        self.assertEqual(self.read(path="crlf.py"), "1|a\n2|b")

    def test_a_form_feed_does_not_start_a_new_line(self):
        # 换页符在老的 C 代码里是当分页用的。splitlines() 会在那儿断行，
        # 行号就和编辑器对不上了——所以切行不用 splitlines()。
        self.make("ff.c", "aaa\x0cbbb\nccc\n")

        self.assertEqual(self.read(path="ff.c"), "1|aaa\x0cbbb\n2|ccc")

    def test_numbers_are_padded_to_the_widest_one(self):
        self.make("m.py", "".join(f"line {i}\n" for i in range(1, 11)))

        result = self.read(path="m.py")

        self.assertTrue(result.startswith(" 1|line 1\n 2|line 2\n"))
        self.assertTrue(result.endswith("\n10|line 10"))

    def test_padding_follows_the_whole_file_not_what_was_returned(self):
        # 宽度按**整个文件**的行数算，不是按这次返回了几行：
        # 否则同一个文件截断前后，同一行的编号会变，模型对不上。
        class SmallReturnTool(ReadFileTool):
            max_return_chars = 40

        self.make("m.py", "".join(f"line {i}\n" for i in range(1, 201)))

        result = SmallReturnTool(root=self.root).run(path="m.py")

        # 200 行 → 宽度 3
        self.assertTrue(result.startswith("  1|line 1\n  2|line 2\n"))

    def test_it_agrees_with_search_code(self):
        # 两个工具都得数出同一套行号，不然模型会被指到错的地方去。
        # 空行最容易露馅：数行时漏掉它，后面的行号就会往前错一位。
        from tools import SearchCodeTool

        self.make("m.py", "alpha\n\nbeta\ngamma\n")

        self.assertIn(
            "m.py:4: gamma",
            SearchCodeTool(root=self.root).run(keyword="gamma"),
        )
        self.assertEqual(self.read(path="m.py").split("\n")[3], "4|gamma")


class ReadFileToolBadPathTest(ReadFileToolTestBase):
    def test_missing_file_rejected(self):
        with self.assertRaises(ToolError):
            self.read(path="nope.py")

    def test_directory_rejected(self):
        (self.root / "sub").mkdir()

        with self.assertRaises(ToolError) as caught:
            self.read(path="sub")

        self.assertIn("目录", str(caught.exception))

    def test_parent_traversal_rejected(self):
        for path in ("..", "../secret.txt", "a/../../secret.txt"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    self.read(path=path)

    def test_absolute_path_outside_root_rejected(self):
        with self.assertRaises(ToolError):
            self.read(path=str(self.root.parent))

    def test_system_files_rejected(self):
        for path in ("C:/Windows/win.ini", "/etc/passwd", "/etc/hosts"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    self.read(path=path)

    def test_traversal_error_message_names_the_root(self):
        with self.assertRaises(ToolError) as caught:
            self.read(path="../secret.txt")

        self.assertIn(str(self.root), str(caught.exception))


class ReadFileToolEncodingTest(ReadFileToolTestBase):
    def test_utf8_bom_is_stripped(self):
        self.make_bytes("bom.py", "你好".encode("utf-8-sig"))

        self.assertEqual(self.read(path="bom.py"), "1|你好")

    def test_gbk_file_is_decoded(self):
        self.make_bytes("gbk.py", "中文注释".encode("gbk"))

        self.assertEqual(self.read(path="gbk.py"), "1|中文注释")

    def test_binary_file_rejected(self):
        self.make_bytes("bin.dat", b"\x00\x01\x02\xff\xfe")

        with self.assertRaises(ToolError) as caught:
            self.read(path="bin.dat")

        self.assertIn("NUL", str(caught.exception))

    def test_undecodable_bytes_rejected(self):
        # 不含 NUL，但既不是合法 UTF-8 也不是合法 GBK
        self.make_bytes("weird.dat", b"\xff\xff\xff\xff")

        with self.assertRaises(ToolError):
            self.read(path="weird.dat")

    def test_truncation_does_not_cut_a_character_in_half(self):
        class TinyReadTool(ReadFileTool):
            # 「你好世界」是 12 字节，读到第 7 字节正好切在一个汉字中间
            max_read_bytes = 7
            max_return_chars = 10_000

        self.make_bytes("cn.py", "你好世界".encode("utf-8"))

        result = TinyReadTool(root=self.root).run(path="cn.py")

        self.assertTrue(result.startswith("1|你好"))
        self.assertNotIn("\ufffd", result)  # 不能出现替换字符式的乱码


class ReadFileToolTruncationTest(ReadFileToolTestBase):
    def test_byte_cap_truncates_and_says_so(self):
        class SmallReadTool(ReadFileTool):
            max_read_bytes = 1024
            max_return_chars = 100_000  # 排除字符上限的干扰

        self.make("big.py", "a" * 5000)

        result = SmallReadTool(root=self.root).run(path="big.py")

        self.assertIn("a" * 1024, result)
        self.assertIn("只读了开头", result)

    def test_char_cap_truncates_and_reports_the_line_range(self):
        class SmallReturnTool(ReadFileTool):
            max_return_chars = 40

        self.make("m.py", "".join(f"line {i}\n" for i in range(1, 21)))

        result = SmallReturnTool(root=self.root).run(path="m.py")

        # 20 行 → 宽度 2，每行 "NN|line N" 是 9 个字符，加换行正好 10
        self.assertEqual(result.split("\n")[0], " 1|line 1")
        self.assertIn("文件共 20 行，只返回了第 1–4 行", result)

    def test_truncation_drops_whole_lines_only(self):
        # 按字符砍的话最后一行会是半截的，「返回了第 1–N 行」这句话也就不成立了
        class SmallReturnTool(ReadFileTool):
            max_return_chars = 15

        self.make("m.py", "aaaa\nbbbb\ncccc\n")

        result = SmallReturnTool(root=self.root).run(path="m.py")

        self.assertEqual(result.split("\n")[:2], ["1|aaaa", "2|bbbb"])
        self.assertIn("文件共 3 行，只返回了第 1–2 行", result)

    def test_one_line_is_written_without_a_range(self):
        # 「第 1–1 行」读着别扭，只剩一行时就写「第 1 行」
        class SmallReturnTool(ReadFileTool):
            max_return_chars = 12

        self.make("m.py", "aaaa\nbbbb\ncccc\n")

        self.assertIn(
            "文件共 3 行，只返回了第 1 行",
            SmallReturnTool(root=self.root).run(path="m.py"),
        )

    def test_a_first_line_that_cannot_fit_returns_no_content(self):
        # 整份压成一行的文件（压缩过的 JS 之类）就是这种情况。
        # 不返回正文比返回好：硬塞四万个字符的乱码只会白占上下文。
        class TinyTool(ReadFileTool):
            max_return_chars = 10

        self.make("f.py", "0123456789ABCDEF\n")

        result = TinyTool(root=self.root).run(path="f.py")

        self.assertTrue(result.startswith("…（"))
        self.assertIn("第 1 行就有 16 个字符", result)
        self.assertIn("没有返回内容", result)

    def test_both_caps_at_once_says_the_line_count_is_partial(self):
        # 字节上限先砍过一刀的话，总行数只能按「读到的部分」说，
        # 不能报一个看着像全文行数的数字
        class SmallTool(ReadFileTool):
            max_read_bytes = 1024
            max_return_chars = 20

        self.make("m.py", "".join(f"line {i}\n" for i in range(1, 301)))

        result = SmallTool(root=self.root).run(path="m.py")

        self.assertIn("只读了开头", result)
        self.assertIn("读到的部分共", result)
        self.assertIn("只返回了第 1 行", result)

    def test_no_note_for_a_small_file(self):
        self.make("small.py", "x = 1\n")

        self.assertEqual(self.read(path="small.py"), "1|x = 1")

    def test_line_numbers_do_not_blow_up_the_result(self):
        # 行号前缀也占额度。它要是把总量撑得远超上限，ContextManager 会把
        # 工具自己写在末尾的截断说明再砍一次，模型就不知道这只是半份文件了。
        # 正文不超过 max_return_chars，末尾那句说明写在这个数之外（+100 是它的余量）。
        self.make("m.py", "".join(f"line {i}\n" for i in range(1, 20001)))

        result = ReadFileTool(root=self.root).run(path="m.py")

        self.assertIn("只返回了第 1–", result)
        self.assertLessEqual(len(result), ReadFileTool.max_return_chars + 100)

    def test_limits_are_actually_bounded(self):
        # 常量本身得够小，否则"限制"等于没有
        self.assertLessEqual(ReadFileTool.max_return_chars, 100_000)
        self.assertLessEqual(ReadFileTool.max_read_bytes, 1024 * 1024)


class ReadFileToolRootTest(unittest.TestCase):
    """root 校验直接复用 paths.check_root，这里确认工具真的用上了。"""

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            ReadFileTool(root=Path("C:/"))

    def test_system_directory_as_root_rejected(self):
        for path in ("C:/Windows", "C:/Program Files", "/etc"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    ReadFileTool(root=Path(path))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            ReadFileTool(root=Path("C:/definitely/not/a/real/dir/xyz"))

    def test_default_root_is_project_root(self):
        self.assertEqual(ReadFileTool().root, PROJECT_ROOT)


class ReadFileToolSymlinkTest(ReadFileToolTestBase):
    def test_symlink_escaping_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.txt"
            secret.write_text("机密", encoding="utf-8")
            link = self.root / "link.txt"
            try:
                link.symlink_to(secret)
            except (OSError, NotImplementedError):
                self.skipTest("当前环境不支持创建符号链接")

            with self.assertRaises(ToolError):
                self.read(path="link.txt")

    def test_symlink_staying_inside_root_is_allowed(self):
        self.make("real.txt", "内容")
        link = self.root / "alias.txt"
        try:
            link.symlink_to(self.root / "real.txt")
        except (OSError, NotImplementedError):
            self.skipTest("当前环境不支持创建符号链接")

        self.assertEqual(self.read(path="alias.txt"), "1|内容")


class ReadFileToolSensitiveTest(ReadFileToolTestBase):
    """密钥 / 凭据文件连「点名读」都不行（13 阶段回归）。

    遍历挡住的是「顺手带出来」；这里挡的是模型直接 `read_file(".env")`——
    那句话一执行，密钥原文就进了模型上下文。
    """

    def test_reading_a_secret_file_is_rejected(self):
        for name in (".env", ".env.local", "server.pem", "id_rsa", "deploy.key"):
            with self.subTest(name=name):
                self.make(name, "OPENAI_API_KEY=sk-真钥匙不该出现在这里")

                with self.assertRaises(ToolError):
                    self.read(path=name)

    def test_the_error_does_not_echo_the_content(self):
        self.make(".env", "OPENAI_API_KEY=sk-真钥匙不该出现在这里")

        with self.assertRaises(ToolError) as caught:
            self.read(path=".env")

        self.assertNotIn("sk-", str(caught.exception))
        self.assertIn(".env.example", str(caught.exception))  # 告诉它该看哪儿

    def test_the_public_template_can_still_be_read(self):
        self.make(".env.example", "OPENAI_API_KEY=\n")

        self.assertEqual(self.read(path=".env.example"), "1|OPENAI_API_KEY=")

    def test_a_normal_file_is_unaffected(self):
        self.make("environment.py", "KEY = '从环境变量读'\n")

        self.assertEqual(self.read(path="environment.py"), "1|KEY = '从环境变量读'")


class ReadFileToolRealProjectTest(unittest.TestCase):
    """直接读本项目，确认默认 root 和真实文件都对得上。"""

    def test_reads_a_real_file(self):
        result = ReadFileTool().run(path="README.md")

        self.assertIn("Universal Code Analysis Agent", result)

    def test_reads_own_source(self):
        result = ReadFileTool().run(path="agent/agent.py")

        self.assertIn("class Agent", result)

    def test_cannot_read_outside_the_project(self):
        with self.assertRaises(ToolError):
            ReadFileTool().run(path="../config.py")

    def test_pairs_with_list_files(self):
        from tools import ListFilesTool

        listing = ListFilesTool().run(path="llm")
        for relative in listing.splitlines():
            # list_files 给出的相对路径应当能直接喂给 read_file
            self.assertTrue(ReadFileTool().run(path=relative))


if __name__ == "__main__":
    unittest.main()
