"""EditFileTool 测试：唯一替换、0 次/多次匹配、按行改、越界、编码与换行保持、写坏不伤原文件。

这是第一个会**改**项目文件的工具，所以一大半测试是「不该动手的时候它没动手」：
匹配 0 次、匹配多次、行号越界、路径越界、编码表示不了、写到一半失败——
每一种都要确认文件**一个字节都没变**。改坏了是静默的：模型以为自己成功了。

另一半是「动完手以后文件长什么样」：编码还是原来那个、换行还是原来那样、
别处一个字节没动、结尾有没有换行没被顺手改掉。

全部离线，不联网。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent
from config import PROJECT_ROOT
from tools import EditFileTool, ReadFileTool, Tool, ToolError, ToolRegistry
from tools.edit_file import BY_LINE, BY_TEXT, SUMMARY_LINES


class EditFileTestBase(unittest.TestCase):
    """一个临时项目目录 + 一个工具。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.tool = EditFileTool(root=self.root)

    def write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        # newline="" 是必须的：Windows 上文本模式会把 \n 翻译成 \r\n，
        # 那样子写出来的 LF 文件根本不存在，测的就不是真实情况了
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding, newline="")
        return path

    def write_bytes(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def bytes_of(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def text_of(self, name: str, encoding: str = "utf-8") -> str:
        return self.bytes_of(name).decode(encoding)

    def edit(self, **kwargs) -> str:
        return self.tool.run(**kwargs)

    def refuse(self, **kwargs) -> str:
        """改不成时把报错文字取出来，顺便确认它抛的是 ToolError。"""
        with self.assertRaises(ToolError) as caught:
            self.edit(**kwargs)
        return str(caught.exception)

    def names(self) -> list[str]:
        """目录里现在有哪些文件。用来盯临时文件有没有留下。"""
        return sorted(item.name for item in self.root.rglob("*") if item.is_file())


class ByTextTest(EditFileTestBase):
    """按原文替换。"""

    def setUp(self):
        super().setUp()
        self.write("a.py", "def add(a, b):\n    return a + b\n")

    def test_a_unique_match_is_replaced(self):
        self.edit(path="a.py", old_text="return a + b", new_text="return a * b")

        self.assertEqual(
            self.text_of("a.py"), "def add(a, b):\n    return a * b\n"
        )

    def test_only_the_matched_place_changes(self):
        # 改的是中间那一小段，两边的内容（包括缩进和注释）都得原样
        self.write("b.py", "value = 1  # 这里是一\nnext = 2  # 这里是二\n")

        self.edit(path="b.py", old_text="1", new_text="100")

        self.assertEqual(
            self.text_of("b.py"), "value = 100  # 这里是一\nnext = 2  # 这里是二\n"
        )

    def test_a_multi_line_old_text_is_replaced(self):
        self.edit(
            path="a.py",
            old_text="def add(a, b):\n    return a + b\n",
            new_text="def add(a, b):\n    return a - b\n",
        )

        self.assertEqual(
            self.text_of("a.py"), "def add(a, b):\n    return a - b\n"
        )

    def test_an_empty_new_text_deletes_the_text(self):
        self.edit(path="a.py", old_text="    return a + b\n", new_text="")

        self.assertEqual(self.text_of("a.py"), "def add(a, b):\n")

    def test_zero_matches_is_refused(self):
        message = self.refuse(
            path="a.py", old_text="return a - b", new_text="return a * b"
        )

        self.assertIn("找不到这段原文", message)
        self.assertEqual(self.text_of("a.py"), "def add(a, b):\n    return a + b\n")

    def test_the_zero_match_message_says_where_to_copy_from(self):
        # 模型最常犯的错是把 read_file 的行号前缀一起抄进来，报错里得点这一句
        message = self.refuse(
            path="a.py", old_text="2|    return a + b", new_text="x"
        )

        self.assertIn("read_file", message)
        self.assertIn("不要抄进去", message)

    def test_several_matches_is_refused(self):
        self.write("c.py", "x = 1\ny = 2\nx = 3\n")

        message = self.refuse(path="c.py", old_text="x = ", new_text="z = ")

        self.assertIn("出现了 2 次", message)
        # 报出是哪几行，模型才知道要带多少上下文
        self.assertIn("第 1、3 行", message)
        self.assertEqual(self.text_of("c.py"), "x = 1\ny = 2\nx = 3\n")

    def test_the_refusal_points_at_the_other_mode(self):
        self.write("c.py", "x = 1\nx = 2\n")

        message = self.refuse(path="c.py", old_text="x = ", new_text="z = ")

        self.assertIn("start_line", message)

    def test_an_empty_old_text_is_refused(self):
        message = self.refuse(path="a.py", old_text="", new_text="x")

        self.assertIn("不能是空字符串", message)
        self.assertEqual(self.text_of("a.py"), "def add(a, b):\n    return a + b\n")

    def test_both_modes_at_once_is_refused(self):
        message = self.refuse(
            path="a.py", old_text="a", new_text="b", start_line=1
        )

        self.assertIn("只能用一种", message)
        self.assertEqual(self.text_of("a.py"), "def add(a, b):\n    return a + b\n")

    def test_neither_mode_is_refused(self):
        message = self.refuse(path="a.py", new_text="b")

        self.assertIn("得说清楚改哪儿", message)

    def test_the_result_says_where_it_changed(self):
        result = self.edit(
            path="a.py", old_text="    return a + b", new_text="    return a * b"
        )

        self.assertIn("结果：成功", result)
        self.assertIn("文件：a.py", result)
        self.assertIn("位置：第 2 行（原文替换，命中 1 处）", result)
        self.assertIn("修改前：\n      return a + b", result)
        self.assertIn("修改后：\n      return a * b", result)

    def test_a_multi_line_match_reports_both_ends(self):
        result = self.edit(
            path="a.py",
            old_text="def add(a, b):\n    return a + b",
            new_text="def add(a, b):\n    return a + b\n",
        )

        self.assertIn("位置：第 1–2 行", result)


class ByLineTest(EditFileTestBase):
    """按行号替换。行号必须和 read_file 是同一套。"""

    def setUp(self):
        super().setUp()
        self.write("a.py", "one\ntwo\nthree\nfour\n")

    def test_one_line_is_replaced(self):
        self.edit(path="a.py", start_line=2, new_text="TWO")

        self.assertEqual(self.text_of("a.py"), "one\nTWO\nthree\nfour\n")

    def test_a_range_of_lines_is_replaced(self):
        self.edit(path="a.py", start_line=2, end_line=3, new_text="MIDDLE")

        self.assertEqual(self.text_of("a.py"), "one\nMIDDLE\nfour\n")

    def test_a_range_can_become_several_lines(self):
        self.edit(path="a.py", start_line=2, end_line=2, new_text="x\ny\nz")

        self.assertEqual(self.text_of("a.py"), "one\nx\ny\nz\nthree\nfour\n")

    def test_end_line_defaults_to_start_line(self):
        self.edit(path="a.py", start_line=4, new_text="FOUR")

        self.assertEqual(self.text_of("a.py"), "one\ntwo\nthree\nFOUR\n")

    def test_the_first_line_can_be_replaced(self):
        self.edit(path="a.py", start_line=1, new_text="ONE")

        self.assertEqual(self.text_of("a.py"), "ONE\ntwo\nthree\nfour\n")

    def test_the_last_line_can_be_replaced(self):
        self.edit(path="a.py", start_line=4, new_text="FOUR")

        self.assertEqual(self.text_of("a.py").split("\n")[3], "FOUR")

    def test_an_empty_new_text_deletes_the_lines(self):
        # 「删掉这几行」就该是没了，不是留下一行空白
        self.edit(path="a.py", start_line=2, end_line=3, new_text="")

        self.assertEqual(self.text_of("a.py"), "one\nfour\n")

    def test_deleting_the_last_line_takes_its_newline(self):
        self.edit(path="a.py", start_line=4, new_text="")

        self.assertEqual(self.text_of("a.py"), "one\ntwo\nthree\n")

    def test_a_missing_trailing_newline_stays_missing(self):
        self.write("b.py", "one\ntwo")

        self.edit(path="b.py", start_line=2, new_text="TWO")

        self.assertEqual(self.text_of("b.py"), "one\nTWO")

    def test_a_trailing_newline_is_not_added(self):
        self.write("b.py", "one\ntwo")

        self.edit(path="b.py", start_line=1, new_text="ONE")

        self.assertEqual(self.text_of("b.py"), "ONE\ntwo")

    def test_a_line_beyond_the_end_is_refused(self):
        message = self.refuse(path="a.py", start_line=5, new_text="x")

        self.assertIn("行号超出范围", message)
        self.assertIn("一共 4 行", message)
        self.assertEqual(self.text_of("a.py"), "one\ntwo\nthree\nfour\n")

    def test_an_end_beyond_the_end_is_refused(self):
        message = self.refuse(path="a.py", start_line=3, end_line=9, new_text="x")

        self.assertIn("行号超出范围", message)
        self.assertEqual(self.text_of("a.py"), "one\ntwo\nthree\nfour\n")

    def test_a_reversed_range_is_refused(self):
        message = self.refuse(path="a.py", start_line=3, end_line=1, new_text="x")

        self.assertIn("还小", message)

    def test_line_zero_is_refused(self):
        message = self.refuse(path="a.py", start_line=0, new_text="x")

        self.assertIn("从 1 开始", message)

    def test_a_negative_line_is_refused(self):
        message = self.refuse(path="a.py", start_line=-1, new_text="x")

        self.assertIn("从 1 开始", message)

    def test_end_line_without_start_line_is_refused(self):
        message = self.refuse(path="a.py", end_line=2, new_text="x")

        self.assertIn("却没给 start_line", message)

    def test_a_line_that_is_not_a_number_is_refused(self):
        # run() 那道类型校验先挡下来
        with self.assertRaises(ToolError):
            self.edit(path="a.py", start_line="2", new_text="x")

    def test_execute_checks_the_type_by_itself(self):
        # 直接调 execute() 也不能中招——"3" 拿去比大小会抛 TypeError，
        # 那是个 Python 错误，不是给模型看的报错
        with self.assertRaises(ToolError) as caught:
            self.tool.execute(path="a.py", new_text="x", start_line="3")

        self.assertIn("得是整数", str(caught.exception))

    def test_true_is_not_a_line_number(self):
        # True 也是 int：不挡住的话它会变成「第 1 行」
        with self.assertRaises(ToolError) as caught:
            self.tool.execute(path="a.py", new_text="x", start_line=True)

        self.assertIn("得是整数", str(caught.exception))

    def test_line_numbers_agree_with_read_file(self):
        # 这条是两种改法共用的地基：行号必须是 read_file 那一套
        self.edit(path="a.py", start_line=3, end_line=3, new_text="THREE")

        shown = ReadFileTool(root=self.root).execute("a.py")

        self.assertIn("3|THREE", shown)
        self.assertIn("4|four", shown)


class PreserveTest(EditFileTestBase):
    """改完以后，除了那一处，文件应该和原来一模一样。"""

    def test_lf_stays_lf(self):
        self.write("a.py", "one\ntwo\nthree\n")

        self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertEqual(self.bytes_of("a.py"), b"one\nTWO\nthree\n")
        self.assertNotIn(b"\r", self.bytes_of("a.py"))

    def test_crlf_stays_crlf(self):
        self.write("a.cpp", "int main() {\r\n    int x = 1;\r\n}\r\n")

        self.edit(path="a.cpp", old_text="int x = 1;", new_text="int x = 2;")

        self.assertEqual(
            self.bytes_of("a.cpp"),
            b"int main() {\r\n    int x = 2;\r\n}\r\n",
        )

    def test_a_crlf_file_is_matched_with_lf_text(self):
        # 模型是从 read_file 抄的原文，那边 \r\n 已经被抹成 \n 了。
        # 不替它换算一下的话，这种文件一个字都匹配不上
        self.write("a.cpp", "int main() {\r\n    int x = 1;\r\n}\r\n")

        self.edit(
            path="a.cpp",
            old_text="int main() {\n    int x = 1;\n}\n",
            new_text="int main() {\n    return 0;\n}\n",
        )

        self.assertEqual(
            self.bytes_of("a.cpp"),
            b"int main() {\r\n    return 0;\r\n}\r\n",
        )

    def test_a_crlf_file_can_be_edited_by_line(self):
        self.write("a.cpp", "one\r\ntwo\r\nthree\r\n")

        self.edit(path="a.cpp", start_line=2, end_line=3, new_text="MIDDLE")

        self.assertEqual(self.bytes_of("a.cpp"), b"one\r\nMIDDLE\r\n")

    def test_the_line_numbers_of_a_crlf_file_are_right(self):
        # 行号不能把 \r 算进去，也不能少算一行
        self.write("a.cpp", "one\r\ntwo\r\nthree\r\n")

        result = self.edit(path="a.cpp", start_line=3, new_text="THREE")

        self.assertIn("位置：第 3 行", result)
        self.assertEqual(self.bytes_of("a.cpp"), b"one\r\ntwo\r\nTHREE\r\n")

    def test_a_utf8_bom_is_kept(self):
        self.write_bytes("bom.py", "\ufeffprint(1)\n".encode("utf-8-sig"))

        self.edit(path="bom.py", old_text="print(1)", new_text="print(2)")

        self.assertEqual(
            self.bytes_of("bom.py"), "\ufeffprint(2)\n".encode("utf-8-sig")
        )

    def test_no_bom_is_added_to_a_plain_utf8_file(self):
        # 真踩过：解 UTF-8 用的是 utf-8-sig，写回去照着这个名字写，
        # 于是每个改过的文件开头都多出三个字节的 BOM。编辑器和 git diff
        # 都看不出来，但编译器会把它当成源码的第一个字符。
        self.write("a.py", "one\ntwo\n")

        self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertEqual(self.bytes_of("a.py"), b"one\nTWO\n")
        self.assertFalse(self.bytes_of("a.py").startswith(b"\xef\xbb\xbf"))

    def test_a_gbk_file_stays_gbk(self):
        self.write("gbk.py", "# 中文注释\nx = 1\n", encoding="gbk")

        self.edit(path="gbk.py", old_text="x = 1", new_text="x = 2")

        self.assertEqual(
            self.bytes_of("gbk.py"), "# 中文注释\nx = 2\n".encode("gbk")
        )

    def test_a_gbk_file_can_be_edited_by_line(self):
        self.write("gbk.py", "# 中文注释\nx = 1\n", encoding="gbk")

        self.edit(path="gbk.py", start_line=1, new_text="# 换了个注释")

        self.assertEqual(
            self.bytes_of("gbk.py"), "# 换了个注释\nx = 1\n".encode("gbk")
        )

    def test_nothing_else_moves(self):
        # 一整个文件改一个字符，逐字节比：除了那一个字符，别处全得一模一样
        before = "".join(f"line {n}\n" for n in range(1, 51))
        self.write("big.py", before)

        self.edit(path="big.py", old_text="line 30\n", new_text="LINE 30\n")

        self.assertEqual(
            self.text_of("big.py"), before.replace("line 30\n", "LINE 30\n")
        )

    def test_a_file_without_a_trailing_newline_keeps_it_that_way(self):
        self.write("a.py", "one\ntwo")

        self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertEqual(self.text_of("a.py"), "one\nTWO")

    def test_an_empty_line_can_be_a_target(self):
        self.write("a.py", "one\n\nthree\n")

        result = self.edit(path="a.py", start_line=2, new_text="two")

        self.assertEqual(self.text_of("a.py"), "one\ntwo\nthree\n")
        self.assertIn("修改前：\n  （空）", result)


class AtomicTest(EditFileTestBase):
    """写坏的一半不能留在盘上。"""

    CONTENT = "one\ntwo\nthree\n"

    def setUp(self):
        super().setUp()
        self.write("a.py", self.CONTENT)

    def test_a_failed_replace_leaves_the_file_alone(self):
        with patch("tools.edit_file.os.replace", side_effect=OSError("磁盘满了")):
            message = self.refuse(path="a.py", old_text="two", new_text="TWO")

        self.assertIn("原文件没动", message)
        self.assertEqual(self.text_of("a.py"), self.CONTENT)

    def test_a_failed_replace_leaves_no_temp_file(self):
        with patch("tools.edit_file.os.replace", side_effect=OSError("磁盘满了")):
            self.refuse(path="a.py", old_text="two", new_text="TWO")

        # 目录里应该还是只有那一个文件
        self.assertEqual(self.names(), ["a.py"])

    def test_a_failed_write_is_a_tool_error(self):
        # 抛 OSError 的话模型收到的是一句 Python 异常，不是「怎么改」
        with patch("tools.edit_file.os.replace", side_effect=OSError("磁盘满了")):
            with self.assertRaises(ToolError):
                self.edit(path="a.py", old_text="two", new_text="TWO")

    def test_a_failed_write_does_not_break_later_edits(self):
        with patch("tools.edit_file.os.replace", side_effect=OSError("磁盘满了")):
            self.refuse(path="a.py", old_text="two", new_text="TWO")

        self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertEqual(self.text_of("a.py"), "one\nTWO\nthree\n")

    def test_the_temp_file_sits_next_to_the_target(self):
        # 和目标是同一个目录，os.replace 才是同一个卷上的原子换名；
        # 丢到系统临时目录去的化就成了跨卷复制，中间态会露出来
        real = os.replace
        seen: list[Path] = []

        def watch(source, destination):
            seen.append(Path(source))
            return real(source, destination)

        with patch("tools.edit_file.os.replace", side_effect=watch):
            self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].parent, self.root)
        self.assertNotEqual(seen[0], self.root / "a.py")

    def test_text_that_cannot_be_encoded_is_refused(self):
        self.write("gbk.py", "# 中文\nx = 1\n", encoding="gbk")

        message = self.refuse(
            path="gbk.py", old_text="x = 1", new_text="x = 1  # 🙂"
        )

        self.assertIn("gbk", message)
        self.assertIn("一个字节都没动", message)
        self.assertEqual(self.bytes_of("gbk.py"), "# 中文\nx = 1\n".encode("gbk"))

    def test_nothing_is_written_when_the_match_is_refused(self):
        # 改之前的校验没过，就不该留下任何痕迹
        self.refuse(path="a.py", old_text="nope", new_text="x")

        self.assertEqual(self.names(), ["a.py"])
        self.assertEqual(self.text_of("a.py"), self.CONTENT)


class GuardTest(EditFileTestBase):
    """路径与文件本身的准入。"""

    def test_a_path_outside_the_root_is_refused(self):
        message = self.refuse(
            path="../outside.py", old_text="a", new_text="b"
        )

        self.assertIn("超出允许范围", message)

    def test_an_absolute_path_outside_the_root_is_refused(self):
        outside = PROJECT_ROOT / "config.py"

        message = self.refuse(path=str(outside), old_text="a", new_text="b")

        self.assertIn("超出允许范围", message)

    def test_a_sneaky_path_is_refused(self):
        # 中间夹一层 .. 也算跑出去了
        message = self.refuse(
            path="sub/../../outside.py", old_text="a", new_text="b"
        )

        self.assertIn("超出允许范围", message)

    def test_a_missing_file_is_refused(self):
        message = self.refuse(path="nope.py", old_text="a", new_text="b")

        self.assertIn("文件不存在", message)
        self.assertIn("不新建", message)

    def test_nothing_is_created_by_a_refused_call(self):
        self.refuse(path="nope.py", old_text="a", new_text="b")

        self.assertEqual(self.names(), [])

    def test_a_directory_is_refused(self):
        (self.root / "sub").mkdir()

        message = self.refuse(path="sub", old_text="a", new_text="b")

        self.assertIn("是目录", message)

    def test_a_binary_file_is_refused(self):
        self.write_bytes("a.bin", b"\x00\x01\x02binary")

        message = self.refuse(path="a.bin", old_text="a", new_text="b")

        self.assertIn("NUL", message)

    def test_a_file_that_is_too_big_is_refused(self):
        self.write("big.py", "x\n" * 200_000)

        message = self.refuse(path="big.py", old_text="x", new_text="y")

        self.assertIn("超过", message)
        self.assertIn("KB", message)

    def test_a_name_with_spaces_is_fine(self):
        # 命令类工具要过字符白名单（空格、中文都不行），这个工具不拼命令，
        # 走的是文件系统，所以不该有那个限制
        self.write("my notes.txt", "hello\n")

        self.edit(path="my notes.txt", old_text="hello", new_text="bye")

        self.assertEqual(self.text_of("my notes.txt"), "bye\n")

    def test_a_subdirectory_file_is_fine(self):
        self.write("src/deep/a.py", "x = 1\n")

        self.edit(path="src/deep/a.py", old_text="x = 1", new_text="x = 2")

        self.assertEqual(self.text_of("src/deep/a.py"), "x = 2\n")


class ResultTest(EditFileTestBase):
    """返回的那段文字，字段得齐、得能直接看懂。"""

    def setUp(self):
        super().setUp()
        self.write("a.py", "one\ntwo\nthree\n")

    def test_all_four_fields_are_there(self):
        result = self.edit(path="a.py", old_text="two", new_text="TWO")

        for label in ("结果：", "文件：", "位置：", "修改前：", "修改后："):
            with self.subTest(label=label):
                self.assertIn(label, result)

    def test_the_path_is_relative_to_the_root(self):
        result = self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertIn("文件：a.py", result)
        self.assertNotIn(str(self.root), result)

    def test_a_nested_path_uses_forward_slashes(self):
        self.write("src/a.py", "x\n")

        result = self.edit(path="src/a.py", old_text="x", new_text="y")

        self.assertIn("文件：src/a.py", result)

    def test_the_file_on_disk_really_changed(self):
        # 说成功就得真改到了，不能只是嘴上说
        self.edit(path="a.py", old_text="two", new_text="TWO")

        self.assertEqual(self.text_of("a.py"), "one\nTWO\nthree\n")

    def test_a_long_change_is_summarized(self):
        long = "\n".join(f"line {n}" for n in range(1, SUMMARY_LINES + 4))
        self.write("b.py", long + "\n")

        result = self.edit(path="b.py", start_line=1, end_line=9, new_text=long)

        self.assertIn("后面还有 3 行", result)
        self.assertNotIn("line 9", result)

    def test_a_deletion_shows_an_empty_after(self):
        result = self.edit(path="a.py", old_text="two\n", new_text="")

        self.assertIn("修改后：\n  （空）", result)

    def test_the_change_object_is_structured(self):
        # 「结构化」的那一层：字段是取得到的，不是从一段文字里抠出来的
        change = self.tool._by_line("one\ntwo\nthree\n", 2, 3, "MID", "a.py")

        self.assertEqual(change.kind, BY_LINE)
        self.assertEqual(change.line, 2)
        self.assertEqual(change.end_line, 3)
        self.assertEqual(change.before, "two\nthree")
        self.assertEqual(change.after, "MID")
        self.assertEqual(change.text, "one\nMID\n")

    def test_the_text_change_carries_its_hits(self):
        change = self.tool._by_text("one\ntwo\n", "two", "TWO", "a.py")

        self.assertEqual(change.kind, BY_TEXT)
        self.assertEqual(change.hits, 1)
        self.assertEqual(change.line, 2)


class ShapeTest(unittest.TestCase):
    def test_it_is_a_tool(self):
        self.assertIsInstance(EditFileTool(root=PROJECT_ROOT), Tool)
        self.assertEqual(EditFileTool.name, "edit_file")

    def test_default_root_is_project_root(self):
        self.assertEqual(EditFileTool().root, PROJECT_ROOT)

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            EditFileTool(root=Path("C:/"))

    def test_schema_shape(self):
        parameters = EditFileTool(root=PROJECT_ROOT).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(
            list(parameters["properties"]),
            ["path", "new_text", "old_text", "start_line", "end_line"],
        )
        self.assertEqual(parameters["required"], ["path", "new_text"])
        self.assertEqual(
            parameters["properties"]["old_text"]["type"], "string"
        )
        self.assertEqual(
            parameters["properties"]["start_line"]["type"], "integer"
        )

    def test_there_is_no_command_parameter(self):
        # 命令得由工具自己拼，模型给不了
        self.assertNotIn("command", EditFileTool.parameters)

    def test_the_description_says_the_match_must_be_unique(self):
        self.assertIn("只出现一次", EditFileTool.description)

    def test_the_description_says_it_does_not_create_files(self):
        self.assertIn("不会新建", EditFileTool.description)

    def test_the_description_names_both_modes(self):
        for name in ("old_text", "start_line"):
            with self.subTest(parameter=name):
                self.assertIn(name, EditFileTool.description)


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


class EditFileAgentTest(EditFileTestBase):
    """走一遍完整的 Agent 调用链：模型要求改代码 → 结果回灌。"""

    def setUp(self):
        super().setUp()
        self.write("a.py", "def add(a, b):\n    return a + b\n")

    def build_agent(self, replies) -> Agent:
        return Agent(
            client=StubClient(replies),
            tools=ToolRegistry([EditFileTool(root=self.root)]),
        )

    def test_agent_can_edit_a_file(self):
        agent = self.build_agent(
            [
                tool_call_reply(
                    "edit_file",
                    json.dumps(
                        {
                            "path": "a.py",
                            "old_text": "return a + b",
                            "new_text": "return a - b",
                        }
                    ),
                ),
                final_reply("改好了"),
            ]
        )

        answer = agent.run("把加法改成减法")

        self.assertEqual(answer, "改好了")
        self.assertIn("结果：成功", agent.messages[3]["content"])
        self.assertEqual(
            self.text_of("a.py"), "def add(a, b):\n    return a - b\n"
        )

    def test_a_refused_edit_does_not_break_the_turn(self):
        agent = self.build_agent(
            [
                tool_call_reply(
                    "edit_file",
                    json.dumps(
                        {"path": "a.py", "old_text": "不存在的原文", "new_text": "x"}
                    ),
                ),
                final_reply("没找到那段原文"),
            ]
        )

        answer = agent.run("改一下")

        self.assertEqual(answer, "没找到那段原文")
        self.assertIn("找不到这段原文", agent.messages[3]["content"])
        # 工具调用本身失败了，但这一轮没崩
        self.assertFalse(agent.get_trace()[0].success)
        self.assertEqual(
            self.text_of("a.py"), "def add(a, b):\n    return a + b\n"
        )

    def test_the_call_is_recorded_in_the_trace(self):
        agent = self.build_agent(
            [
                tool_call_reply(
                    "edit_file",
                    json.dumps(
                        {"path": "a.py", "start_line": 2, "new_text": "    pass"}
                    ),
                ),
                final_reply("好了"),
            ]
        )

        agent.run("改第 2 行")

        events = agent.get_trace()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tool, "edit_file")
        self.assertTrue(events[0].success)
        self.assertEqual(self.text_of("a.py"), "def add(a, b):\n    pass\n")

    def test_it_is_registered_in_main(self):
        # main.build_agent() 会去读 API Key，这里把客户端换掉再装配
        import main

        with patch("agent.agent.LLMClient", StubClient):
            agent = main.build_agent()

        self.assertIn("edit_file", [tool.name for tool in agent.tools.list()])


if __name__ == "__main__":
    unittest.main()
