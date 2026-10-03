"""ListFilesTool 测试：正常目录、空目录、非法路径、越界与忽略规则。

全部在临时目录里跑，不联网、不碰真实项目文件。
"""

import tempfile
import unittest
from pathlib import Path

from config import PROJECT_ROOT
from tools import ListFilesTool, Tool, ToolError


class ListFilesToolTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str = "x") -> Path:
        """在 root 下建一个文件，父目录自动创建。"""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def mkdir(self, relative: str) -> Path:
        path = self.root / relative
        path.mkdir(parents=True, exist_ok=True)
        return path

    def tool(self, **kwargs) -> ListFilesTool:
        return ListFilesTool(root=self.root, **kwargs)

    def lines(self, **kwargs) -> list[str]:
        return self.tool().run(**kwargs).splitlines()


class ListFilesToolBasicTest(ListFilesToolTestBase):
    def test_lists_files_recursively(self):
        self.make("main.py")
        self.make("agent/agent.py")
        self.make("agent/llm/client.py")

        self.assertEqual(
            self.lines(path="."),
            ["agent/agent.py", "agent/llm/client.py", "main.py"],
        )

    def test_path_defaults_to_root(self):
        self.make("main.py")

        self.assertEqual(self.lines(), ["main.py"])

    def test_listing_a_subdirectory(self):
        self.make("main.py")
        self.make("agent/agent.py")

        self.assertEqual(self.lines(path="agent"), ["agent/agent.py"])

    def test_result_is_relative_to_root_not_to_the_subdirectory(self):
        # 相对 root 才能直接交给后续的读文件工具
        self.make("agent/tools/base.py")

        self.assertEqual(self.lines(path="agent"), ["agent/tools/base.py"])

    def test_separator_is_always_forward_slash(self):
        self.make("a/b/c.py")

        self.assertEqual(self.lines(), ["a/b/c.py"])

    def test_result_is_sorted(self):
        for name in ("b.py", "a.py", "c.py"):
            self.make(name)

        self.assertEqual(self.lines(), ["a.py", "b.py", "c.py"])

    def test_empty_directories_are_not_listed(self):
        self.make("a.py")
        self.mkdir("empty")

        self.assertEqual(self.lines(), ["a.py"])

    def test_it_is_a_tool(self):
        tool = self.tool()

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "list_files")
        self.assertTrue(tool.description)

    def test_path_is_optional_in_schema(self):
        parameters = self.tool().to_schema()["function"]["parameters"]

        self.assertEqual(list(parameters["properties"]), ["path"])
        self.assertNotIn("required", parameters)  # 全可选时这个键应当缺席


class ListFilesToolEmptyTest(ListFilesToolTestBase):
    def test_empty_root(self):
        self.assertIn("没有文件", self.tool().run())

    def test_empty_subdirectory(self):
        self.mkdir("empty")

        self.assertIn("没有文件", self.tool().run(path="empty"))

    def test_directory_containing_only_ignored_dirs(self):
        self.mkdir(".git")
        self.mkdir("__pycache__")

        self.assertIn("没有文件", self.tool().run())


class ListFilesToolIgnoreTest(ListFilesToolTestBase):
    def test_noise_directories_are_skipped(self):
        self.make("main.py")
        for noise in (".git", ".venv", "__pycache__", "node_modules", "dist"):
            self.make(f"{noise}/junk.py")

        self.assertEqual(self.lines(), ["main.py"])

    def test_nested_noise_directory_is_skipped(self):
        self.make("agent/agent.py")
        self.make("agent/__pycache__/agent.cpython-311.pyc")

        self.assertEqual(self.lines(), ["agent/agent.py"])

    def test_a_directory_merely_resembling_a_noise_name_is_kept(self):
        # 只按完整目录名匹配，不做前缀/子串匹配，避免误杀
        self.make("build_tools/make.py")
        self.make("my_venv_notes/note.md")

        self.assertEqual(
            self.lines(), ["build_tools/make.py", "my_venv_notes/note.md"]
        )


class ListFilesToolBadPathTest(ListFilesToolTestBase):
    def test_missing_path_rejected(self):
        with self.assertRaises(ToolError):
            self.tool().run(path="nope")

    def test_file_instead_of_directory_rejected(self):
        self.make("main.py")

        with self.assertRaises(ToolError):
            self.tool().run(path="main.py")


class ListFilesToolBoundaryTest(ListFilesToolTestBase):
    def test_parent_traversal_rejected(self):
        for path in ("..", "../..", "agent/../../.."):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    self.tool().run(path=path)

    def test_absolute_path_outside_root_rejected(self):
        with self.assertRaises(ToolError):
            self.tool().run(path=str(self.root.parent))

    def test_system_directories_rejected(self):
        for path in (
            "C:/Windows",
            "C:/Windows/System32",
            "/etc",
            "/etc/passwd",
            "/proc",
        ):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    self.tool().run(path=path)

    def test_traversal_error_message_names_the_root(self):
        with self.assertRaises(ToolError) as caught:
            self.tool().run(path="..")

        self.assertIn(str(self.root), str(caught.exception))


class ListFilesToolRootTest(unittest.TestCase):
    """root 本身也不允许设成危险位置——这是越界防护的前提。"""

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            ListFilesTool(root=Path("C:/"))

    def test_system_directory_as_root_rejected(self):
        for path in ("C:/Windows", "C:/Windows/System32", "C:/Program Files", "/etc"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    ListFilesTool(root=Path(path))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            ListFilesTool(root=Path("C:/definitely/not/a/real/dir/xyz"))

    def test_file_as_root_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file_path = Path(temp) / "a.txt"
            file_path.write_text("x", encoding="utf-8")

            with self.assertRaises(ToolError):
                ListFilesTool(root=file_path)

    def test_default_root_is_project_root(self):
        self.assertEqual(ListFilesTool().root, PROJECT_ROOT)

    def test_root_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            tool = ListFilesTool(root=Path(temp) / "." / ".." / Path(temp).name)

            self.assertEqual(tool.root, Path(temp).resolve())


class ListFilesToolRealProjectTest(unittest.TestCase):
    """直接列本项目，确认忽略规则真的把 .venv 这类目录挡掉了。"""

    def test_lists_this_project_without_noise(self):
        result = ListFilesTool().run()

        self.assertIn("agent/agent.py", result)
        self.assertIn("tools/list_files.py", result)
        self.assertIn("tests/test_list_files.py", result)
        self.assertNotIn(".venv", result)
        self.assertNotIn("__pycache__", result)

    def test_missing_root_is_reported_not_crashed(self):
        with self.assertRaises(ToolError):
            ListFilesTool().run(path="这个目录不存在")


class ListFilesToolLimitTest(ListFilesToolTestBase):
    def test_truncates_and_says_how_many_were_found(self):
        class TinyLimitTool(ListFilesTool):
            max_files = 3

        for index in range(5):
            self.make(f"f{index}.py")

        result = TinyLimitTool(root=self.root).run()

        lines = result.splitlines()
        self.assertEqual(lines[:3], ["f0.py", "f1.py", "f2.py"])
        self.assertIn("共 5 个文件", lines[-1])

    def test_no_note_when_under_the_limit(self):
        self.make("only.py")

        self.assertEqual(self.lines(), ["only.py"])


class ListFilesToolSymlinkTest(ListFilesToolTestBase):
    def test_symlinked_directory_is_not_followed(self):
        self.make("real/a.py")

        with tempfile.TemporaryDirectory() as outside:
            Path(outside, "secret.py").write_text("x", encoding="utf-8")
            link = self.root / "link_to_outside"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("当前环境不支持创建符号链接")

            # 链接指向 root 外面，如果不跳过就能顺着它列出系统目录
            self.assertEqual(self.lines(), ["real/a.py"])


if __name__ == "__main__":
    unittest.main()
