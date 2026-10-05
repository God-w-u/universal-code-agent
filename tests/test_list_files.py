"""ListFilesTool 测试：正常目录、空目录、非法路径、越界与忽略规则。

全部在临时目录里跑，不联网、不碰真实项目文件。
"""

import os
import subprocess
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


class ListFilesToolSensitiveTest(ListFilesToolTestBase):
    """密钥 / 凭据一类文件不进遍历（13 阶段回归）。

    它们本来就在 root 里面，`resolve_in_root` 拦不住也不该拦；靠的是
    `tools/paths.py` 的 is_sensitive_file——按**文件名**整条排掉。要看配置
    长什么样有 `.env.example`（值全是空的，本来就该给人看）。
    """

    SECRETS = (
        ".env",
        ".env.local",
        ".env.production",
        "server.pem",
        "id_rsa",
        "id_rsa.pub",
        "deploy.key",
        "cert.p12",
        "keystore.jks",
        "credentials.json",
        ".netrc",
        ".pgpass",
        ".htpasswd",
        "my.ppk",
    )

    def test_secret_files_are_not_listed(self):
        self.make("main.py")
        for name in self.SECRETS:
            self.make(name, "OPENAI_API_KEY=sk-真钥匙不该出现在这里")

        self.assertEqual(self.lines(), ["main.py"])

    def test_nested_secret_files_are_not_listed_either(self):
        # 光挡住根目录那几个没用：真项目里密钥常常躺在子目录里
        self.make("app/config.py")
        self.make("deploy/.env", "TOKEN=真令牌")
        self.make("deploy/id_rsa", "真私钥")
        self.make("deploy/certs/site.pem", "真证书")

        self.assertEqual(self.lines(), ["app/config.py"])

    def test_the_public_template_is_still_listed(self):
        # .env.example 是公开模板（值全空），属于项目该给人看的一部分；
        # 它要是被 `.env.*` 一起吃掉，模型就再也看不到「有这个变量」了
        self.make("main.py")
        self.make(".env.example", "OPENAI_API_KEY=\n")

        self.assertEqual(self.lines(), [".env.example", "main.py"])

    def test_a_name_that_merely_resembles_one_is_kept(self):
        # 白名单式的全名 / 通配匹配，不做子串匹配：这几种都不是密钥文件
        for name in ("environment.py", "keys.py", "my.env.example.txt", "env"):
            self.make(name)

        self.assertEqual(
            self.lines(), ["env", "environment.py", "keys.py", "my.env.example.txt"]
        )

    def test_the_secret_content_never_shows_up_anywhere(self):
        self.make("main.py", "print('你好')\n")
        self.make(".env", "OPENAI_API_KEY=sk-真钥匙不该出现在这里")

        result = self.tool().run()

        self.assertNotIn("sk-", result)  # 连值都没露出来


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

    def test_a_junction_is_not_followed_either(self):
        """Windows 的 junction 也是「链出去的门」，不能当普通目录跟进去。

        junction 和符号链接是两回事：建它不需要管理员权限（`mklink /J`），
        但 `Path.is_symlink()` 对它返回 **False**（走的是 reparse point 的
        另一个标记）。只看 is_symlink() 就会把它当成普通目录，跟着它列出
        root 外面的整个目录树。
        """
        self.make("real/a.py")

        with tempfile.TemporaryDirectory() as outside:
            Path(outside, "secret.py").write_text("x", encoding="utf-8")
            link = self.root / "junction_to_outside"
            if not _make_junction(link, Path(outside)):
                self.skipTest("当前环境不支持创建 junction")
            self.assertFalse(link.is_symlink())  # 前提：它不是符号链接

            self.assertEqual(self.lines(), ["real/a.py"])


def _make_junction(link: Path, target: Path) -> bool:
    """在 link 处建一个指向 target 的 junction；建不成返回 False。

    只有在 Windows 上才可能成功，其它平台（或没有权限）直接 False——
    调用方 skipTest，和符号链接那条用例的约定一致。
    """
    if os.name != "nt":
        return False
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and link.exists()


if __name__ == "__main__":
    unittest.main()
