"""Project Index 测试：扫到什么、每一条记成什么样、哪些不收。

全部在临时目录里跑，不联网、不碰真实项目文件。语言识别、忽略目录、
二进制 / 超大 / 读不动的判定、root 边界各测一遍，最后拿本项目真扫一次。

「索引看到的文件和 list_files 看到的一模一样」这条单独钉着：两边共用
tools/tree.py 的遍历规则，谁哪天给索引另写一套，这条会红。
"""

import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

from config import PROJECT_ROOT
from index import (
    BINARY,
    INDEXED,
    TOO_LARGE,
    UNREADABLE,
    IndexedFile,
    ProjectIndex,
    language_of,
)
from index.languages import LANGUAGE_BY_SUFFIX, UNKNOWN_LANGUAGE
from tools import ListFilesTool, ToolError
from tools.textfile import read_bytes as real_read_bytes


class IndexTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str = "x", encoding: str = "utf-8") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline=""：换行原样写，别让 Windows 把 \n 变成 \r\n
        path.write_text(content, encoding=encoding, newline="")
        return path

    def make_bytes(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def mkdir(self, relative: str) -> Path:
        path = self.root / relative
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scan(self, **kwargs) -> ProjectIndex:
        """建一个索引并扫一遍，返回它。"""
        index = ProjectIndex(root=self.root, **kwargs)
        index.build()
        return index

    def scan_tiny(self) -> ProjectIndex:
        """上限缩到 32 字节的索引，专门用来测「太大不收」。"""

        class TinyIndex(ProjectIndex):
            max_file_bytes = 32

        index = TinyIndex(root=self.root)
        index.build()
        return index

    def by_path(self, index: ProjectIndex) -> dict[str, IndexedFile]:
        return {entry.path: entry for entry in index.entries}


# ---------------------------------------------------------------------------
# 一条记录里都记了些什么
# ---------------------------------------------------------------------------


class IndexedEntryTest(IndexTestBase):
    def test_a_text_file_is_indexed_with_all_its_facts(self):
        path = self.make("main.py", "print('你好')\n")

        entry = self.by_path(self.scan())["main.py"]

        self.assertEqual(entry.path, "main.py")
        self.assertEqual(entry.language, "python")
        self.assertEqual(entry.status, INDEXED)
        self.assertTrue(entry.indexed)
        raw = path.read_bytes()
        self.assertEqual(entry.size, len(raw))
        self.assertEqual(entry.mtime, path.stat().st_mtime)
        self.assertEqual(entry.digest, hashlib.sha256(raw).hexdigest()[:16])

    def test_size_counts_bytes_not_characters(self):
        content = "中文\n"  # 2 个汉字各 3 字节 + 换行
        self.make("note.py", content)

        entry = self.by_path(self.scan())["note.py"]

        self.assertEqual(entry.size, len(content.encode("utf-8")))
        self.assertNotEqual(entry.size, len(content))

    def test_paths_are_relative_and_use_forward_slashes(self):
        self.make("agent/llm/client.py")

        entry = self.scan().entries[0]

        self.assertEqual(entry.path, "agent/llm/client.py")
        self.assertFalse(Path(entry.path).is_absolute())

    def test_entries_are_sorted_by_path(self):
        for name in ("b.py", "a.py", "agent/z.py", "agent/a.py"):
            self.make(name)

        self.assertEqual(
            [entry.path for entry in self.scan().entries],
            ["a.py", "agent/a.py", "agent/z.py", "b.py"],
        )

    def test_the_same_content_gives_the_same_digest(self):
        self.make("one.py", "same")
        self.make("two.py", "same")

        entries = self.by_path(self.scan())

        self.assertEqual(entries["one.py"].digest, entries["two.py"].digest)

    def test_different_content_gives_a_different_digest(self):
        self.make("one.py", "same")
        self.make("two.py", "different")

        entries = self.by_path(self.scan())

        self.assertNotEqual(entries["one.py"].digest, entries["two.py"].digest)

    def test_an_entry_cannot_be_edited_afterwards(self):
        # 冻住的记录才敢直接交给后面的阶段用：谁拿到都不会把它改坏
        self.make("main.py")

        entry = self.scan().entries[0]

        with self.assertRaises(FrozenInstanceError):
            entry.status = BINARY

    def test_empty_project_has_no_entries(self):
        self.assertEqual(self.scan().entries, ())
        self.assertEqual(self.scan().files(), ())

    def test_before_build_the_index_is_empty(self):
        self.make("main.py")

        index = ProjectIndex(root=self.root)

        self.assertEqual(index.entries, ())  # 没扫就是没扫，不偷偷扫
        self.assertFalse(index.built)  # 空清单 ≠ 项目里没文件，这两件事分得开

    def test_built_turns_true_after_a_scan(self):
        self.make("main.py")

        index = self.scan()

        self.assertTrue(index.built)

    def test_files_returns_only_indexed_ones(self):
        self.make("good.py")
        self.make_bytes("bad.py", b"\x00\x01\x02")

        index = self.scan()

        self.assertEqual([entry.path for entry in index.files()], ["good.py"])
        self.assertEqual(len(index.entries), 2)


# ---------------------------------------------------------------------------
# 认语言
# ---------------------------------------------------------------------------


class LanguageTest(IndexTestBase):
    def test_python_files(self):
        for name in ("a.py", "a.pyi", "a.pyw"):
            self.make(name)

        languages = {entry.path: entry.language for entry in self.scan().entries}

        self.assertEqual(languages["a.py"], "python")
        self.assertEqual(languages["a.pyi"], "python")
        self.assertEqual(languages["a.pyw"], "python")

    def test_c_and_cpp_files_share_one_language(self):
        # C 和 C++ 合成一门：提取器本来也不分（tools/cpp.py 一套吃两种）
        for name in ("a.c", "a.h", "a.cc", "a.cpp", "a.hpp", "a.hxx"):
            self.make(name)

        languages = {entry.path: entry.language for entry in self.scan().entries}

        self.assertEqual(set(languages.values()), {"cpp"})

    def test_suffix_is_case_insensitive(self):
        # Windows 上 MAIN.PY 和 main.py 是同一类文件，不该一个认得一个不认得
        self.make("MAIN.PY")

        self.assertEqual(self.scan().entries[0].language, "python")

    def test_later_languages_are_recognized_by_name(self):
        # 这几个现在只是"认得出名字"：结构提取器还没有，索引先照记
        expected = {
            "a.java": "java",
            "a.go": "go",
            "a.rs": "rust",
            "a.js": "javascript",
            "a.jsx": "javascript",
            "a.ts": "typescript",
            "a.tsx": "typescript",
        }
        for name in expected:
            self.make(name)

        languages = {entry.path: entry.language for entry in self.scan().entries}

        self.assertEqual(languages, expected)

    def test_unknown_extensions_are_marked_unknown(self):
        for name in ("notes.txt", "README.md", "data.json", "Makefile", ".gitignore"):
            self.make(name)

        languages = {entry.path: entry.language for entry in self.scan().entries}

        self.assertEqual(set(languages.values()), {UNKNOWN_LANGUAGE})
        # 认不出语言不等于不收：它们还是文本，照样进索引
        self.assertEqual(len(self.scan().files()), 5)

    def test_language_of_answers_directly(self):
        self.assertEqual(language_of(Path("src/main.rs")), "rust")
        self.assertEqual(language_of(Path("a/b/HEADER.HPP")), "cpp")
        self.assertEqual(language_of(Path("no-extension")), UNKNOWN_LANGUAGE)

    def test_registered_suffixes_are_all_lowercase(self):
        # language_of 会把后缀转小写再查表，表里写着大写键就是永远查不到的死键
        for suffix in LANGUAGE_BY_SUFFIX:
            self.assertEqual(suffix, suffix.lower())

    def test_files_can_be_filtered_by_language(self):
        self.make("a.py")
        self.make("b.py")
        self.make("c.cpp")

        index = self.scan()

        self.assertEqual(
            [entry.path for entry in index.files("python")], ["a.py", "b.py"]
        )
        self.assertEqual([entry.path for entry in index.files("cpp")], ["c.cpp"])
        self.assertEqual(index.files("rust"), ())


# ---------------------------------------------------------------------------
# 哪些目录不看
# ---------------------------------------------------------------------------


class IgnoredDirectoryTest(IndexTestBase):
    def test_noise_directories_are_skipped(self):
        self.make("main.py")
        for noise in (".git", ".venv", "__pycache__", "node_modules", "build", "dist"):
            self.make(f"{noise}/junk.py")

        self.assertEqual([entry.path for entry in self.scan().entries], ["main.py"])

    def test_nested_noise_directory_is_skipped(self):
        self.make("agent/agent.py")
        self.make("agent/__pycache__/agent.cpython-311.pyc")

        self.assertEqual([entry.path for entry in self.scan().entries], ["agent/agent.py"])

    def test_a_directory_merely_resembling_a_noise_name_is_kept(self):
        self.make("build_tools/make.py")
        self.make("my_venv_notes/note.py")

        self.assertEqual(
            [entry.path for entry in self.scan().entries],
            ["build_tools/make.py", "my_venv_notes/note.py"],
        )

    def test_the_index_sees_exactly_what_list_files_sees(self):
        # 两边共用 tools/tree.py 的遍历规则——这里直接对起来看
        self.make("main.py")
        self.make("agent/deep/file.py")
        for noise in (".git", "node_modules", "build", "__pycache__"):
            self.make(f"{noise}/junk.py")

        from_list_files = set(ListFilesTool(root=self.root).run().splitlines())
        from_index = {entry.path for entry in self.scan().entries}

        self.assertEqual(from_index, from_list_files)


# ---------------------------------------------------------------------------
# 收不进来的那些，记一条写明为什么
# ---------------------------------------------------------------------------


class SkippedFileTest(IndexTestBase):
    def test_binary_file_is_recorded_as_binary(self):
        self.make_bytes("blob.py", b"text\x00more")

        entry = self.by_path(self.scan())["blob.py"]

        self.assertEqual(entry.status, BINARY)
        self.assertFalse(entry.indexed)
        self.assertIsNone(entry.digest)  # 不收的东西不算摘要
        self.assertEqual(entry.size, 9)  # 大小还是记着的
        self.assertEqual(entry.language, "python")  # 后缀认得出来就照记

    def test_utf16_is_binary_even_with_a_source_suffix(self):
        # UTF-16 的文件带 NUL 字节，read_file 读不了，索引也不收——
        # 两边用的是同一个判断（tools/textfile.py 的 decode_text）
        self.make_bytes("wide.py", "print(1)\n".encode("utf-16"))

        self.assertEqual(self.by_path(self.scan())["wide.py"].status, BINARY)

    def test_undecodable_bytes_are_binary(self):
        # 0xFF 不是合法的 UTF-8，也不是合法的 GBK 首字节
        self.make_bytes("weird.py", b"\xff\xff\xff")

        self.assertEqual(self.by_path(self.scan())["weird.py"].status, BINARY)

    def test_gbk_source_is_indexed(self):
        # 中文 Windows 上的 GBK 源码要能认（用 read_file 同一套解码）
        self.make("gbk.py", "# 中文注释\n", encoding="gbk")

        self.assertEqual(self.by_path(self.scan())["gbk.py"].status, INDEXED)

    def test_oversized_file_is_recorded_as_too_large(self):
        self.make("big.py", "x" * 100)

        index = self.scan_tiny()
        entry = self.by_path(index)["big.py"]

        self.assertEqual(entry.status, TOO_LARGE)
        self.assertFalse(entry.indexed)
        self.assertIsNone(entry.digest)
        self.assertEqual(entry.size, 100)
        self.assertEqual(index.files(), ())

    def test_a_file_exactly_at_the_limit_is_indexed(self):
        # 边界：正好等于上限要收，只超一个字节才不收
        self.make("edge.py", "x" * 32)

        entry = self.by_path(self.scan_tiny())["edge.py"]

        self.assertEqual(entry.status, INDEXED)
        self.assertEqual(entry.size, 32)

    def test_an_unreadable_file_is_recorded_not_crashed(self):
        self.make("locked.py")
        self.make("fine.py")

        def refuse(path, max_bytes):
            if path.name == "locked.py":
                raise OSError("被别的程序占着")
            return real_read_bytes(path, max_bytes)

        with mock.patch("index.project.read_bytes", side_effect=refuse):
            index = self.scan()

        entries = self.by_path(index)
        self.assertEqual(entries["locked.py"].status, UNREADABLE)
        self.assertIsNone(entries["locked.py"].digest)
        self.assertIsNotNone(entries["locked.py"].size)  # stat 过了，大小有
        # 一个文件读不动，别的照收
        self.assertEqual(entries["fine.py"].status, INDEXED)

    def test_a_file_that_vanishes_mid_scan_is_recorded(self):
        # 列出来之后、看它之前就没了（真扫的时候有人在删就是这个样子）
        ghost = self.root / "ghost.py"

        with mock.patch("index.project.iter_files", return_value=iter([ghost])):
            index = self.scan()

        entry = index.entries[0]
        self.assertEqual(entry.path, "ghost.py")
        self.assertEqual(entry.status, UNREADABLE)
        self.assertIsNone(entry.size)  # stat 都没成功，大小和修改时间都没有
        self.assertIsNone(entry.mtime)

    def test_skipped_files_are_not_in_files(self):
        self.make("good.py")
        self.make_bytes("blob.py", b"\x00")
        self.make("big.py", "x" * 100)

        index = self.scan_tiny()

        self.assertEqual([entry.path for entry in index.files()], ["good.py"])


# ---------------------------------------------------------------------------
# 只扫 root 以内
# ---------------------------------------------------------------------------


class RootSafetyTest(unittest.TestCase):
    def test_default_root_is_the_project_root(self):
        self.assertEqual(ProjectIndex().root, PROJECT_ROOT)

    def test_root_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp:
            index = ProjectIndex(root=Path(temp) / "." / ".." / Path(temp).name)

            self.assertEqual(index.root, Path(temp).resolve())

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            ProjectIndex(root=Path("C:/"))

    def test_system_directory_rejected(self):
        for path in ("C:/Windows", "C:/Windows/System32", "C:/Program Files", "/etc"):
            with self.subTest(path=path):
                with self.assertRaises(ToolError):
                    ProjectIndex(root=Path(path))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            ProjectIndex(root=Path("C:/definitely/not/a/real/dir/xyz"))

    def test_file_as_root_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file_path = Path(temp) / "a.txt"
            file_path.write_text("x", encoding="utf-8")

            with self.assertRaises(ToolError):
                ProjectIndex(root=file_path)

    def test_symlinked_directory_outside_root_is_not_followed(self):
        with tempfile.TemporaryDirectory() as inside:
            root = Path(inside).resolve()
            (root / "real").mkdir()
            (root / "real" / "a.py").write_text("x", encoding="utf-8")

            with tempfile.TemporaryDirectory() as outside:
                Path(outside, "secret.py").write_text("x", encoding="utf-8")
                link = root / "link_to_outside"
                try:
                    link.symlink_to(outside, target_is_directory=True)
                except (OSError, NotImplementedError):
                    self.skipTest("当前环境不支持创建符号链接")

                index = ProjectIndex(root=root)
                index.build()

                # 链接指向 root 外面，跟进去就等于把项目外的文件也索引进来了
                self.assertEqual([entry.path for entry in index.entries], ["real/a.py"])


# ---------------------------------------------------------------------------
# 刷新：再扫一遍就把变化收进来
# ---------------------------------------------------------------------------


class RefreshTest(IndexTestBase):
    def test_rebuild_picks_up_a_new_file(self):
        self.make("a.py")
        index = self.scan()

        self.make("b.py")
        index.build()

        self.assertEqual([entry.path for entry in index.entries], ["a.py", "b.py"])

    def test_rebuild_drops_a_deleted_file(self):
        self.make("a.py")
        self.make("b.py")
        index = self.scan()

        (self.root / "b.py").unlink()
        index.build()

        self.assertEqual([entry.path for entry in index.entries], ["a.py"])

    def test_rebuild_sees_changed_content_in_the_digest(self):
        self.make("a.py", "old")
        index = self.scan()
        before = self.by_path(index)["a.py"].digest

        self.make("a.py", "new")
        index.build()
        after = self.by_path(index)["a.py"]

        # 同一秒里改两次 mtime 可能一模一样，能认出变化的是内容摘要
        self.assertNotEqual(before, after.digest)
        self.assertEqual(after.digest, hashlib.sha256(b"new").hexdigest()[:16])

    def test_rebuild_keeps_one_entry_per_path(self):
        self.make("a.py")
        index = self.scan()

        index.build()
        index.build()

        self.assertEqual(len(index.entries), 1)

    def test_rebuild_does_not_touch_the_project(self):
        self.make("a.py", "content")
        self.make("b/c.py")
        before = {
            path: (path.stat().st_size, path.stat().st_mtime)
            for path in self.root.rglob("*")
            if path.is_file()
        }

        index = self.scan()
        index.build()
        index.files()

        after = {
            path: (path.stat().st_size, path.stat().st_mtime)
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)  # 索引只读：扫多少遍项目都是原样


# ---------------------------------------------------------------------------
# 拿本项目真扫一次
# ---------------------------------------------------------------------------


class RealProjectTest(unittest.TestCase):
    def test_indexes_this_project(self):
        index = ProjectIndex()
        index.build()

        entries = {entry.path: entry for entry in index.entries}

        self.assertIn("agent/agent.py", entries)
        self.assertIn("tools/search_code.py", entries)
        self.assertEqual(entries["tools/search_code.py"].language, "python")
        self.assertEqual(entries["tools/search_code.py"].status, INDEXED)
        self.assertGreater(len(index.files("python")), 10)

    def test_noise_directories_do_not_leak(self):
        index = ProjectIndex()
        index.build()

        for entry in index.entries:
            self.assertFalse(entry.path.startswith(".venv/"))
            self.assertNotIn("__pycache__", entry.path)

    def test_every_indexed_entry_has_the_facts_filled_in(self):
        index = ProjectIndex()
        index.build()

        self.assertTrue(index.files())
        for entry in index.files():
            self.assertEqual(len(entry.digest), 16)
            self.assertGreaterEqual(entry.size, 0)  # 空文件也是合法文本
            self.assertGreater(entry.mtime, 0)

    def test_scanning_twice_gives_the_same_index(self):
        index = ProjectIndex()
        index.build()
        first = index.entries

        index.build()

        self.assertEqual(index.entries, first)


if __name__ == "__main__":
    unittest.main()
