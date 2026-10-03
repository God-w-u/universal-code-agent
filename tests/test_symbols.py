"""Symbol Index 测试：每个文件里提取出哪些符号、记成什么样、什么情况下不给。

全部在临时目录里跑，不联网、不碰真实项目文件。四种符号类型（class /
struct / function / method）、父级链、行号、digest 保护、跳过原因、查询
各测一遍，最后拿本项目真跑一次。

「行号必须对应真实源文件」这条最要紧——Code Chunk 要拿它去切代码，
所以这里是把源文件切出来、看名字在不在里面，而不是只比数字。
"""

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from config import PROJECT_ROOT
from index import (
    CHANGED,
    CLASS,
    FUNCTION,
    METHOD,
    STRUCT,
    UNPARSABLE,
    ProjectIndex,
    Skipped,
    Symbol,
    SymbolIndex,
)
from index.project import digest_of
from tools.base import ToolError


class SymbolTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str = "x") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def make_bytes(self, relative: str, data: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def build(self) -> SymbolIndex:
        """建一个自己管项目索引的符号索引，扫一遍。"""
        index = SymbolIndex(root=self.root)
        index.build()
        return index

    def build_with(self, project: ProjectIndex) -> SymbolIndex:
        """用外面传进来的项目索引建（那份清单什么时候刷由调用方说了算）。"""
        index = SymbolIndex(project)
        index.build()
        return index

    def names(self, index: SymbolIndex) -> list[str]:
        return [item.qualified for item in index.symbols()]


PY_SOURCE = (
    "import os\n"
    "\n"
    "\n"
    "class Alpha:\n"
    "    def run(self):\n"
    "        return 1\n"
    "\n"
    "    async def go(self):\n"
    "        pass\n"
    "\n"
    "\n"
    "def top():\n"
    "    pass\n"
)

CPP_SOURCE = (
    "#include <string>\n"
    "\n"
    "namespace app {\n"
    "\n"
    "class Box {\n"
    "public:\n"
    "    int get() const;\n"
    "    void put(int value) { value_ = value; }\n"
    "};\n"
    "\n"
    "}  // namespace app\n"
    "\n"
    "struct Point { int x; int y; };\n"
    "\n"
    "int add(int a, int b) { return a + b; }\n"
    "\n"
    "void Box::put(int value) { }\n"
)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


class PythonSymbolTest(SymbolTestBase):
    def test_class_function_and_method(self):
        self.make("a.py", PY_SOURCE)

        index = self.build()

        self.assertEqual(
            self.names(index),
            ["Alpha", "Alpha.run", "Alpha.go", "top"],
        )
        self.assertEqual(
            [item.kind for item in index.symbols()],
            [CLASS, METHOD, METHOD, FUNCTION],
        )

    def test_a_method_is_not_a_function(self):
        self.make("a.py", PY_SOURCE)

        kinds = {item.qualified: item.kind for item in self.build().symbols()}

        self.assertEqual(kinds["Alpha.run"], METHOD)
        self.assertEqual(kinds["top"], FUNCTION)

    def test_an_async_method_is_found(self):
        self.make("a.py", PY_SOURCE)

        self.assertIn("Alpha.go", self.names(self.build()))

    def test_the_path_is_relative_and_uses_forward_slashes(self):
        self.make("agent/llm/client.py", "def run():\n    pass\n")

        item = self.build().symbols()[0]

        self.assertEqual(item.path, "agent/llm/client.py")
        self.assertFalse(Path(item.path).is_absolute())

    def test_the_parent_chain_is_dot_connected(self):
        self.make(
            "a.py",
            "class Outer:\n"
            "    class Inner:\n"
            "        def run(self):\n"
            "            pass\n",
        )

        index = self.build()

        self.assertEqual(self.names(index), ["Outer", "Outer.Inner", "Outer.Inner.run"])
        self.assertEqual(index.symbols(name="Outer.Inner.run")[0].parent, "Outer.Inner")

    def test_a_decorated_definition_starts_at_the_decorator(self):
        self.make("a.py", "@app.route('/')\ndef index():\n    pass\n")

        item = self.build().symbols()[0]

        self.assertEqual((item.start_line, item.end_line), (1, 3))

    def test_lines_point_into_the_real_source(self):
        # 这是 Code Chunk 的命根子：按 start / end 切出来的块里必须有这个符号
        self.make("a.py", PY_SOURCE)
        lines = PY_SOURCE.splitlines()

        for item in self.build().symbols():
            chunk = "\n".join(lines[item.start_line - 1 : item.end_line])
            self.assertIn(item.name, chunk, f"{item.qualified} 的行号对不上")

    def test_lines_are_one_based_and_closed(self):
        self.make("a.py", "class Alpha:\n    def run(self):\n        pass\n")

        method = self.build().symbols(kind=METHOD)[0]

        self.assertEqual((method.start_line, method.end_line), (2, 3))

    def test_one_file_can_hold_many_symbols(self):
        self.make("a.py", PY_SOURCE)

        self.assertEqual(len(self.build().symbols(path="a.py")), 4)

    def test_top_level_symbols_have_no_parent(self):
        self.make("a.py", PY_SOURCE)

        top = self.build().symbols(name="top")[0]

        self.assertIsNone(top.parent)
        self.assertEqual(top.qualified, "top")


# ---------------------------------------------------------------------------
# C/C++
# ---------------------------------------------------------------------------


class CppSymbolTest(SymbolTestBase):
    def test_class_struct_function_and_method(self):
        self.make("a.cpp", CPP_SOURCE)

        found = {
            item.qualified: item.kind for item in self.build().symbols()
        }

        self.assertEqual(found["app.Box"], CLASS)
        self.assertEqual(found["app.Box.get"], METHOD)
        self.assertEqual(found["app.Box.put"], METHOD)
        self.assertEqual(found["Point"], STRUCT)
        self.assertEqual(found["add"], FUNCTION)

    def test_a_class_outside_a_namespace_has_no_parent(self):
        self.make("a.h", "struct Point { int x; int y; };\n")

        item = self.build().symbols(name="Point")[0]

        self.assertEqual(item.kind, STRUCT)
        self.assertIsNone(item.parent)

    def test_a_namespace_gives_its_members_a_parent(self):
        self.make("a.cpp", "namespace app {\nvoid run() { }\n}\n")

        item = self.build().symbols(name="run")[0]

        self.assertEqual((item.kind, item.parent), (FUNCTION, "app"))

    def test_an_out_of_line_definition_is_a_method(self):
        # void Box::put(...) —— 定义写在类外面的成员函数
        self.make("a.cpp", "void Renderer::draw() { }\n")

        item = self.build().symbols(name="draw")[0]

        self.assertEqual((item.kind, item.parent), (METHOD, "Renderer"))

    def test_a_template_starts_at_the_template_line(self):
        self.make(
            "a.cpp",
            "template <typename T>\nclass Box {\n    T value_;\n};\n",
        )

        item = self.build().symbols(name="Box")[0]

        self.assertEqual((item.start_line, item.end_line), (1, 4))

    def test_nested_classes_keep_the_chain(self):
        self.make(
            "a.cpp",
            "class Outer {\n"
            "public:\n"
            "    class Inner {\n"
            "    public:\n"
            "        void deep();\n"
            "    };\n"
            "};\n",
        )

        index = self.build()

        self.assertEqual(
            self.names(index), ["Outer", "Outer.Inner", "Outer.Inner.deep"]
        )
        self.assertEqual(index.symbols(name="Outer.Inner.deep")[0].kind, METHOD)

    def test_an_access_specifier_is_not_part_of_the_symbol(self):
        self.make("a.cpp", "class Box {\npublic:\n    void put(int v);\n};\n")

        method = self.build().symbols(kind=METHOD)[0]

        self.assertEqual(method.start_line, 3)  # 不是带 public: 的第 2 行

    def test_an_include_is_not_a_symbol(self):
        self.make("a.cpp", '#include <string>\n#include "a.h"\n')

        index = self.build()

        self.assertEqual(index.symbols(), ())

    def test_union_enum_and_namespace_do_not_produce_symbols_yet(self):
        # 它们现在只参与父级链：以后要加，就是往 _SYMBOL_KIND 里补一条
        self.make(
            "a.h",
            "union U { int a; };\nenum Color { Red };\nnamespace app { }\n",
        )

        self.assertEqual(self.build().symbols(), ())

    def test_lines_point_into_the_real_source(self):
        self.make("a.cpp", CPP_SOURCE)
        lines = CPP_SOURCE.splitlines()

        for item in self.build().symbols():
            chunk = "\n".join(lines[item.start_line - 1 : item.end_line])
            self.assertIn(item.name, chunk, f"{item.qualified} 的行号对不上")

    def test_a_file_with_nothing_to_extract_has_no_symbols(self):
        self.make("empty.c", "\n\n")
        self.make("globals.c", "int x = 1;\nstatic int y = 2;\n")

        index = self.build()

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, ())  # 没有结构不是错


# ---------------------------------------------------------------------------
# digest：符号跟着文件走
# ---------------------------------------------------------------------------


class DigestTest(SymbolTestBase):
    def test_a_symbol_carries_the_digest_of_its_file(self):
        self.make("a.py", PY_SOURCE)

        project = ProjectIndex(root=self.root)
        project.build()
        expected = {entry.path: entry.digest for entry in project.files()}

        for item in self.build_with(project).symbols():
            self.assertEqual(item.digest, expected[item.path])

    def test_a_symbol_digest_is_the_file_digest_not_the_symbol_digest(self):
        self.make("a.py", PY_SOURCE)
        raw = (self.root / "a.py").read_bytes()

        item = self.build().symbols()[0]

        self.assertEqual(item.digest, digest_of(raw))

    def test_all_symbols_of_one_file_share_the_digest(self):
        self.make("a.py", PY_SOURCE)

        digests = {item.digest for item in self.build().symbols()}

        self.assertEqual(len(digests), 1)

    def test_editing_the_file_changes_the_digest(self):
        self.make("a.py", "def run():\n    pass\n")
        before = self.build().symbols()[0].digest

        self.make("a.py", "def run():\n    return 1\n")
        after = self.build().symbols()[0].digest

        self.assertNotEqual(before, after)

    def test_a_changed_file_produces_no_symbols(self):
        # 文件清单建好之后文件被改了：digest 对不上，宁可一条不给
        self.make("a.py", "def run():\n    pass\n")
        project = ProjectIndex(root=self.root)
        project.build()

        self.make("a.py", "def totally_different():\n    pass\n")
        index = self.build_with(project)

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, (Skipped("a.py", CHANGED),))

    def test_one_changed_file_does_not_take_down_the_others(self):
        self.make("a.py", "def run():\n    pass\n")
        self.make("b.py", "def keep():\n    pass\n")
        project = ProjectIndex(root=self.root)
        project.build()

        self.make("a.py", "def changed():\n    pass\n")
        index = self.build_with(project)

        self.assertEqual(self.names(index), ["keep"])
        self.assertEqual([item.path for item in index.skipped], ["a.py"])

    def test_a_deleted_file_produces_no_symbols(self):
        self.make("a.py", "def run():\n    pass\n")
        project = ProjectIndex(root=self.root)
        project.build()

        (self.root / "a.py").unlink()
        index = self.build_with(project)

        self.assertEqual(index.symbols(), ())
        self.assertEqual(
            [item.reason for item in index.skipped],
            ["unreadable"],
        )

    def test_a_rebuild_after_the_change_sees_the_new_symbols(self):
        # 刷新文件清单再建符号：拿到的就是新内容，不是过期数据
        self.make("a.py", "def run():\n    pass\n")
        index = SymbolIndex(root=self.root)
        index.build()

        self.make("a.py", "def other():\n    pass\n")
        index.build()

        self.assertEqual(self.names(index), ["other"])
        self.assertEqual(index.skipped, ())


# ---------------------------------------------------------------------------
# 不产出符号的那些情况
# ---------------------------------------------------------------------------


class SkippedFileTest(SymbolTestBase):
    def test_a_python_file_with_a_syntax_error_is_unparsable(self):
        self.make("broken.py", "def broken(:\n")

        index = self.build()

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, (Skipped("broken.py", UNPARSABLE),))

    def test_a_syntax_error_does_not_take_down_the_others(self):
        self.make("broken.py", "def broken(:\n")
        self.make("fine.py", "def run():\n    pass\n")

        index = self.build()

        self.assertEqual(self.names(index), ["run"])
        self.assertEqual([item.path for item in index.skipped], ["broken.py"])

    def test_a_language_without_an_extractor_is_silently_skipped(self):
        # java / go / rust / js / ts 现在只认得出来，还不会解析——不是出错
        for name in ("a.java", "a.go", "a.rs", "a.js", "a.ts"):
            self.make(name, "class A { }\n")

        index = self.build()

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, ())

    def test_a_text_file_of_an_unknown_language_is_ignored(self):
        self.make("README.md", "# 标题\n")
        self.make("notes.txt", "随便写点什么\n")

        index = self.build()

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, ())

    def test_a_file_with_no_definitions_is_not_an_error(self):
        self.make("a.py", "x = 1\nprint(x)\n")

        index = self.build()

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, ())

    def test_files_not_indexed_by_the_project_are_never_read(self):
        self.make("good.py", "def run():\n    pass\n")
        self.make_bytes("blob.py", b"\x00\x01\x02")

        index = self.build()

        self.assertEqual(self.names(index), ["run"])
        self.assertEqual(index.skipped, ())

    def test_an_empty_project_has_nothing(self):
        index = self.build()

        self.assertEqual(index.symbols(), ())
        self.assertEqual(index.skipped, ())


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


class QueryTest(SymbolTestBase):
    def setUp(self):
        super().setUp()
        self.make("a.py", PY_SOURCE)
        self.make("sub/b.py", "def top():\n    pass\n")
        self.make("c.cpp", "int add(int a, int b);\n")
        self.index = self.build()

    def test_no_condition_returns_everything(self):
        self.assertEqual(len(self.index.symbols()), 6)

    def test_by_path(self):
        self.assertEqual(
            self.names(self.index),
            ["Alpha", "Alpha.run", "Alpha.go", "top", "add", "top"],
        )
        self.assertEqual(
            [item.path for item in self.index.symbols(path="a.py")],
            ["a.py"] * 4,
        )

    def test_by_simple_name(self):
        found = self.index.symbols(name="top")

        self.assertEqual([item.path for item in found], ["a.py", "sub/b.py"])

    def test_by_qualified_name(self):
        found = self.index.symbols(name="Alpha.run")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kind, METHOD)

    def test_by_kind(self):
        self.assertEqual(
            [item.name for item in self.index.symbols(kind=FUNCTION)], ["top", "add", "top"]
        )
        self.assertEqual(
            [item.name for item in self.index.symbols(kind=METHOD)], ["run", "go"]
        )

    def test_conditions_are_combined_with_and(self):
        found = self.index.symbols(kind=FUNCTION, path="sub/b.py")

        self.assertEqual([item.qualified for item in found], ["top"])

    def test_no_match_is_an_empty_tuple(self):
        self.assertEqual(self.index.symbols(name="没有这个东西"), ())
        self.assertEqual(self.index.symbols(path="nope.py"), ())

    def test_results_follow_path_order(self):
        paths = [item.path for item in self.index.symbols()]

        self.assertEqual(paths, sorted(paths))


class SymbolRecordTest(SymbolTestBase):
    def test_a_symbol_cannot_be_edited_afterwards(self):
        self.make("a.py", PY_SOURCE)

        item = self.build().symbols()[0]

        with self.assertRaises(FrozenInstanceError):
            item.name = "别的"

    def test_a_symbol_is_a_plain_value(self):
        # 全是算好的值：不指向文件、不指向索引，交到哪儿都还好使
        source = "def run():\n    pass\n"
        self.make("a.py", source)

        item = self.build().symbols()[0]

        self.assertEqual(
            item,
            Symbol(
                name="run",
                kind=FUNCTION,
                path="a.py",
                start_line=1,
                end_line=2,
                digest=digest_of(source.encode("utf-8")),
            ),
        )

    def test_building_does_not_touch_the_project(self):
        self.make("a.py", PY_SOURCE)
        self.make("b.cpp", CPP_SOURCE)
        before = {
            path: (path.stat().st_size, path.stat().st_mtime)
            for path in self.root.rglob("*")
            if path.is_file()
        }

        self.build()

        after = {
            path: (path.stat().st_size, path.stat().st_mtime)
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)  # 只读：建索引不写项目


# ---------------------------------------------------------------------------
# 和 Project Index 的配合
# ---------------------------------------------------------------------------


class ProjectIndexLinkTest(SymbolTestBase):
    def test_an_own_project_index_is_scanned_first(self):
        self.make("a.py", "def run():\n    pass\n")

        index = SymbolIndex(root=self.root)
        index.build()

        self.assertEqual(self.names(index), ["run"])

    def test_an_own_project_index_is_refreshed_on_every_build(self):
        self.make("a.py", "def run():\n    pass\n")
        index = SymbolIndex(root=self.root)
        index.build()

        self.make("b.py", "def later():\n    pass\n")
        index.build()

        self.assertEqual(self.names(index), ["run", "later"])

    def test_a_project_index_from_outside_is_not_rescanned(self):
        # 外面传进来的清单不动：什么时候刷新由调用方决定
        self.make("a.py", "def run():\n    pass\n")
        project = ProjectIndex(root=self.root)
        project.build()

        self.make("b.py", "def later():\n    pass\n")
        index = self.build_with(project)

        self.assertEqual(self.names(index), ["run"])
        self.assertEqual(index.skipped, ())  # 没扫过它，不算「跳过」

    def test_built_is_false_until_the_first_scan(self):
        self.make("a.py", "def run():\n    pass\n")
        index = SymbolIndex(root=self.root)

        self.assertFalse(index.built)  # 空索引 ≠ 项目里没有符号
        self.assertEqual(index.symbols(), ())

        index.build()

        self.assertTrue(index.built)

    def test_an_unbuilt_project_index_from_outside_is_scanned_once(self):
        # 传了一份没扫过的清单：替它扫一次，而不是静默返回零个符号
        self.make("a.py", "def run():\n    pass\n")
        index = SymbolIndex(ProjectIndex(root=self.root))
        index.build()

        self.assertEqual(self.names(index), ["run"])

    def test_giving_both_a_project_and_a_root_is_rejected(self):
        with self.assertRaises(ToolError):
            SymbolIndex(ProjectIndex(root=self.root), root=self.root)

    def test_the_default_project_is_the_real_one(self):
        self.assertEqual(SymbolIndex().project.root, PROJECT_ROOT)

    def test_a_rebuild_drops_the_old_symbols(self):
        self.make("a.py", "def run():\n    pass\n")
        index = SymbolIndex(root=self.root)
        index.build()

        (self.root / "a.py").unlink()
        index.build()

        self.assertEqual(index.symbols(), ())


# ---------------------------------------------------------------------------
# 拿本项目真跑一次
# ---------------------------------------------------------------------------


class RealProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = SymbolIndex()
        cls.index.build()

    def test_finds_symbols_in_this_project(self):
        self.assertIn("ProjectIndex", self.names_of("index/project.py"))
        self.assertIn("ProjectIndex.build", self.names_of("index/project.py"))
        self.assertIn("extract", self.names_of("tools/cpp.py"))

    def names_of(self, path: str) -> list[str]:
        return [item.qualified for item in self.index.symbols(path=path)]

    def test_kinds_are_all_known(self):
        known = {CLASS, STRUCT, FUNCTION, METHOD}

        self.assertTrue(self.index.symbols())
        for item in self.index.symbols():
            self.assertIn(item.kind, known)

    def test_every_line_range_points_into_the_real_file(self):
        # 全项目的行号体检：范围合法、切出来不空、名字在块里
        checked = set()
        for item in self.index.symbols():
            if item.path in checked:
                continue
            checked.add(item.path)
            lines = (PROJECT_ROOT / item.path).read_text(encoding="utf-8").splitlines()

            for other in self.index.symbols(path=item.path):
                self.assertGreaterEqual(other.start_line, 1)
                self.assertLessEqual(other.start_line, other.end_line)
                self.assertLessEqual(other.end_line, len(lines))
                chunk = "\n".join(lines[other.start_line - 1 : other.end_line])
                if other.path.endswith(".py"):  # C++ 的名字可能被宏拆散，只对 Python 严
                    self.assertIn(other.name, chunk, f"{other.qualified} 对不上")

        self.assertGreater(len(checked), 5)

    def test_nothing_in_this_project_is_skipped(self):
        self.assertEqual(self.index.skipped, ())

    def test_every_digest_still_matches_the_file_on_disk(self):
        # 索引是刚建的，项目又没动过：每条符号的 digest 都该和文件对得上。
        # 对不上就说明「过期保护」要么没生效、要么在乱报
        for item in self.index.symbols():
            raw = (PROJECT_ROOT / item.path).read_bytes()
            self.assertEqual(item.digest, digest_of(raw))


if __name__ == "__main__":
    unittest.main()
