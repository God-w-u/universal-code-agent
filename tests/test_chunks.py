"""Code Chunk 测试：切出来的是不是真代码、边界对不对、什么情况下不给。

全部在临时目录里跑，不联网、不碰真实项目文件。最要紧的一条是**正文必须
逐字节等于源文件的那几行**——切错了行，检索命中之后 Agent 会照着它去看一段
完全不相干的代码，比没有 chunk 更糟。所以这里到处是「拿源文件切出来对」。

「行范围一模一样的嵌套符号只留一条」这种去重规则也单独钉着：它很容易在
以后重构时被顺手改掉，一改就会多出一堆重复的 chunk。
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
    ChunkIndex,
    CodeChunk,
    ProjectIndex,
    Skipped,
    SymbolIndex,
)
from index.project import digest_of
from tools.base import ToolError


class ChunkTestBase(unittest.TestCase):
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

    def build(self) -> ChunkIndex:
        """建一个自己管索引的切块索引，扫一遍。"""
        index = ChunkIndex(root=self.root)
        index.build()
        return index

    def build_with(self, symbols: SymbolIndex) -> ChunkIndex:
        """用外面传进来的符号索引建（那份索引什么时候刷由调用方说了算）。"""
        index = ChunkIndex(symbols=symbols)
        index.build()
        return index

    def slice_of(self, relative: str, start: int, end: int) -> str:
        """源文件里那几行的原文——拿来和 chunk.code 逐字节比。

        这里刻意自己切一遍（不调 chunks.py 用的那个函数）：要验的就是
        「按行号切出来的对不对」，用被测代码自己的切法去验它没有意义。
        """
        text = (self.root / relative).read_bytes().decode("utf-8")
        return "\n".join(text.replace("\r\n", "\n").split("\n")[start - 1 : end])

    def symbols_of(self, index: ChunkIndex) -> list[str]:
        return [item.symbol for item in index.chunks()]

    def symbols_index(self) -> SymbolIndex:
        index = SymbolIndex(root=self.root)
        index.build()
        return index


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


class PythonChunkTest(ChunkTestBase):
    def test_one_chunk_per_symbol(self):
        self.make("a.py", PY_SOURCE)

        index = self.build()

        self.assertEqual(
            self.symbols_of(index), ["Alpha", "Alpha.run", "Alpha.go", "top"]
        )
        self.assertEqual(
            [item.kind for item in index.chunks()],
            [CLASS, METHOD, METHOD, FUNCTION],
        )

    def test_the_code_is_the_real_text_of_those_lines(self):
        self.make("a.py", PY_SOURCE)

        for item in self.build().chunks():
            self.assertEqual(
                item.code,
                self.slice_of("a.py", item.start_line, item.end_line),
                f"{item.id} 的正文和源文件对不上",
            )

    def test_a_class_chunk_holds_its_whole_body(self):
        self.make("a.py", PY_SOURCE)

        chunk = self.build().chunks(symbol="Alpha")[0]

        self.assertEqual((chunk.start_line, chunk.end_line), (4, 9))
        self.assertIn("class Alpha:", chunk.code)
        self.assertIn("def run(self):", chunk.code)  # 类里含着方法，整段都在

    def test_a_method_chunk_starts_at_its_def_and_stops_at_its_body(self):
        self.make("a.py", PY_SOURCE)

        chunk = self.build().chunks(symbol="Alpha.run")[0]

        self.assertEqual((chunk.start_line, chunk.end_line), (5, 6))
        self.assertEqual(chunk.code, "    def run(self):\n        return 1")

    def test_a_decorated_definition_includes_the_decorator(self):
        self.make("a.py", "@app.route('/')\ndef index():\n    return 'hi'\n")

        chunk = self.build().chunks()[0]

        self.assertEqual((chunk.start_line, chunk.end_line), (1, 3))
        self.assertTrue(chunk.code.startswith("@app.route"))

    def test_the_path_symbol_and_language_are_filled_in(self):
        self.make("agent/llm/client.py", "def run():\n    pass\n")

        chunk = self.build().chunks()[0]

        self.assertEqual(chunk.path, "agent/llm/client.py")
        self.assertEqual(chunk.symbol, "run")
        self.assertEqual(chunk.language, "python")
        self.assertIsNone(chunk.parent)

    def test_the_id_says_where_the_chunk_is(self):
        self.make("a.py", PY_SOURCE)

        self.assertEqual(
            self.build().chunks(symbol="Alpha.run")[0].id,
            "a.py:Alpha.run:5-6",
        )


class NestedChunkTest(ChunkTestBase):
    NESTED = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def run(self):\n"
        "            return 1\n"
        "\n"
        "    def outer_run(self):\n"
        "        pass\n"
    )

    def test_nested_symbols_each_get_a_chunk_with_a_parent(self):
        self.make("a.py", self.NESTED)

        index = self.build()

        self.assertEqual(
            [(item.symbol, item.parent) for item in index.chunks()],
            [
                ("Outer", None),
                ("Outer.Inner", "Outer"),
                ("Outer.Inner.run", "Outer.Inner"),
                ("Outer.outer_run", "Outer"),
            ],
        )

    def test_the_outer_chunk_contains_the_inner_code(self):
        # 这是有意的：外层 chunk 的范围是它在文件里的真实边界，
        # 检索时「整个类」和「类里某个方法」是两种粒度
        self.make("a.py", self.NESTED)

        outer = self.build().chunks(symbol="Outer")[0]

        self.assertIn("class Inner:", outer.code)
        self.assertIn("def outer_run(self):", outer.code)

    def test_nesting_does_not_produce_two_identical_chunks(self):
        # 嵌套会带来包含关系（外层正文里含着内层），但不该有两条一模一样的
        self.make("a.py", self.NESTED)

        codes = [item.code for item in self.build().chunks()]

        self.assertEqual(len(codes), len(set(codes)))

    def test_no_two_chunks_of_a_file_cover_the_same_lines(self):
        # 去重规则的底线：同一段行范围只可能有一条 chunk。
        # （正文相同是另一回事：两个类里各有一个写法和名字都一样的方法，
        # 代码一样、位置不同，那本来就是两条，不该合。）
        self.make("a.py", self.NESTED)

        spans = [(item.start_line, item.end_line) for item in self.build().chunks()]

        self.assertEqual(len(spans), len(set(spans)))

    def test_children_can_be_queried_by_parent(self):
        self.make("a.py", self.NESTED)

        index = self.build()

        self.assertEqual(
            [item.symbol for item in index.chunks(parent="Outer")],
            ["Outer.Inner", "Outer.outer_run"],
        )
        self.assertEqual(
            [item.symbol for item in index.chunks(parent="Outer.Inner")],
            ["Outer.Inner.run"],
        )


# ---------------------------------------------------------------------------
# C/C++
# ---------------------------------------------------------------------------


class CppChunkTest(ChunkTestBase):
    def test_one_chunk_per_symbol(self):
        self.make("a.cpp", CPP_SOURCE)

        index = self.build()

        self.assertEqual(
            self.symbols_of(index),
            ["app.Box", "app.Box.get", "app.Box.put", "Point", "add", "Box.put"],
        )
        self.assertEqual(
            [item.kind for item in index.chunks()],
            [CLASS, METHOD, METHOD, STRUCT, FUNCTION, METHOD],
        )

    def test_the_code_is_the_real_text_of_those_lines(self):
        self.make("a.cpp", CPP_SOURCE)

        for item in self.build().chunks():
            self.assertEqual(
                item.code,
                self.slice_of("a.cpp", item.start_line, item.end_line),
                f"{item.id} 的正文和源文件对不上",
            )

    def test_a_method_of_a_namespaced_class_knows_its_parent(self):
        self.make("a.cpp", CPP_SOURCE)

        chunk = self.build().chunks(symbol="app.Box.get")[0]

        self.assertEqual((chunk.parent, chunk.language), ("app.Box", "cpp"))
        self.assertEqual(chunk.code, "    int get() const;")

    def test_a_free_function_has_no_parent(self):
        self.make("a.cpp", CPP_SOURCE)

        chunk = self.build().chunks(symbol="add")[0]

        self.assertIsNone(chunk.parent)
        self.assertEqual(chunk.code, "int add(int a, int b) { return a + b; }")

    def test_an_out_of_line_definition_is_its_own_chunk(self):
        self.make("a.cpp", CPP_SOURCE)

        chunk = self.build().chunks(symbol="Box.put")[0]

        self.assertEqual((chunk.kind, chunk.parent), (METHOD, "Box"))
        self.assertEqual(chunk.code, "void Box::put(int value) { }")

    def test_a_one_line_struct_and_its_method_give_one_chunk(self):
        # `struct A { void f(); };` 一行写完：两条 symbol 的行范围一模一样，
        # 切出来的代码会一个字不差——只留外层那条
        self.make("a.h", "struct A { void f(); };\n")

        index = self.build()

        self.assertEqual(self.symbols_of(index), ["A"])
        self.assertEqual(index.chunks()[0].code, "struct A { void f(); };")

    def test_a_multiline_class_keeps_both_levels(self):
        # 换个写法（类和方法不在一行）就该有两条：代码不一样
        self.make("a.h", "struct A {\n    void f();\n};\n")

        index = self.build()

        self.assertEqual(self.symbols_of(index), ["A", "A.f"])
        self.assertEqual(index.chunks(symbol="A.f")[0].code, "    void f();")

    def test_a_template_chunk_starts_at_the_template_line(self):
        self.make(
            "a.cpp",
            "template <typename T>\nstruct Box {\n    T value_;\n};\n",
        )

        chunk = self.build().chunks(symbol="Box")[0]

        self.assertEqual((chunk.start_line, chunk.end_line), (1, 4))
        self.assertTrue(chunk.code.startswith("template <typename T>"))


# ---------------------------------------------------------------------------
# digest：chunk 跟着文件走
# ---------------------------------------------------------------------------


class DigestTest(ChunkTestBase):
    def test_a_chunk_carries_the_digest_of_its_file(self):
        self.make("a.py", PY_SOURCE)
        expected = digest_of((self.root / "a.py").read_bytes())

        for item in self.build().chunks():
            self.assertEqual(item.digest, expected)

    def test_a_changed_file_produces_no_chunks(self):
        # 符号索引建好之后文件被改了：行号已经不作数，一条都不切
        self.make("a.py", "def run():\n    pass\n")
        symbols = self.symbols_index()

        self.make("a.py", "def totally_different():\n    pass\n")
        index = self.build_with(symbols)

        self.assertEqual(index.chunks(), ())
        self.assertEqual(index.skipped, (Skipped("a.py", CHANGED),))

    def test_one_changed_file_does_not_take_down_the_others(self):
        self.make("a.py", "def run():\n    pass\n")
        self.make("b.py", "def keep():\n    pass\n")
        symbols = self.symbols_index()

        self.make("a.py", "def changed():\n    pass\n")
        index = self.build_with(symbols)

        self.assertEqual(self.symbols_of(index), ["keep"])
        self.assertEqual(index.chunks(symbol="keep")[0].code, "def keep():\n    pass")
        self.assertEqual([item.path for item in index.skipped], ["a.py"])

    def test_a_deleted_file_produces_no_chunks(self):
        self.make("a.py", "def run():\n    pass\n")
        symbols = self.symbols_index()

        (self.root / "a.py").unlink()
        index = self.build_with(symbols)

        self.assertEqual(index.chunks(), ())
        self.assertEqual([item.path for item in index.skipped], ["a.py"])

    def test_a_rebuild_after_the_change_slices_the_new_code(self):
        self.make("a.py", "def run():\n    pass\n")
        index = ChunkIndex(root=self.root)
        index.build()

        self.make("a.py", "def run():\n    return 1\n")
        index.build()

        self.assertEqual(index.chunks()[0].code, "def run():\n    return 1")
        self.assertEqual(index.skipped, ())

    def test_the_code_is_not_rebuilt_from_the_symbol(self):
        # 正文只能是文件里的原文：把同名符号放到别的行数上，切出来的也得是文件
        self.make("a.py", "x = 1\n\n\ndef run():\n    pass\n")

        chunk = self.build().chunks()[0]

        self.assertEqual(chunk.symbol, "run")
        self.assertEqual((chunk.start_line, chunk.end_line), (4, 5))
        self.assertEqual(chunk.code, "def run():\n    pass")


# ---------------------------------------------------------------------------
# 不切块的几种情况
# ---------------------------------------------------------------------------


class NoChunkTest(ChunkTestBase):
    def test_a_file_without_symbols_produces_nothing(self):
        self.make("a.py", "x = 1\nprint(x)\n")

        index = self.build()

        self.assertEqual(index.chunks(), ())
        self.assertEqual(index.skipped, ())  # 没切不是错

    def test_text_files_of_other_languages_produce_nothing(self):
        self.make("README.md", "# 标题\n")
        self.make("a.java", "class A { }\n")
        self.make("notes.txt", "随便写点什么\n")

        index = self.build()

        self.assertEqual(index.chunks(), ())
        self.assertEqual(index.skipped, ())

    def test_an_empty_file_produces_nothing(self):
        self.make("empty.py", "")

        self.assertEqual(self.build().chunks(), ())

    def test_an_empty_project_produces_nothing(self):
        index = self.build()

        self.assertEqual(index.chunks(), ())
        self.assertEqual(index.skipped, ())

    def test_a_file_that_failed_to_parse_produces_nothing(self):
        # 解析失败在符号索引那边已经记过了，切块这边不必再说一遍
        self.make("broken.py", "def broken(:\n")
        self.make("fine.py", "def run():\n    pass\n")

        index = self.build()

        self.assertEqual(self.symbols_of(index), ["run"])
        self.assertEqual(index.skipped, ())

    def test_files_the_project_index_skipped_are_never_cut(self):
        self.make("good.py", "def run():\n    pass\n")
        self.make_bytes("blob.py", b"\x00\x01\x02")

        index = self.build()

        self.assertEqual(self.symbols_of(index), ["run"])
        self.assertEqual(index.skipped, ())


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


class QueryTest(ChunkTestBase):
    def setUp(self):
        super().setUp()
        self.make("a.py", PY_SOURCE)
        self.make("sub/b.py", "def top():\n    pass\n")
        self.make("c.cpp", "int add(int a, int b);\n")
        self.index = self.build()

    def test_no_condition_returns_everything(self):
        self.assertEqual(len(self.index.chunks()), 6)

    def test_by_path(self):
        self.assertEqual(
            [item.symbol for item in self.index.chunks(path="a.py")],
            ["Alpha", "Alpha.run", "Alpha.go", "top"],
        )

    def test_by_simple_symbol_name(self):
        found = self.index.chunks(symbol="top")

        self.assertEqual([item.path for item in found], ["a.py", "sub/b.py"])

    def test_by_qualified_symbol_name(self):
        found = self.index.chunks(symbol="Alpha.run")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kind, METHOD)

    def test_by_kind(self):
        self.assertEqual(
            [item.symbol for item in self.index.chunks(kind=FUNCTION)],
            ["top", "add", "top"],
        )

    def test_conditions_are_combined_with_and(self):
        found = self.index.chunks(kind=FUNCTION, path="sub/b.py")

        self.assertEqual([item.symbol for item in found], ["top"])

    def test_no_match_is_an_empty_tuple(self):
        self.assertEqual(self.index.chunks(symbol="没有这个东西"), ())

    def test_results_follow_path_order(self):
        paths = [item.path for item in self.index.chunks()]

        self.assertEqual(paths, sorted(paths))


class ChunkRecordTest(ChunkTestBase):
    def test_a_chunk_cannot_be_edited_afterwards(self):
        self.make("a.py", "def run():\n    pass\n")

        chunk = self.build().chunks()[0]

        with self.assertRaises(FrozenInstanceError):
            chunk.code = "别的"

    def test_a_chunk_is_a_plain_value(self):
        source = "def run():\n    pass\n"
        self.make("a.py", source)

        chunk = self.build().chunks()[0]

        self.assertEqual(
            chunk,
            CodeChunk(
                path="a.py",
                symbol="run",
                kind=FUNCTION,
                language="python",
                start_line=1,
                end_line=2,
                digest=digest_of(source.encode("utf-8")),
                code=source.rstrip("\n"),
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
        self.assertEqual(before, after)  # 只读：切块不写项目


# ---------------------------------------------------------------------------
# 和 Symbol Index 的配合
# ---------------------------------------------------------------------------


class SymbolIndexLinkTest(ChunkTestBase):
    def test_an_own_symbol_index_is_scanned_first(self):
        self.make("a.py", "def run():\n    pass\n")

        index = ChunkIndex(root=self.root)
        index.build()

        self.assertEqual(self.symbols_of(index), ["run"])

    def test_an_own_symbol_index_is_refreshed_on_every_build(self):
        self.make("a.py", "def run():\n    pass\n")
        index = ChunkIndex(root=self.root)
        index.build()

        self.make("b.py", "def later():\n    pass\n")
        index.build()

        self.assertEqual(self.symbols_of(index), ["run", "later"])

    def test_a_symbol_index_from_outside_is_not_rescanned(self):
        self.make("a.py", "def run():\n    pass\n")
        symbols = self.symbols_index()

        self.make("b.py", "def later():\n    pass\n")
        index = self.build_with(symbols)

        self.assertEqual(self.symbols_of(index), ["run"])
        self.assertEqual(index.skipped, ())  # 没扫过它，不算「跳过」

    def test_an_unbuilt_symbol_index_from_outside_is_scanned_once(self):
        self.make("a.py", "def run():\n    pass\n")

        index = self.build_with(SymbolIndex(root=self.root))

        self.assertEqual(self.symbols_of(index), ["run"])

    def test_giving_both_symbols_and_a_root_is_rejected(self):
        with self.assertRaises(ToolError):
            ChunkIndex(SymbolIndex(root=self.root), root=self.root)

    def test_the_default_project_is_the_real_one(self):
        self.assertEqual(ChunkIndex().symbols.project.root, PROJECT_ROOT)

    def test_a_rebuild_drops_the_old_chunks(self):
        self.make("a.py", "def run():\n    pass\n")
        index = ChunkIndex(root=self.root)
        index.build()

        (self.root / "a.py").unlink()
        index.build()

        self.assertEqual(index.chunks(), ())

    def test_the_project_index_can_be_reached_through_the_symbols(self):
        self.make("a.py", "def run():\n    pass\n")

        index = ChunkIndex(root=self.root)
        index.build()

        self.assertIsInstance(index.symbols.project, ProjectIndex)


# ---------------------------------------------------------------------------
# 拿本项目真跑一次
# ---------------------------------------------------------------------------


class RealProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ChunkIndex()
        cls.index.build()

    def test_cuts_chunks_out_of_this_project(self):
        self.assertTrue(self.index.chunks())
        self.assertEqual(self.index.skipped, ())
        self.assertIn("ProjectIndex", self.symbols_of("index/project.py"))

    def symbols_of(self, path: str) -> list[str]:
        return [item.symbol for item in self.index.chunks(path=path)]

    def test_every_code_is_verbatim_from_the_file(self):
        # 全项目的正文体检：每条 chunk 的 code 必须逐字节等于源文件那几行
        checked = set()
        for chunk in self.index.chunks():
            if chunk.path in checked:
                continue
            checked.add(chunk.path)
            text = (PROJECT_ROOT / chunk.path).read_bytes().decode("utf-8")
            lines = text.replace("\r\n", "\n").split("\n")
            if lines and lines[-1] == "":
                lines.pop()

            for item in self.index.chunks(path=chunk.path):
                self.assertGreaterEqual(item.start_line, 1)
                self.assertLessEqual(item.start_line, item.end_line)
                self.assertLessEqual(item.end_line, len(lines))
                self.assertEqual(
                    item.code,
                    "\n".join(lines[item.start_line - 1 : item.end_line]),
                    f"{item.id} 的正文和源文件对不上",
                )

        self.assertGreater(len(checked), 5)

    def test_ids_are_unique(self):
        ids = [item.id for item in self.index.chunks()]

        self.assertEqual(len(ids), len(set(ids)))

    def test_no_two_chunks_of_a_file_cover_the_same_lines(self):
        # 去重规则保证的那条。正文相同不算错：两个类里各有一个写法和名字
        # 都一样的方法，本来就该是两条（本项目的 tests/test_build.py 里就有）
        by_path: dict[str, list[tuple[int, int]]] = {}
        for item in self.index.chunks():
            by_path.setdefault(item.path, []).append((item.start_line, item.end_line))

        for path, spans in by_path.items():
            self.assertEqual(len(spans), len(set(spans)), f"{path} 里有行范围重复的 chunk")

    def test_every_chunk_knows_where_it_came_from(self):
        for item in self.index.chunks():
            self.assertTrue(item.symbol)
            self.assertTrue(item.path)
            self.assertEqual(len(item.digest), 16)
            self.assertIn(item.kind, {CLASS, STRUCT, FUNCTION, METHOD})
            self.assertFalse(item.code.startswith("\n"))

    def test_a_text_file_without_symbols_has_no_chunks(self):
        self.assertEqual(self.index.chunks(path="README.md"), ())

    def test_parents_point_at_a_symbol_that_exists(self):
        # Python 这边父级一定也有 chunk（类 / 方法都会切）——C++ 的 namespace
        # 不产出符号，所以只对 .py 立这条规矩
        for item in self.index.chunks():
            if item.parent is None or not item.path.endswith(".py"):
                continue
            self.assertTrue(
                self.index.chunks(symbol=item.parent),
                f"{item.symbol} 的父级 {item.parent} 没有对应的 chunk",
            )


if __name__ == "__main__":
    unittest.main()
