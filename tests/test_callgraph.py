"""Call Graph 测试：调用点提取得对不对、边认得准不准、查询稳不稳。

源文件夹具全在临时目录里写（最后拿本项目真跑一次烟雾测试），不联网、
不碰仓外的文件。规格里点名的九类都在这儿：Python 普通 / 方法调用、
C++ 普通 / namespace 调用、caller / callee 双向查询、不存在的函数、
重复边去重、解析不了不崩、多语言互不污染。
"""

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from callgraph import CallEdge, CallGraph, CallGraphIndex, CallPath
from callgraph.cpp import extract_sites as cpp_sites
from callgraph.python import extract_sites as python_sites
from index import METHOD, SymbolIndex
from tools.base import ToolError


class CallGraphTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def build(self) -> CallGraphIndex:
        """建一个自己管索引的调用图，扫一遍。"""
        index = CallGraphIndex(root=self.root)
        index.build()
        return index


# ---------------------------------------------------------------------------
# 边和图
# ---------------------------------------------------------------------------


class EdgeTest(unittest.TestCase):
    def test_key_is_path_caller_callee_line(self):
        edge = CallEdge("Agent.run", "helper", "agent/agent.py", 12, "python")

        self.assertEqual(edge.key, ("agent/agent.py", "Agent.run", "helper", 12))

    def test_edge_is_frozen(self):
        edge = CallEdge("Agent.run", "helper", "agent/agent.py", 12, "python")

        with self.assertRaises(FrozenInstanceError):
            edge.line = 13  # type: ignore[misc]


class CallGraphQueryTest(unittest.TestCase):
    """摆一张手写的小图，把两个方向的查询和排序钉住。"""

    def graph(self) -> CallGraph:
        graph = CallGraph()
        graph.add(CallEdge("Agent.run", "helper", "agent/agent.py", 10, "python"))
        graph.add(CallEdge("Agent.run", "self.step", "agent/agent.py", 11, "python"))
        graph.add(CallEdge("Agent.step", "helper", "agent/agent.py", 20, "python"))
        graph.add(CallEdge("Cli.main", "Agent.run", "main.py", 5, "python"))
        return graph

    def test_add_and_len(self):
        self.assertEqual(len(self.graph()), 4)

    def test_add_rejects_things_that_are_not_edges(self):
        with self.assertRaises(TypeError):
            CallGraph().add(("Agent.run", "helper"))

    def test_repeated_edge_is_not_added(self):
        graph = self.graph()

        added = graph.add(
            CallEdge("Agent.run", "helper", "agent/agent.py", 10, "python")
        )

        self.assertFalse(added)
        self.assertEqual(len(graph), 4)

    def test_callees(self):
        graph = self.graph()

        self.assertEqual(graph.callees("Agent.run"), ("helper", "self.step"))
        # 简单名：叫 run 的函数不止一个时边都算上——不做类型推导
        self.assertEqual(graph.callees("run"), ("helper", "self.step"))

    def test_callers(self):
        graph = self.graph()

        self.assertEqual(graph.callers("helper"), ("Agent.run", "Agent.step"))
        # 被叫的名字是 self.step，按最后一段也认
        self.assertEqual(graph.callers("step"), ("Agent.run",))
        self.assertEqual(graph.callers("Agent.run"), ("Cli.main",))

    def test_a_qualified_query_is_exact(self):
        # 12.2 的测试发现的：限定名查询原先也按最后一段比，于是
        # callees("Alpha.go") 会把 Beta.go 的边也算上——「要精确就给
        # 限定名」成了空话。简单名那条近似（run 命中 Agent.run）不变。
        graph = CallGraph()
        graph.add(CallEdge("Alpha.go", "helper", "a.py", 10, "python"))
        graph.add(CallEdge("Beta.go", "other", "a.py", 20, "python"))

        self.assertEqual(graph.callees("Alpha.go"), ("helper",))
        self.assertEqual(graph.callers("helper"), ("Alpha.go",))
        # 简单名照旧：两个都算上
        self.assertEqual(graph.callees("go"), ("helper", "other"))

    def test_missing_function_has_no_edges(self):
        graph = self.graph()

        self.assertEqual(graph.callers("nope"), ())
        self.assertEqual(graph.callees("nope"), ())

    def test_edges_filter_and_order(self):
        graph = self.graph()

        self.assertEqual(
            [edge.line for edge in graph.edges(path="agent/agent.py")], [10, 11, 20]
        )
        self.assertEqual(len(graph.edges(caller="Agent.run")), 2)
        self.assertEqual(
            [edge.caller for edge in graph.edges(callee="helper")],
            ["Agent.run", "Agent.step"],
        )

    def test_edge_order_does_not_depend_on_insertion_order(self):
        first = self.graph()
        second = CallGraph()
        for edge in reversed(first.edges()):
            second.add(edge)

        self.assertEqual(first.edges(), second.edges())


class FindPathTest(unittest.TestCase):
    """调用链：BFS 找最短的一条，环不会转不完，顺序可复现。"""

    def chain(self) -> CallGraph:
        """a→b→c，再加一条 c→a 的环。"""
        graph = CallGraph()
        graph.add(CallEdge("a", "b", "m.py", 10, "python"))
        graph.add(CallEdge("b", "c", "m.py", 20, "python"))
        graph.add(CallEdge("c", "a", "m.py", 30, "python"))
        return graph

    def test_a_direct_call(self):
        graph = CallGraph()
        graph.add(CallEdge("a", "b", "m.py", 3, "python"))

        path = graph.find_path("a", "b")

        self.assertEqual(path.nodes, ("a", "b"))
        self.assertEqual(path.length, 1)
        self.assertEqual([(edge.caller, edge.callee, edge.line) for edge in path.edges],
                         [("a", "b", 3)])

    def test_a_multi_level_chain(self):
        path = self.chain().find_path("a", "c")

        self.assertEqual(path.nodes, ("a", "b", "c"))
        self.assertEqual(path.length, 2)

    def test_the_shortest_chain_wins(self):
        graph = CallGraph()
        graph.add(CallEdge("a", "b", "m.py", 10, "python"))
        graph.add(CallEdge("b", "c", "m.py", 20, "python"))
        graph.add(CallEdge("a", "c", "m.py", 30, "python"))  # 直达

        path = graph.find_path("a", "c")

        self.assertEqual(path.nodes, ("a", "c"))  # 不是那条两跳的
        self.assertEqual(path.edges[0].line, 30)

    def test_the_result_does_not_depend_on_insertion_order(self):
        # 两条一样短的路：按 路径、行号 定先后，两张图给同一条
        first = CallGraph()
        first.add(CallEdge("a", "b2", "m.py", 5, "python"))
        first.add(CallEdge("a", "b1", "m.py", 6, "python"))
        first.add(CallEdge("b1", "c", "m.py", 7, "python"))
        first.add(CallEdge("b2", "c", "m.py", 8, "python"))

        second = CallGraph()
        for edge in reversed(first.edges()):
            second.add(edge)

        self.assertEqual(first.find_path("a", "c"), second.find_path("a", "c"))
        self.assertEqual(first.find_path("a", "c").nodes, ("a", "b2", "c"))

    def test_no_path(self):
        graph = self.chain()

        self.assertIsNone(graph.find_path("a", "zzz"))
        self.assertIsNone(CallGraph().find_path("a", "b"))

    def test_start_is_the_target(self):
        graph = self.chain()

        path = graph.find_path("a", "a")

        self.assertEqual(path.nodes, ("a",))
        self.assertEqual(path.length, 0)
        self.assertEqual(path.edges, ())
        # 名字在图里压根不出现时不算「起点就是终点」
        self.assertIsNone(graph.find_path("nope", "nope"))

    def test_cycles_terminate(self):
        graph = self.chain()

        # A→B→C→A 转得回来，但不会转不完；反向也找得到
        self.assertEqual(graph.find_path("a", "c").nodes, ("a", "b", "c"))
        self.assertEqual(graph.find_path("c", "b").nodes, ("c", "a", "b"))
        self.assertIsNone(graph.find_path("b", "zzz"))

    def test_max_depth(self):
        graph = self.chain()

        self.assertIsNone(graph.find_path("a", "c", max_depth=1))
        self.assertEqual(graph.find_path("a", "c", max_depth=2).length, 2)
        # 0 层只够认出「起点就是终点」
        self.assertEqual(graph.find_path("a", "a", max_depth=0).length, 0)
        self.assertIsNone(graph.find_path("a", "b", max_depth=0))

    def test_simple_and_qualified_names(self):
        graph = CallGraph()
        graph.add(CallEdge("Alpha.go", "helper", "a.py", 10, "python"))
        graph.add(CallEdge("Beta.go", "other", "a.py", 20, "python"))

        # 简单名起点：同名的都算起点，Alpha.go 排在前面，先展开它
        self.assertEqual(
            graph.find_path("go", "helper").nodes, ("Alpha.go", "helper")
        )
        # 限定名精确
        self.assertEqual(
            graph.find_path("Alpha.go", "helper").nodes, ("Alpha.go", "helper")
        )
        self.assertIsNone(graph.find_path("Beta.go", "helper"))

    def test_the_middle_of_a_chain_is_expanded_exactly(self):
        # 简单名只开在起点上：链走到中间就不撒网了——这一跳记的是
        # self.helper，图上挂在 Obj.helper 名下，接不下去，宁可说没有
        graph = CallGraph()
        graph.add(CallEdge("Obj.start", "self.helper", "m.py", 10, "python"))
        graph.add(CallEdge("Obj.helper", "leaf", "m.py", 20, "python"))

        self.assertIsNone(graph.find_path("start", "leaf"))
        # 第一跳本身是通的：起点简单名命中，终点精确命中
        self.assertEqual(
            graph.find_path("start", "self.helper").nodes,
            ("Obj.start", "self.helper"),
        )

    def test_a_broken_chain_is_rejected(self):
        edge = CallEdge("a", "b", "m.py", 1, "python")

        with self.assertRaises(ValueError):
            CallPath(nodes=("a",), edges=(edge,))


# ---------------------------------------------------------------------------
# 调用点：Python
# ---------------------------------------------------------------------------


class PythonSitesTest(unittest.TestCase):
    def test_plain_and_method_calls(self):
        sites = python_sites(
            "def f():\n"
            "    helper()\n"
            "    self.step()\n"
            "    pen.stroke(1)\n"
        )

        self.assertEqual(
            [(site.line, site.callee) for site in sites],
            [(2, "helper"), (3, "self.step"), (4, "pen.stroke")],
        )

    def test_calls_in_decorators_and_defaults_are_skipped(self):
        # 这两处在**外层作用域**执行，行号却落在函数自己的范围里：
        # 记下来就会被算到被装饰的函数头上，宁可不记
        sites = python_sites(
            "@deco(tag)\n"
            "def f(x=make()):\n"
            "    helper()\n"
        )

        self.assertEqual([(site.line, site.callee) for site in sites], [(3, "helper")])

    def test_calls_without_a_name(self):
        # 调的是「算出来的东西」：这一条没有名字不记，里面包着的真实调用照记
        sites = python_sites("def f():\n    get()(1)\n    table[0](2)\n")

        self.assertEqual([(site.line, site.callee) for site in sites], [(2, "get")])

    def test_partial_attribute_chain_keeps_the_last_name(self):
        sites = python_sites(
            "def f():\n"
            "    build().run()\n"
            "    items[0].save()\n"
        )

        pairs = sorted((site.line, site.callee) for site in sites)
        self.assertEqual(pairs, [(2, "build"), (2, "run"), (3, "save")])

    def test_nested_definitions_and_lambdas(self):
        sites = python_sites(
            "def outer():\n"
            "    def inner():\n"
            "        helper()\n"
            "    inner()\n"
            "    run = lambda: helper()\n"
            "    run()\n"
        )

        self.assertEqual(
            [(site.line, site.callee) for site in sites],
            [(3, "helper"), (4, "inner"), (5, "helper"), (6, "run")],
        )

    def test_unparsable_source_raises(self):
        with self.assertRaises(SyntaxError):
            python_sites("def broken(:\n")


# ---------------------------------------------------------------------------
# 调用点：C/C++
# ---------------------------------------------------------------------------


class CppSitesTest(unittest.TestCase):
    def test_calls_in_a_function_body(self):
        sites = cpp_sites(
            "#include <cstdio>\n"
            "\n"
            "void helper();\n"
            "\n"
            "void run() {\n"
            "    helper();\n"
            "    obj.foo(1);\n"
            "    ns::foo();\n"
            "    this->step();\n"
            "}\n"
        )

        self.assertEqual(
            [(site.line, site.callee) for site in sites],
            [(6, "helper"), (7, "obj.foo"), (8, "ns.foo"), (9, "this.step")],
        )

    def test_comments_strings_declarations_and_directives(self):
        sites = cpp_sites(
            "#define WRAP(x) helper(x)\n"
            "\n"
            "void run() {\n"
            "    // helper();\n"
            "    const char* text = \"helper(\";\n"
            "    int x(3);\n"
            "    Point p(1, 2);\n"
            "    if (ready()) { go(); }\n"
            "}\n"
        )

        self.assertEqual(
            [(site.line, site.callee) for site in sites],
            [(8, "ready"), (8, "go")],
        )

    def test_declarations_have_no_calls(self):
        sites = cpp_sites(
            "void helper(int x);\n"
            "int compute();\n"
            "struct Box { int get() const; void put(int v); };\n"
        )

        self.assertEqual(sites, [])

    def test_method_body_inside_a_class(self):
        sites = cpp_sites(
            "class Box {\n"
            "public:\n"
            "    void put(int v) { store(v); }\n"
            "    int get() const;\n"
            "};\n"
        )

        self.assertEqual([(site.line, site.callee) for site in sites], [(3, "store")])

    def test_broken_text_does_not_crash(self):
        sites = cpp_sites("}}}} (( void f( { ) helper\nint ;;; class {\n")

        self.assertIsInstance(sites, list)


# ---------------------------------------------------------------------------
# 整份图
# ---------------------------------------------------------------------------


PY_CALLS = (
    "def helper():\n"
    "    pass\n"
    "\n"
    "\n"
    "class Runner:\n"
    "    def run(self):\n"
    "        helper()\n"
    "        self.step()\n"
    "\n"
    "    def step(self):\n"
    "        helper()\n"
)


CPP_CALLS = (
    "namespace app {\n"
    "\n"
    "void helper() { }\n"
    "\n"
    "void run() {\n"
    "    helper();\n"
    "    obj.foo();\n"
    "    ns::foo();\n"
    "}\n"
    "\n"
    "void Renderer::draw() {\n"
    "    clear();\n"
    "}\n"
    "\n"
    "}  // namespace app\n"
)


class CallGraphIndexTest(CallGraphTestBase):
    def test_python_plain_and_method_calls(self):
        self.make("a.py", PY_CALLS)

        index = self.build()

        self.assertEqual(index.callees("Runner.run"), ("helper", "self.step"))
        self.assertEqual(index.callers("helper"), ("Runner.run", "Runner.step"))
        # 被叫的是 self.step，按最后一段也认
        self.assertEqual(index.callers("step"), ("Runner.run",))
        edges = index.edges(path="a.py")
        self.assertEqual(
            [(edge.line, edge.caller, edge.callee) for edge in edges],
            [
                (7, "Runner.run", "helper"),
                (8, "Runner.run", "self.step"),
                (11, "Runner.step", "helper"),
            ],
        )
        self.assertTrue(all(edge.language == "python" for edge in edges))

    def test_cpp_plain_namespace_and_method_calls(self):
        self.make("a.cpp", CPP_CALLS)

        index = self.build()

        self.assertEqual(index.callees("app.run"), ("helper", "ns.foo", "obj.foo"))
        self.assertEqual(index.callees("draw"), ("clear",))
        self.assertEqual(index.callers("clear"), ("app.Renderer.draw",))
        # obj.foo / ns::foo 都按最后一段认
        self.assertEqual(index.callers("foo"), ("app.run",))
        edges = index.edges(path="a.cpp")
        self.assertEqual(
            [(edge.line, edge.caller, edge.callee) for edge in edges],
            [
                (6, "app.run", "helper"),
                (7, "app.run", "obj.foo"),
                (8, "app.run", "ns.foo"),
                (12, "app.Renderer.draw", "clear"),
            ],
        )
        self.assertTrue(all(edge.language == "cpp" for edge in edges))

    def test_missing_function_returns_empty(self):
        self.make("a.py", PY_CALLS)

        index = self.build()

        self.assertEqual(index.callers("nope"), ())
        self.assertEqual(index.callees("nope"), ())

    def test_find_path_over_a_python_chain(self):
        self.make("a.py", PY_CALLS)

        index = self.build()

        path = index.find_path("Runner.run", "helper")
        self.assertEqual(path.nodes, ("Runner.run", "helper"))
        self.assertEqual(path.length, 1)
        # 简单名起点、限定名终点都认
        self.assertEqual(index.find_path("run", "helper").length, 1)
        # self.step 这种按最后一段回解不了「谁是终点」——不做类型推导
        self.assertIsNone(index.find_path("helper", "Runner.run"))

    def test_calls_outside_functions_have_no_caller(self):
        # 模块级 / 全局区的调用没有「哪个函数调的」，不记边
        self.make("a.py", "helper()\n\n\ndef helper():\n    pass\n")
        self.make("b.cpp", "void helper();\n\nint value = helper();\n")

        index = self.build()

        self.assertEqual(len(index.graph), 0)

    def test_same_line_calls_are_one_edge(self):
        self.make("a.py", "def helper():\n    pass\n\n\ndef go():\n    helper(); helper()\n")

        index = self.build()

        self.assertEqual(len(index.edges(caller="go")), 1)
        self.assertEqual(index.callees("go"), ("helper",))

    def test_languages_do_not_mix(self):
        self.make(
            "pypart.py",
            "def helper():\n    return 1\n\n\ndef py_caller():\n    helper()\n",
        )
        self.make(
            "cpppart.cpp",
            "void helper() { }\n\nvoid cpp_caller() {\n    helper();\n}\n",
        )

        index = self.build()

        self.assertEqual(index.callers("helper"), ("cpp_caller", "py_caller"))
        self.assertEqual(
            [(edge.path, edge.language) for edge in index.edges(callee="helper")],
            [("cpppart.cpp", "cpp"), ("pypart.py", "python")],
        )

    def test_unparsable_file_does_not_break_the_rest(self):
        self.make("bad.py", "def broken(:\n")
        self.make(
            "good.py", "def helper():\n    pass\n\n\ndef go():\n    helper()\n"
        )

        index = self.build()

        self.assertEqual(index.callees("go"), ("helper",))
        self.assertEqual(index.edges(path="bad.py"), ())

    def test_symbol_resolves_back(self):
        self.make("a.py", PY_CALLS)

        index = self.build()

        self.assertEqual(index.symbol("Runner.run").kind, METHOD)
        self.assertIsNone(index.symbol("nope"))

    def test_changed_file_is_skipped(self):
        self.make("a.py", "def go():\n    helper()\n")
        symbols = SymbolIndex(root=self.root)
        symbols.build()

        # 索引建完之后文件又被人改了：拿旧行号建边就是错的，整个跳过
        self.make("a.py", "def go():\n    helper()\n    pass\n")

        index = CallGraphIndex(symbols=symbols)
        index.build()

        self.assertEqual(index.edges(), ())
        # 重扫一遍（文件已是新的）：边照常建得出来
        fresh = self.build()
        self.assertEqual(fresh.callees("go"), ("helper",))

    def test_build_twice_gives_the_same_graph(self):
        self.make("a.py", PY_CALLS)

        index = self.build()
        first = index.edges()
        index.build()

        self.assertEqual(index.edges(), first)

    def test_symbols_and_root_are_exclusive(self):
        symbols = SymbolIndex(root=self.root)

        with self.assertRaises(ToolError):
            CallGraphIndex(symbols=symbols, root=self.root)


class RealProjectCallGraphTest(unittest.TestCase):
    """拿本项目真跑一遍：边建得出来，每个 caller 都能对回符号索引。"""

    def test_project_graph(self):
        index = CallGraphIndex()
        index.build()

        self.assertTrue(index.built)
        self.assertGreater(len(index.graph), 0)

        qualified = {symbol.qualified for symbol in index.symbols.symbols()}
        for edge in index.edges():
            self.assertIn(edge.caller, qualified, edge)
            self.assertIn(edge.language, {"python", "cpp"})


if __name__ == "__main__":
    unittest.main()
