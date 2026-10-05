"""位置提取（tools/locations.py）的测试：**只出位置，不出正文；认不出来就不猜**。

这层存在的理由只有一个：工作台要显示「这次调用落在哪个文件的哪一段」，而
工具的返回正文从 13.2 起就不进 Trace、也不进任何一帧 SSE（正文里可能有整份
源码、编译输出）。所以测试钉的是两条：

- 认得出来的：路径 / 行号 / 起止行 / 符号 / 类型各就各位；
- 认不出来的（白名单外、格式变了、压根没匹配上）：空列表——不硬凑、不按
  别的行倒推。
- 另外：相似度分数、诊断消息、命中的源码行这些**正文内容一个都不许出现**
  在结果里，测试直接拿它们做「不该出现」的断言。

下面每段样例都是从各工具**真实输出格式**里抄的（搜出来的格式见
tools/search_code.py 的 _render、tools/callgraph.py 的 _symbol_text /
_edge_text、tools/edit_file.py 的 as_text、tools/diagnostics.py 的 render），
不是凭空编的；真工具的输出怎么走到这里来在 tests/test_trace.py 的
ExecutorLocationsTest。
"""

import unittest

from tools.locations import MAX_LOCATIONS, MAX_PATH, MAX_SYMBOL, extract

# 白名单外（或压根不存在）的工具名，给什么正文都不该出位置
OUTSIDE_THE_WHITELIST = [
    "run_command",
    "run_tests",
    "calculator",
    "read_file",
    "list_files",
    "code_summary",
    "不存在的工具",
]

# 各种格式的位置行混在一起，用来喂白名单外的工具（一条都不该被认走）
LOCATION_LOOKING_JUNK = "\n".join(
    [
        "1. index/chunks.py:12-40  ChunkIndex.build  [method]",
        "main.cpp:3:5: error: 'x' was not declared in this scope",
        "agent/agent.py:42: def run(self):",
        "文件：a.py",
        "位置：第 7 行（原文替换，命中 1 处）",
        "1. tools/x.py:3  别的名字  （调用点；符号索引里没有定义）",
    ]
)


class ExtractBoundaryTest(unittest.TestCase):
    """这一层的分寸：只认白名单，认不出来就给空的。"""

    def test_a_tool_outside_the_whitelist_never_gives_locations(self):
        # 白名单是这一层最重要的分寸：往 run_command 的输出里捞「文件:行号」
        # 会捞出一堆似是而非的东西（测试输出里随便一行都可能是）
        for tool in OUTSIDE_THE_WHITELIST:
            with self.subTest(tool=tool):
                self.assertEqual(extract(tool, LOCATION_LOOKING_JUNK), [])

    def test_empty_content_gives_nothing(self):
        for tool in ("search_code", "search_code_chunks", "call_graph", "compile", "edit_file"):
            with self.subTest(tool=tool):
                self.assertEqual(extract(tool, ""), [])

    def test_whitelisted_tools_survive_junk_without_raising(self):
        # 提位置是纯附加的活儿，不能因为正文长得怪就把执行层带崩
        for tool in ("search_code", "search_code_chunks", "call_graph", "compile", "build", "edit_file"):
            with self.subTest(tool=tool):
                extract(tool, LOCATION_LOOKING_JUNK)
                extract(tool, "\n" * 100)
                extract(tool, "1. :0  [x]\n2. a:b-c  d  []")

    def test_nothing_matched_is_an_empty_list(self):
        self.assertEqual(extract("search_code", "（没有找到包含 'x' 的行）"), [])
        self.assertEqual(extract("call_graph", "（没有找到 'x'：符号索引里没有这个函数）"), [])
        self.assertEqual(extract("search_code_chunks", "（没有找到相关代码块）"), [])


class SearchCodeTest(unittest.TestCase):
    """search_code：`路径:行号: 那行代码` —— 只要前两截。"""

    def test_each_line_becomes_path_and_line(self):
        content = "\n".join(
            [
                "agent/agent.py:42: def run(self):",
                "agent/agent.py:80:         return self.messages[-1]",
                "tools/executor.py:61:     def execute(self, name: str, **arguments: Any) -> ToolResult:",
            ]
        )

        self.assertEqual(
            extract("search_code", content),
            [
                {"path": "agent/agent.py", "line": 42},
                {"path": "agent/agent.py", "line": 80},
                {"path": "tools/executor.py", "line": 61},
            ],
        )

    def test_the_matched_source_line_never_shows_up(self):
        result = extract("search_code", "agent/agent.py:42: def run(self):")

        self.assertEqual(result, [{"path": "agent/agent.py", "line": 42}])
        recorded = str(result)
        for leaked in ("def run", "self", "("):
            self.assertNotIn(leaked, recorded)

    def test_windows_paths_with_a_numeric_segment(self):
        # 路径里带冒号、带数字目录：非贪婪匹配要从右边那个真冒号切开
        self.assertEqual(
            extract("search_code", r"C:\proj\2\main.py:7: int main() {"),
            [{"path": r"C:\proj\2\main.py", "line": 7}],
        )

    def test_lines_that_are_not_matches_are_skipped(self):
        content = "\n".join(
            [
                "agent/agent.py:42: def run(self):",
                "…（匹配过多，只显示前 200 条）",
                "…（有 3 个文件超过 512 KB，只搜索了开头部分）",
                "",
            ]
        )

        self.assertEqual(extract("search_code", content), [{"path": "agent/agent.py", "line": 42}])

    def test_a_header_line_is_skipped(self):
        self.assertEqual(extract("search_code", "找到 0 处："), [])


class SearchChunksTest(unittest.TestCase):
    """search_code_chunks：`N. 路径:起-止  符号  [类型]  分数  原因`。"""

    def test_path_range_symbol_and_kind(self):
        content = "\n".join(
            [
                "1. index/chunks.py:12-40  ChunkIndex.build  [method]  88.5  名字里的 build（+60）",
                "2. index/chunks.py:1-11  ChunkIndex  [class]  40  限定名（+40）",
            ]
        )

        self.assertEqual(
            extract("search_code_chunks", content),
            [
                {
                    "path": "index/chunks.py",
                    "line": 12,
                    "end_line": 40,
                    "symbol": "ChunkIndex.build",
                    "kind": "method",
                },
                {
                    "path": "index/chunks.py",
                    "line": 1,
                    "end_line": 11,
                    "symbol": "ChunkIndex",
                    "kind": "class",
                },
            ],
        )

    def test_the_score_and_the_reason_never_show_up(self):
        content = "1. index/chunks.py:12-40  ChunkIndex.build  [method]  88.5  正文里的 build（+60）"

        result = extract("search_code_chunks", content)

        recorded = str(result)
        self.assertNotIn("88.5", recorded)
        self.assertNotIn("正文里的", recorded)
        self.assertNotIn("+60", recorded)

    def test_a_truncation_note_is_skipped(self):
        content = "\n".join(
            [
                "1. a.py:1-2  f  [function]  1  名字（+1）",
                "…（top_k 最多给到 50）",
            ]
        )

        self.assertEqual(
            extract("search_code_chunks", content),
            [{"path": "a.py", "line": 1, "end_line": 2, "symbol": "f", "kind": "function"}],
        )


class CallGraphTest(unittest.TestCase):
    """call_graph：调用者 / 被调用者列表和调用链的每一跳。"""

    CALLERS = "\n".join(
        [
            "leaf 的调用者（共 1 个）：",
            "1. alpha.py:5-6  middle  [function]",
        ]
    )

    PATH = "\n".join(
        [
            "top → middle → leaf（3 个函数，2 次调用）：",
            "1. alpha.py:9-10  top  [function]",
            "2. alpha.py:5-6  middle  [function]",
            "3. alpha.py:1-2  leaf  [function]",
        ]
    )

    def test_callers_list(self):
        self.assertEqual(
            extract("call_graph", self.CALLERS),
            [{"path": "alpha.py", "line": 5, "end_line": 6, "symbol": "middle", "kind": "function"}],
        )

    def test_a_path_keeps_the_order_of_the_chain(self):
        # 顺序就是链的顺序：前端照着画 A → B → C，不能打乱
        self.assertEqual(
            [item["symbol"] for item in extract("call_graph", self.PATH)],
            ["top", "middle", "leaf"],
        )

    def test_the_arrow_header_is_not_a_location(self):
        # 头一行是「A → B → C（…）：」，不是位置行
        self.assertEqual(extract("call_graph", "top → top（起点就是终点，长度 0）：").__len__(), 0)

    def test_a_call_site_fallback_line(self):
        # 对不上符号索引的那些退回调用点：只有路径和行号，没有类型
        self.assertEqual(
            extract("call_graph", "1. alpha.py:2  print  （调用点；符号索引里没有定义）"),
            [{"path": "alpha.py", "line": 2, "symbol": "print"}],
        )

    def test_truncation_notes_are_skipped(self):
        content = self.PATH + "\n…（共 12 个，只显示前 3 个）"

        self.assertEqual(len(extract("call_graph", content)), 3)

    def test_the_empty_and_no_path_messages_give_nothing(self):
        self.assertEqual(extract("call_graph", "（f 没有已知的调用者）"), [])
        self.assertEqual(
            extract("call_graph", "（没有找到 a → b 的调用链：在 8 层内没走通；两头都在）"),
            [],
        )


class DiagnosticsTest(unittest.TestCase):
    """compile / build 末尾的诊断块：`路径:行[:列]: 级别: 消息`。"""

    def test_error_with_a_column(self):
        self.assertEqual(
            extract("compile", "  /tmp/proj/main.cpp:3:5: error: 'x' was not declared"),
            [{"path": "/tmp/proj/main.cpp", "line": 3, "kind": "error"}],
        )

    def test_warning_without_a_column(self):
        self.assertEqual(
            extract("build", "  main.cpp:9: warning: unused variable 'y'"),
            [{"path": "main.cpp", "line": 9, "kind": "warning"}],
        )

    def test_note_lines_are_recognised_too(self):
        self.assertEqual(
            extract("build", "  main.cpp:12: note: declared here"),
            [{"path": "main.cpp", "line": 12, "kind": "note"}],
        )

    def test_a_whole_block(self):
        content = "\n".join(
            [
                "诊断（共 2 条：1 个 error，1 个 warning）：",
                "  /tmp/proj/main.cpp:3:5: error: 'x' was not declared in this scope",
                "  /tmp/proj/main.cpp:9: warning -Wunused-variable: unused variable 'y'",
            ]
        )

        self.assertEqual(
            extract("compile", content),
            [
                {"path": "/tmp/proj/main.cpp", "line": 3, "kind": "error"},
                {"path": "/tmp/proj/main.cpp", "line": 9, "kind": "warning"},
            ],
        )

    def test_the_message_and_the_header_never_show_up(self):
        content = "\n".join(
            [
                "诊断（共 1 条：1 个 error）：",
                "  main.cpp:3:5: error: 'secret' was not declared in this scope",
            ]
        )

        result = extract("compile", content)

        recorded = str(result)
        self.assertNotIn("secret", recorded)
        self.assertNotIn("诊断（共", recorded)

    def test_a_diagnostic_shaped_english_sentence_is_not_a_location(self):
        self.assertEqual(extract("compile", "编译失败，退出码：1"), [])
        self.assertEqual(extract("compile", "error: 3 errors generated"), [])


class EditFileTest(unittest.TestCase):
    """edit_file：头部两行——`文件：路径` + `位置：第 n 行…`。"""

    def test_a_single_line_replacement(self):
        content = "\n".join(
            [
                "结果：成功",
                "文件：alpha.py",
                "位置：第 2 行（原文替换，命中 1 处）",
                "修改前：",
                "  return 1",
                "修改后：",
                "  return 2",
            ]
        )

        self.assertEqual(
            extract("edit_file", content),
            [{"path": "alpha.py", "line": 2, "kind": "原文替换"}],
        )

    def test_a_range_replacement(self):
        content = "\n".join(
            [
                "结果：成功",
                "文件：src/agent.py",
                "位置：第 3–5 行（按行替换）",
                "修改前：",
                "…",
            ]
        )

        self.assertEqual(
            extract("edit_file", content),
            [{"path": "src/agent.py", "line": 3, "end_line": 5, "kind": "按行替换"}],
        )

    def test_the_before_and_after_code_never_show_up(self):
        content = "\n".join(
            [
                "结果：成功",
                "文件：alpha.py",
                "位置：第 2 行（原文替换，命中 1 处）",
                "修改前：",
                "  secret_token = 1",
                "修改后：",
                "  secret_token = 2",
            ]
        )

        recorded = str(extract("edit_file", content))

        self.assertNotIn("secret_token", recorded)

    def test_a_missing_location_line_gives_a_path_without_a_line_number(self):
        # 格式变了（只有「文件：」）：宁可给一条没有行号的，也不编一个
        content = "结果：成功\n文件：alpha.py\n改好了\n"

        self.assertEqual(extract("edit_file", content), [{"path": "alpha.py", "line": 0}])

    def test_content_without_a_file_line_gives_nothing(self):
        self.assertEqual(extract("edit_file", "结果：失败\n错误：找不到这个文件"), [])

    def test_the_path_line_is_only_looked_for_near_the_top(self):
        # 「文件：」出现在正文很靠后的地方（比如被改的代码里就有这么一行），不当头部
        content = "\n".join(["结果：成功"] + ["填充"] * 20 + ["文件：其实在正文里"])

        self.assertEqual(extract("edit_file", content), [])


class LimitsTest(unittest.TestCase):
    """上限：条数封顶、路径/符号/类型都截断——这是给界面显示的。"""

    def test_the_number_of_locations_is_capped(self):
        content = "\n".join(
            f"a.py:{number}: 第 {number} 行" for number in range(1, MAX_LOCATIONS + 30)
        )

        result = extract("search_code", content)

        self.assertEqual(len(result), MAX_LOCATIONS)
        self.assertEqual(result[0]["line"], 1)
        self.assertEqual(result[-1]["line"], MAX_LOCATIONS)

    def test_long_paths_are_truncated(self):
        long_path = "deep/" * 100 + "main.py"
        content = f"{long_path}:7: int main() {{"

        path = extract("search_code", content)[0]["path"]

        self.assertLessEqual(len(path), MAX_PATH)
        self.assertTrue(path.endswith("…"))

    def test_long_symbols_are_truncated(self):
        long_symbol = "a" * 500
        content = f"1. main.py:1-2  {long_symbol}  [function]  1  名字（+1）"

        symbol = extract("search_code_chunks", content)[0]["symbol"]

        self.assertLessEqual(len(symbol), MAX_SYMBOL)
        self.assertTrue(symbol.endswith("…"))

    def test_a_zero_line_number_is_dropped(self):
        # 行号从 1 起：0 说明这行不是正经的位置行，宁可不给
        self.assertEqual(extract("search_code", "a.py:0: 不该有这一条"), [])
        self.assertEqual(extract("search_code_chunks", "1. a.py:0-0  f  [function]  1  名字（+1）"), [])

    def test_an_end_line_that_is_not_after_the_start_is_dropped(self):
        self.assertEqual(
            extract("search_code_chunks", "1. a.py:9-9  f  [function]  1  名字（+1）"),
            [{"path": "a.py", "line": 9, "symbol": "f", "kind": "function"}],
        )


if __name__ == "__main__":
    unittest.main()
