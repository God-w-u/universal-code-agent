"""SearchChunksTool 测试：检索调用、结果转换、Agent 侧可见、参数校验。

全部离线：要么在临时目录里搭个小项目，要么在本项目上只读地跑一遍。
最要紧的两条：结果里**只有位置和分数、没有代码正文**（正文会把模型的上下文
撑爆，也让它读到的是索引里的旧副本而不是磁盘上的当前代码），以及工具结果
真的回灌到了模型面前——Agent 的 tool 消息里看得到那些位置。
"""

import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from agent import Agent
from config import PROJECT_ROOT
from tests.test_agent import FakeLLMClient, tool_call_reply
from tools import SearchChunksTool, Tool, ToolError, ToolExecutor, ToolRegistry

READER = '''\
"""读文件的小模块。"""


class FileReader:
    """把文件读成文本。"""

    def read_text(self, path):
        raw = open(path, "rb").read()
        return raw.decode("utf-8")

    def close(self):
        self.handle = None
'''

WRITER = '''\
"""写文件的小模块。"""


def write_text(path, text):
    handle = open(path, "w")
    handle.write(text)
    handle.close()
'''

MANY = '''\
def alpha():
    return 1


def beta():
    return 2


def gamma():
    return 3
'''


class SearchChunksToolTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str) -> Path:
        """精确写入 UTF-8 字节，不做换行转换，保证跨平台一致。"""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(textwrap.dedent(content).encode("utf-8"))
        return path

    def tool(self) -> SearchChunksTool:
        return SearchChunksTool(root=self.root)

    def search(self, **kwargs) -> str:
        return self.tool().run(**kwargs)

    def lines(self, **kwargs) -> list[str]:
        """结果里的位置行（跳过空行和末尾的「…（说明）」行）。"""
        return [
            line
            for line in self.search(**kwargs).splitlines()
            if line and not line.startswith("…（")
        ]


class ResultConversionTest(SearchChunksToolTestBase):
    """Retriever 的 SearchResult 怎么变成给模型看的一行行文字。"""

    def test_a_line_carries_path_symbol_kind_and_score(self):
        self.make("pipeline/reader.py", READER)

        first = self.lines(query="FileReader")[0]

        # 4-12 是类在文件里的真实范围：从 class 那行到最后一个方法体
        self.assertIn("pipeline/reader.py:4-12", first)
        self.assertIn("FileReader", first)
        self.assertIn("[class]", first)
        self.assertRegex(first, r"\[class\]  [0-9.]+")

    def test_a_method_keeps_its_own_line_range(self):
        self.make("pipeline/reader.py", READER)

        text = self.search(query="read_text")

        # 7-9 是方法自己的范围，不是包着它的类那一段
        self.assertIn("pipeline/reader.py:7-9", text)
        self.assertIn("read_text", text)

    def test_the_code_body_is_not_included(self):
        # 命中的就是这段正文（decode 只在方法体里），但结果里不能出现它——
        # 工具只给位置，读原文是 read_file 的事
        self.make("pipeline/reader.py", READER)

        text = self.search(query="decode")

        self.assertIn("pipeline/reader.py:7-9", text)
        self.assertNotIn("raw.decode", text)
        self.assertNotIn("self.handle", text)

    def test_it_says_why_each_hit_matched(self):
        self.make("pipeline/reader.py", READER)

        text = self.search(query="FileReader.read_text")

        self.assertIn("限定名 FileReader.read_text", text)

    def test_results_are_ordered_like_the_retriever_said(self):
        self.make("pipeline/reader.py", READER)
        self.make("pipeline/writer.py", WRITER)

        lines = self.lines(query="write_text")

        self.assertTrue(lines[0].startswith("1. "))
        self.assertIn("pipeline/writer.py", lines[0])
        self.assertEqual(len(lines), 1)  # writer 那一条；reader 里没有 write_text


class NoMatchTest(SearchChunksToolTestBase):
    def test_no_match_says_so_instead_of_raising(self):
        self.make("pipeline/reader.py", READER)

        text = self.search(query="百分之百不存在的词")

        self.assertIn("没有找到", text)

    def test_an_empty_project_returns_nothing(self):
        self.assertIn("没有找到", self.search(query="FileReader"))


class TopKTest(SearchChunksToolTestBase):
    def test_top_k_limits_the_lines(self):
        self.make("m.py", MANY)

        lines = self.lines(query="return", top_k=2)

        self.assertEqual(len(lines), 2)
        self.assertIn("alpha", lines[0])
        self.assertIn("beta", lines[1])

    def test_top_k_zero_is_rejected(self):
        self.make("m.py", MANY)

        with self.assertRaises(ToolError) as caught:
            self.search(query="return", top_k=0)

        self.assertIn("至少为 1", str(caught.exception))

    def test_a_top_k_over_the_cap_is_capped_and_says_so(self):
        self.make("m.py", MANY)

        class CapTwoTool(SearchChunksTool):
            """把上限压到 2：要 10 条时应该只给 2 条，不用真堆 50 条结果。"""

            max_top_k = 2

        text = CapTwoTool(root=self.root).run(query="return", top_k=10)

        self.assertEqual(len(self.lines(query="return", top_k=10)), 3)  # 对照：不压上限有 3 条
        self.assertIn("top_k 最多给到 2", text)
        self.assertEqual(len([line for line in text.splitlines() if line]), 3)  # 2 条结果 + 1 条说明

    def test_no_cap_note_when_top_k_is_within_reach(self):
        self.make("m.py", MANY)

        text = self.search(query="return", top_k=2)

        self.assertNotIn("最多给到", text)


class SearchChunksToolContractTest(SearchChunksToolTestBase):
    """工具协议：名称、描述、参数声明。"""

    def test_it_is_a_tool_with_the_expected_name(self):
        tool = self.tool()

        self.assertIsInstance(tool, Tool)
        self.assertEqual(tool.name, "search_code_chunks")
        self.assertTrue(tool.description)

    def test_query_is_required_and_top_k_is_not(self):
        schema = self.tool().to_schema()["function"]

        self.assertEqual(schema["name"], "search_code_chunks")
        self.assertEqual(schema["parameters"]["required"], ["query"])
        self.assertEqual(schema["parameters"]["properties"]["top_k"]["type"], "integer")

    def test_a_missing_query_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.search()

        self.assertIn("query", str(caught.exception))

    def test_a_non_integer_top_k_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.search(query="FileReader", top_k="two")

        self.assertIn("integer", str(caught.exception))

    def test_an_unknown_parameter_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.search(keyword="FileReader")

        self.assertIn("不接受参数", str(caught.exception))

    def test_a_blank_query_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.search(query="   ")

        self.assertIn("不能为空", str(caught.exception))

    def test_the_default_root_is_the_project_root(self):
        self.assertEqual(SearchChunksTool().root, PROJECT_ROOT)


class ThroughExecutorTest(SearchChunksToolTestBase):
    """走 ToolExecutor：和 Agent 里的路径一致，出错也变成结果而不是异常。"""

    def setUp(self):
        super().setUp()
        self.executor = ToolExecutor(ToolRegistry([self.tool()]))

    def test_executor_runs_it(self):
        self.make("pipeline/reader.py", READER)

        result = self.executor.execute("search_code_chunks", query="FileReader")

        self.assertTrue(result.success)
        self.assertIn("pipeline/reader.py:4-12", result.content)

    def test_a_bad_argument_comes_back_as_a_failed_result(self):
        result = self.executor.execute("search_code_chunks", query="x", top_k=0)

        self.assertFalse(result.success)
        self.assertIn("至少为 1", result.error)


class AgentSeesChunksTest(SearchChunksToolTestBase):
    """Agent 这条线：schema 发给模型、结果回灌到 tool 消息里。"""

    def test_the_tool_schema_is_offered_to_the_model(self):
        client = FakeLLMClient(["回答"])
        agent = Agent(client=client, tools=ToolRegistry([self.tool()]))

        agent.run("问题")

        names = [item["function"]["name"] for item in client.tool_schemas[0]]
        self.assertIn("search_code_chunks", names)

    def test_the_result_is_fed_back_to_the_model(self):
        self.make("pipeline/reader.py", READER)
        client = FakeLLMClient(
            [
                tool_call_reply(
                    ("search_code_chunks", '{"query": "FileReader", "top_k": 3}')
                ),
                "FileReader 在 pipeline/reader.py 里。",
            ]
        )
        agent = Agent(client=client, tools=ToolRegistry([self.tool()]))

        answer = agent.run("FileReader 在哪儿？")

        self.assertEqual(answer, "FileReader 在 pipeline/reader.py 里。")
        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        content = tool_messages[0]["content"]
        self.assertIn("pipeline/reader.py:4-12", content)
        self.assertIn("[class]", content)
        # 回灌的是位置而不是正文——模型要看代码得自己 read_file
        self.assertNotIn("self.handle", content)


class RealProjectSearchChunksTest(unittest.TestCase):
    """在本项目上跑一遍：索引只建一次，几类断言共用。只读，不联网。"""

    @classmethod
    def setUpClass(cls):
        cls.tool = SearchChunksTool()  # 默认 root 就是项目根

    def test_a_qualified_name_lands_on_its_definition(self):
        first = self.tool.run(query="Agent.run", top_k=5).splitlines()[0]

        self.assertTrue(first.startswith("1. agent/agent.py:"))
        self.assertIn("Agent.run", first)

    def test_it_never_returns_code(self):
        text = self.tool.run(query="Agent.run", top_k=5)

        # 真代码才会有的行；出现任何一条都说明正文漏出去了
        self.assertNotIn("def run(self", text)
        self.assertNotIn("chat_with_tools", text)

    def test_an_unknown_query_is_empty(self):
        # 写死的词会命中这个测试文件自己（它也在项目里、也被切了块）。
        # 现拼一个带时间戳的名字：源码里不可能有。
        self.assertIn("没有找到", self.tool.run(query=f"zqxv{time.time_ns()}"))


if __name__ == "__main__":
    unittest.main()
