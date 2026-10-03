"""RAG 在 Agent 干活链路里的位置。

test_search_chunks.py 钉的是工具本身的输入输出；这里钉的是接力：
提示词把 search_code_chunks 安排在「不知道怎么下手就先缩范围」这一步；
RAG 报出「路径:起止行号」→ read_file 照着读回正文 → edit_file 改 →
编辑结果后面自动接一次验证。

检索走本项目的假 embedding，模型是脚本化的替身——全离线，不联网。
"""

import json
import tempfile
import unittest
from pathlib import Path

from agent import DEFAULT_SYSTEM_PROMPT, Agent
from tests.test_agent import FakeLLMClient, tool_call_reply
from tools import EditFileTool, ReadFileTool, SearchChunksTool, TestTool, ToolRegistry

PRICE = '''\
"""打折计算。"""


def final_price(price, discount):
    """折后价；discount 是 0~1 的折扣率。"""
    return price * discount
'''

SHIPPING = '''\
"""运费计算。"""


def shipping_fee(weight):
    """按公斤算运费。"""
    return weight * 5
'''

PRICE_TEST = '''\
import unittest

from shop.price import final_price


class PriceTest(unittest.TestCase):
    def test_half_off(self):
        self.assertEqual(final_price(100, 0.5), 50)
'''


class PromptPlacementTest(unittest.TestCase):
    """提示词里给 RAG 安排的位置。"""

    def test_the_workflow_line_puts_chunks_in_the_narrowing_step(self):
        # 提示词里提到它的那一行，说的正是「先缩范围再精读」这一步
        line = next(
            text
            for text in DEFAULT_SYSTEM_PROMPT.splitlines()
            if "search_code_chunks" in text
        )

        self.assertIn("缩小范围", line)

    def test_the_description_says_positions_not_code(self):
        # 只给位置是设计，不是巧合：正文撑上下文，而且索引里的副本可能已经过时
        description = SearchChunksTool.description

        self.assertIn("不返回代码正文", description)
        self.assertIn("read_file", description)

    def test_the_description_draws_the_line_against_keyword_search(self):
        # 和 search_code 的分工说清楚了，模型才知道什么时候挑它
        self.assertIn("search_code", SearchChunksTool.description)


class RagWorkflowTest(unittest.TestCase):
    """一条完整链：RAG 定位 → read_file 读正文 → edit_file 改 → 自动验证。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.write("shop/__init__.py", "")
        self.write("shop/price.py", PRICE)
        self.write("shop/shipping.py", SHIPPING)
        self.write("tests/__init__.py", "")
        self.write("tests/test_price.py", PRICE_TEST)

    def write(self, name: str, text: str) -> Path:
        # newline=""：Windows 上文本模式会把 \n 翻成 \r\n，编辑时匹配不上
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def read(self, name: str) -> str:
        return (self.root / name).read_text("utf-8")

    def build_agent(self, replies) -> Agent:
        self.client = FakeLLMClient(replies)
        return Agent(
            client=self.client,
            tools=ToolRegistry(
                [
                    EditFileTool(root=self.root),
                    ReadFileTool(root=self.root),
                    SearchChunksTool(root=self.root),
                    TestTool(root=self.root),
                ]
            ),
        )

    def tool_messages(self, agent: Agent) -> list[str]:
        return [m["content"] for m in agent.messages if m["role"] == "tool"]

    def test_rag_then_read_then_edit_then_verify(self):
        agent = self.build_agent(
            [
                tool_call_reply(("search_code_chunks", '{"query": "final_price"}')),
                tool_call_reply(("read_file", '{"path": "shop/price.py"}')),
                tool_call_reply(
                    (
                        "edit_file",
                        json.dumps(
                            {
                                "path": "shop/price.py",
                                "old_text": "    return price * discount\n",
                                "new_text": (
                                    "    return round(price * discount, 2)\n"
                                ),
                            }
                        ),
                    )
                ),
                "改好了。",
            ]
        )

        answer = agent.run("final_price 的返回值帮我保留两位小数")

        # 四步一条链，每一步都真的发生了（最后那步是改完自动验的）
        self.assertEqual(
            [event.tool for event in agent.get_trace()],
            ["search_code_chunks", "read_file", "edit_file", "run_tests"],
        )

        found, body, edited = self.tool_messages(agent)[0:3]

        # RAG 只报位置：哪个文件、哪几行、是不是 final_price——
        # 正文（price * discount）留给下一步去读，它不该出现
        self.assertIn("shop/price.py:4-6", found)
        self.assertIn("final_price", found)
        self.assertNotIn("price * discount", found)

        # read_file 读回的正文里，第 4 行正是 RAG 指的那一行
        self.assertIn("4|def final_price(price, discount):", body)
        self.assertIn("6|    return price * discount", body)

        # 编辑成功了，验证结果就接在编辑结果后面；测试真的是绿的
        self.assertIn("结果：成功", edited)
        self.assertIn("验证（run_tests）：通过", edited)

        # 磁盘上的文件真的变了；回答以模型的话收尾
        self.assertIn("round(price * discount, 2)", self.read("shop/price.py"))
        self.assertIn("最终成功", answer)
        self.assertTrue(answer.endswith("改好了。"))


if __name__ == "__main__":
    unittest.main()
