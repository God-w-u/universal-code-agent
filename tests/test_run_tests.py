"""TestTool 测试：测试通过、测试失败、退出码、Agent 调用。

刻意不去跑本项目真实的测试套件——那等于在测试里再跑一遍全套测试（递归），
而且临时目录里也根本跑不起来。这里搭一个假项目：一个只有几条用例的
tests/ 目录，命令结构（discover -s tests -t .）和真实的完全一样。

全部离线，不联网。
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent
from config import PROJECT_ROOT
from tools import RunCommandTool, TestTool, Tool, ToolError, ToolRegistry
from tools.run_tests import DEFAULT_TEST_TIMEOUT, TEST_COMMAND

PASSING_SOURCE = '''\
import unittest


class SampleTest(unittest.TestCase):
    def test_adds(self):
        self.assertEqual(1 + 1, 2)

    def test_joins(self):
        self.assertEqual("-".join(["a", "b"]), "a-b")
'''

FAILING_SOURCE = '''\
import unittest


class BrokenTest(unittest.TestCase):
    def test_this_one_fails(self):
        self.assertEqual(1, 2, "这条断言是故意写错的")
'''


def parts(result: str) -> tuple[str, str, str]:
    """把结果拆成 (头部, 标准输出, 标准错误) 三段。"""
    head, _, rest = result.partition("\n\n标准输出：\n")
    stdout, _, stderr = rest.partition("\n\n标准错误：\n")
    return head, stdout, stderr


class FakeProjectTest(unittest.TestCase):
    """搭一个假项目：临时目录 + 自己的 tests/，跑的是本机解释器。"""

    # 假项目里的测试文件：文件名 -> 内容。子类按需覆盖。
    sources = {"test_sample.py": PASSING_SOURCE}

    @classmethod
    def setUpClass(cls):
        temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temp.cleanup)
        cls.root = Path(temp.name).resolve()

        tests = cls.root / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("", encoding="utf-8")
        for name, source in cls.sources.items():
            (tests / name).write_text(source, encoding="utf-8")

    def command(self) -> str:
        """和 TEST_COMMAND 同构，只是换成当前这个解释器。"""
        return f'"{sys.executable}" -m unittest discover -s tests -t .'

    def make_tool(self, **kwargs) -> TestTool:
        return TestTool(root=self.root, command=self.command(), **kwargs)

    def run_tests(self, **kwargs) -> str:
        return self.make_tool(**kwargs).run()


class TestToolPassingTest(FakeProjectTest):
    def setUp(self):
        self.result = self.run_tests()
        self.head, self.stdout, self.stderr = parts(self.result)

    def test_a_passing_suite_succeeds(self):
        self.assertTrue(self.result.startswith("命令："))

    def test_the_exit_code_is_zero(self):
        self.assertIn("退出码：0", self.head)

    def test_zero_exit_code_is_not_marked_as_a_failure(self):
        self.assertNotIn("非 0", self.head)

    def test_the_command_is_echoed_back(self):
        self.assertIn(sys.executable, self.head)

    def test_the_test_output_is_kept(self):
        # unittest 把报告写在标准错误里，不是标准输出
        self.assertIn("Ran 2 tests", self.stderr)
        self.assertIn("OK", self.stderr)

    def test_empty_stdout_is_marked(self):
        self.assertIn("（空）", self.stdout)

    def test_all_four_pieces_are_present(self):
        # 命令、退出码、标准输出、标准错误，缺一不可
        self.assertIn("命令：", self.head)
        self.assertIn("退出码：", self.head)
        self.assertIn("标准输出：", self.result)
        self.assertIn("标准错误：", self.result)


class TestToolFailingTest(FakeProjectTest):
    sources = {"test_broken.py": FAILING_SOURCE}

    def setUp(self):
        self.result = self.run_tests()
        self.head, self.stdout, self.stderr = parts(self.result)

    def test_a_failing_suite_reports_a_non_zero_exit_code(self):
        self.assertIn("退出码：1", self.head)

    def test_a_failing_suite_does_not_raise(self):
        self.assertIsInstance(self.result, str)

    def test_failure_is_stated_in_words(self):
        self.assertIn("非 0，命令失败", self.head)

    def test_the_failure_report_is_kept(self):
        self.assertIn("FAILED", self.stderr)

    def test_the_failing_assertion_message_is_kept(self):
        # 模型要看到「哪条用例挂了、期望什么、实际什么」才能继续推理
        self.assertIn("这条断言是故意写错的", self.stderr)

    def test_the_number_of_tests_is_kept(self):
        self.assertIn("Ran 1 test", self.stderr)


class TestToolTimeoutTest(unittest.TestCase):
    """超时用例刻意不继承 FakeProjectTest。

    理由和 test_run_command.py 里那组一样：被杀的只是 shell，睡着的子进程
    还活着并以 root 为工作目录，Windows 上临时目录会删不掉（WinError 32）。
    所以拿项目根目录当 root，命令也不碰任何文件。
    """

    def make_tool(self, **kwargs) -> TestTool:
        command = f'"{sys.executable}" -c "import time; time.sleep(5)"'
        return TestTool(root=PROJECT_ROOT, command=command, **kwargs)

    def test_a_slow_suite_times_out(self):
        started = time.perf_counter()
        result = self.make_tool(timeout=0.3).run()
        elapsed = time.perf_counter() - started

        self.assertIn("超时", result)
        # 见 test_run_command.py 里同名测试的说明：这条路必须是「立刻返回」，
        # 不能等 sleep 自己醒来
        self.assertLess(elapsed, 3)

    def test_the_timeout_is_not_raised(self):
        self.assertIsInstance(self.make_tool(timeout=0.3).run(), str)

    def test_the_timeout_reaches_run_command(self):
        # 超时不是在这里实现的，是转手给 RunCommandTool 的，得确认真的传下去了
        self.assertEqual(self.make_tool(timeout=7).command_tool.timeout, 7)


class TestToolShapeTest(unittest.TestCase):
    def test_it_is_a_tool(self):
        self.assertIsInstance(TestTool(root=PROJECT_ROOT), Tool)
        self.assertEqual(TestTool.name, "run_tests")

    def test_it_reuses_run_command(self):
        # 这一层不许自己再写一遍 subprocess / 超时 / 输出处理
        self.assertIsInstance(
            TestTool(root=PROJECT_ROOT).command_tool, RunCommandTool
        )

    def test_the_boundary_is_shared_with_run_command(self):
        tool = TestTool(root=PROJECT_ROOT)

        self.assertEqual(tool.command_tool.root, tool.root)

    def test_the_default_command_is_the_pinned_one(self):
        self.assertEqual(TestTool().command, TEST_COMMAND)
        self.assertEqual(
            TEST_COMMAND, "python -m unittest discover -s tests -t ."
        )

    def test_the_default_timeout_is_sane(self):
        # 跑测试比跑一条普通命令慢得多，默认值不能被压到 run_command 那个量级
        self.assertGreaterEqual(DEFAULT_TEST_TIMEOUT, 60)

    def test_schema_shape(self):
        parameters = TestTool(root=PROJECT_ROOT).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(list(parameters["properties"]), ["timeout"])
        self.assertEqual(parameters["properties"]["timeout"]["type"], "number")
        # 没有必需参数时这个键应当缺席，而不是空列表
        self.assertNotIn("required", parameters)

    def test_it_takes_no_arguments(self):
        # 只校验、不执行：真的 run() 一下会去跑本项目自己的测试套件，
        # 也就是「在测试里跑测试」，一跑就递归了
        TestTool(root=PROJECT_ROOT).validate_args({})

    def test_default_root_is_project_root(self):
        self.assertEqual(TestTool().root, PROJECT_ROOT)

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            TestTool(root=Path("C:/"))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            TestTool(root=Path("C:/definitely/not/a/real/dir/xyz"))


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


class TestToolAgentTest(FakeProjectTest):
    """走一遍完整的 Agent 调用链：模型要求跑测试 → 结果回灌。"""

    def build(self, replies):
        return Agent(
            client=StubClient(replies),
            tools=ToolRegistry([self.make_tool()]),
        )

    def test_agent_can_run_the_tests(self):
        agent = self.build(
            [tool_call_reply("run_tests", "{}"), final_reply("测试全过了")]
        )

        answer = agent.run("跑一下测试")

        self.assertEqual(answer, "测试全过了")
        self.assertIn("退出码：0", agent.messages[3]["content"])
        self.assertIn("Ran 2 tests", agent.messages[3]["content"])

    def test_the_call_is_recorded_in_the_trace(self):
        agent = self.build(
            [tool_call_reply("run_tests", "{}"), final_reply("好了")]
        )

        agent.run("跑一下测试")

        events = agent.get_trace()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].tool, "run_tests")
        self.assertTrue(events[0].success)

    def test_a_timeout_does_not_break_the_turn(self):
        # 超时同样按正常结果回灌，模型能据此改主意。
        # root 用项目根目录，理由同 TestToolTimeoutTest。
        slow = TestTool(
            root=PROJECT_ROOT,
            command=f'"{sys.executable}" -c "import time; time.sleep(5)"',
            timeout=0.3,
        )
        agent = Agent(
            client=StubClient(
                [tool_call_reply("run_tests", "{}"), final_reply("太慢了，算了")]
            ),
            tools=ToolRegistry([slow]),
        )

        answer = agent.run("跑一下测试")

        self.assertEqual(answer, "太慢了，算了")
        self.assertIn("超时", agent.messages[3]["content"])

    def test_it_is_registered_in_main(self):
        # main.build_agent() 会去读 API Key，这里把客户端换掉再装配
        import main

        with patch("agent.agent.LLMClient", StubClient):
            agent = main.build_agent()

        self.assertIn("run_tests", [tool.name for tool in agent.tools.list()])


class TestToolAgentFailureTest(FakeProjectTest):
    """测试失败时，Agent 这一轮照样得走完。"""

    sources = {"test_broken.py": FAILING_SOURCE}

    def test_a_failing_suite_does_not_break_the_turn(self):
        agent = Agent(
            client=StubClient(
                [tool_call_reply("run_tests", "{}"), final_reply("有测试挂了")]
            ),
            tools=ToolRegistry([self.make_tool()]),
        )

        answer = agent.run("跑一下测试")

        self.assertEqual(answer, "有测试挂了")
        self.assertIn("退出码：1", agent.messages[3]["content"])
        self.assertTrue(agent.get_trace()[0].success)


if __name__ == "__main__":
    unittest.main()
