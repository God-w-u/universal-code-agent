"""call_graph 工具测试：查得对、给的是位置不是正文、接得上 Agent。

工具本身的输入输出在这里钉；Call Graph 的提取和查询在
test_callgraph.py 里（12.1）。源文件夹具全写在临时目录里，Agent 用脚本化
的假客户端——全离线，不联网。
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from agent import Agent
from tests.test_agent import FakeLLMClient, tool_call_reply
from tools import CallGraphTool, ReadFileTool, ToolExecutor, ToolRegistry
from tools.base import ToolError

APP = '''\
"""小模块：给调用图工具用。"""


def helper():
    return 1


def other():
    return 2


def caller_one():
    return helper()


class Alpha:
    def go(self):
        helper()


class Beta:
    def go(self):
        other()


class Runner:
    def run(self):
        helper()
        self.step()

    def step(self):
        return 1
'''

CPP = '''\
namespace app {

void helper() { }

void run() {
    helper();
    obj.foo();
}
'''

CHAIN = '''\
"""一条调用链。"""


def bottom():
    return 1


def middle():
    return bottom()


def top():
    return middle()


def shortcut():
    bottom()
    middle()


def island():
    return 0
'''


class ToolCase(unittest.TestCase):
    """搭一个临时项目，工具指着它跑。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.tool = CallGraphTool(root=self.root)

    def write(self, name: str, text: str) -> Path:
        # newline=""：Windows 上文本模式会把 \n 翻成 \r\n，行号就对不上了
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return path


class CallersTest(ToolCase):
    def setUp(self):
        super().setUp()
        self.write("app.py", APP)

    def test_callers_are_named_with_their_definition(self):
        # 位置给的是**调用方自己的定义处**：拿到就能去 read_file 读它
        self.assertEqual(
            self.tool.run(function="helper", direction="callers"),
            "helper 的调用者（共 3 个）：\n"
            "1. app.py:17-18  Alpha.go  [method]\n"
            "2. app.py:27-29  Runner.run  [method]\n"
            "3. app.py:12-13  caller_one  [function]",
        )

    def test_a_call_site_is_matched_by_its_last_segment(self):
        # self.step() 记的是 "self.step"，按最后一段认，所以 run 也算 step 的调用者
        self.assertEqual(
            self.tool.run(function="step", direction="callers"),
            "step 的调用者（共 1 个）：\n"
            "1. app.py:27-29  Runner.run  [method]",
        )


class CalleesTest(ToolCase):
    def setUp(self):
        super().setUp()
        self.write("app.py", APP)

    def test_callees_of_a_qualified_name(self):
        self.assertEqual(
            self.tool.run(function="Runner.run", direction="callees"),
            "Runner.run 调用了谁（共 2 个）：\n"
            "1. app.py:4-5  helper  [function]\n"
            "2. app.py:31-32  Runner.step  [method]",
        )

    def test_a_simple_name_matches_all_of_them(self):
        # go 同时是 Alpha.go 和 Beta.go：两个都查，两个都算
        self.assertEqual(
            self.tool.run(function="go", direction="callees"),
            "go 调用了谁（共 2 个）：\n"
            "1. app.py:4-5  helper  [function]\n"
            "2. app.py:8-9  other  [function]",
        )

    def test_a_qualified_name_is_exact(self):
        # 给限定名就只要这一个——和上面那条对照着看
        self.assertEqual(
            self.tool.run(function="Alpha.go", direction="callees"),
            "Alpha.go 调用了谁（共 1 个）：\n"
            "1. app.py:4-5  helper  [function]",
        )


class TopKTest(ToolCase):
    def setUp(self):
        super().setUp()
        self.write("app.py", APP)

    def test_top_k_truncates_and_says_so(self):
        text = self.tool.run(function="helper", direction="callers", top_k=2)

        self.assertIn("1. app.py:17-18", text)
        self.assertIn("2. app.py:27-29", text)
        self.assertNotIn("3. app.py:12-13", text)
        # 截断了必须说：不说的话模型分不清「就这么多」和「只给了前几条」
        self.assertIn("共 3 个，只显示前 2 个", text)

    def test_no_note_when_nothing_was_cut(self):
        text = self.tool.run(function="helper", direction="callers")

        self.assertNotIn("只显示前", text)

    def test_a_top_k_over_the_cap_is_capped(self):
        text = self.tool.run(function="helper", direction="callers", top_k=80)

        self.assertIn("top_k 最多给到 50", text)
        self.assertIn("3. app.py:12-13", text)  # 只有 3 条，一条都没少

    def test_top_k_must_be_at_least_one(self):
        with self.assertRaises(ToolError):
            self.tool.run(function="helper", direction="callers", top_k=0)


class MissingAndEmptyTest(ToolCase):
    def setUp(self):
        super().setUp()
        self.write("app.py", APP)

    def test_a_function_that_does_not_exist(self):
        text = self.tool.run(function="nope", direction="callers")

        self.assertIn("没有找到", text)
        self.assertIn("nope", text)

    def test_a_function_without_callers(self):
        # 函数在、就是没人调它：这是「空结果」，不是「查不到这个函数」
        text = self.tool.run(function="Alpha.go", direction="callers")

        self.assertEqual(text, "（Alpha.go 没有已知的调用者）")

    def test_a_function_without_callees(self):
        # other 在、就是谁也不调它
        text = self.tool.run(function="other", direction="callees")

        self.assertEqual(text, "（other 没有调用任何已知的函数）")

    def test_a_bad_direction_is_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(function="helper", direction="upwards")

    def test_an_empty_function_name_is_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(function="   ", direction="callers")


class OutputShapeTest(ToolCase):
    """只给位置：正文留给 read_file，索引里的副本可能已经过时。"""

    def setUp(self):
        super().setUp()
        self.write("app.py", APP)

    def test_no_source_code_in_the_output(self):
        for direction in ("callers", "callees"):
            with self.subTest(direction=direction):
                text = self.tool.run(function="helper", direction=direction)

                self.assertNotIn("def ", text)
                self.assertNotIn("return 1", text)
                self.assertNotIn("helper()", text)

    def test_the_position_is_there(self):
        text = self.tool.run(function="caller_one", direction="callees")

        self.assertIn("app.py:4-5", text)

    def test_a_callee_outside_the_symbols_falls_back_to_the_call_site(self):
        # 库函数 / 宏对不上符号索引：退回调用点，还是能去 read_file 看上下文
        self.write("lib.py", "def go():\n    print(1)\n")

        text = self.tool.run(function="go", direction="callees")

        self.assertIn("lib.py:2", text)
        self.assertIn("print", text)
        self.assertIn("调用点", text)

    def test_the_graph_is_built_once_and_reused(self):
        self.tool.run(function="helper", direction="callers")

        with patch.object(
            self.tool.index, "build", side_effect=AssertionError("又扫了一遍")
        ):
            text = self.tool.run(function="helper", direction="callees")

        self.assertIn("helper", text)


class CppTest(ToolCase):
    """C/C++ 走的是同一条路：提取器换了，工具一行都不用改。"""

    def setUp(self):
        super().setUp()
        self.write("a.cpp", CPP)

    def test_callees_with_namespace_and_method(self):
        self.assertEqual(
            self.tool.run(function="app.run", direction="callees"),
            "app.run 调用了谁（共 2 个）：\n"
            "1. a.cpp:3-3  app.helper  [function]\n"
            "2. a.cpp:7  obj.foo  （调用点；符号索引里没有定义）",
        )

    def test_callers(self):
        self.assertEqual(
            self.tool.run(function="helper", direction="callers"),
            "helper 的调用者（共 1 个）：\n"
            "1. a.cpp:5-8  app.run  [function]",
        )


class PathModeTest(ToolCase):
    """direction=path：从 A 怎么走到 B，只给链和位置。"""

    def setUp(self):
        super().setUp()
        self.write("chain.py", CHAIN)

    def test_a_multi_level_chain(self):
        self.assertEqual(
            self.tool.run(function="top", direction="path", target="bottom"),
            "top → middle → bottom（3 个函数，2 次调用）：\n"
            "1. chain.py:12-13  top  [function]\n"
            "2. chain.py:8-9  middle  [function]\n"
            "3. chain.py:4-5  bottom  [function]",
        )

    def test_a_direct_call_is_one_hop(self):
        self.assertEqual(
            self.tool.run(function="middle", direction="path", target="bottom"),
            "middle → bottom（2 个函数，1 次调用）：\n"
            "1. chain.py:8-9  middle  [function]\n"
            "2. chain.py:4-5  bottom  [function]",
        )

    def test_the_shortest_chain_wins(self):
        # shortcut 直接调 bottom，也经 middle 绕一下：给最短的那条
        self.assertEqual(
            self.tool.run(function="shortcut", direction="path", target="bottom"),
            "shortcut → bottom（2 个函数，1 次调用）：\n"
            "1. chain.py:16-18  shortcut  [function]\n"
            "2. chain.py:4-5  bottom  [function]",
        )

    def test_start_is_the_target(self):
        self.assertEqual(
            self.tool.run(function="middle", direction="path", target="middle"),
            "middle → middle（起点就是终点，长度 0）：\n"
            "1. chain.py:8-9  middle  [function]",
        )

    def test_no_path_says_why(self):
        # island 谁也不调：起点就没有出边，链无从谈起
        text = self.tool.run(function="island", direction="path", target="bottom")

        self.assertIn("没有找到", text)
        self.assertIn("没有出边", text)

    def test_a_target_that_is_not_there(self):
        text = self.tool.run(function="top", direction="path", target="nope")

        self.assertIn("没有找到", text)
        self.assertIn("没有入边", text)

    def test_a_missing_name_is_not_a_crash(self):
        text = self.tool.run(function="nope", direction="path", target="nope")

        self.assertIn("没有找到", text)

    def test_a_break_in_the_middle_says_what_to_do(self):
        # 两头都在、就是没走通：除了「加深」还得提一句另一种断法——
        # 链中间经对象调的那一步（self.x / obj.x）是接不下去的
        text = self.tool.run(
            function="top", direction="path", target="bottom", max_depth=1
        )

        self.assertIn("加大 max_depth", text)
        self.assertIn("callers / callees", text)

    def test_max_depth_defaults_and_caps(self):
        # 不给就用默认 8 层，这条两级链走得通
        self.assertIn(
            "top → middle → bottom",
            self.tool.run(function="top", direction="path", target="bottom"),
        )

        # 1 层走不到 bottom：它在第二跳
        short = self.tool.run(
            function="top", direction="path", target="bottom", max_depth=1
        )
        self.assertIn("在 1 层内没走通", short)

        # 超上限截到 20：链照样给，但要说明（和 top_k 一个规矩）
        capped = self.tool.run(
            function="top", direction="path", target="bottom", max_depth=50
        )
        self.assertIn("max_depth 最多给到 20", capped)
        self.assertIn("top → middle → bottom", capped)

    def test_max_depth_must_be_at_least_one(self):
        with self.assertRaises(ToolError):
            self.tool.run(
                function="top", direction="path", target="bottom", max_depth=0
            )

    def test_path_needs_a_target(self):
        with self.assertRaises(ToolError):
            self.tool.run(function="top", direction="path")

    def test_names_follow_the_existing_rules(self):
        self.write("app.py", APP)

        # 简单名起点：go 命中的两个里 Alpha.go 排在前面，先展开它
        self.assertEqual(
            self.tool.run(function="go", direction="path", target="helper"),
            "Alpha.go → helper（2 个函数，1 次调用）：\n"
            "1. app.py:17-18  Alpha.go  [method]\n"
            "2. app.py:4-5  helper  [function]",
        )
        # 限定名精确：Beta.go 走的是 other，不是 helper
        self.assertIn(
            "没有找到",
            self.tool.run(function="Beta.go", direction="path", target="helper"),
        )

    def test_no_source_code_in_the_output(self):
        text = self.tool.run(function="top", direction="path", target="bottom")

        self.assertNotIn("def ", text)
        self.assertNotIn("return 1", text)
        self.assertNotIn("bottom()", text)

    def test_a_step_outside_the_symbols_falls_back_to_the_call_site(self):
        # 中间的库函数对不上符号索引：退回调用点，还是能去 read_file 看上下文
        self.write("lib.py", "def go():\n    print(1)\n")

        self.assertEqual(
            self.tool.run(function="go", direction="path", target="print"),
            "go → print（2 个函数，1 次调用）：\n"
            "1. lib.py:1-2  go  [function]\n"
            "2. lib.py:2  print  （调用点；符号索引里没有定义）",
        )

    def test_the_graph_is_not_rebuilt_for_path_queries(self):
        self.tool.run(function="top", direction="path", target="bottom")

        with patch.object(
            self.tool.index, "build", side_effect=AssertionError("又扫了一遍")
        ):
            text = self.tool.run(function="top", direction="path", target="bottom")

        self.assertIn("top → middle → bottom", text)

    def test_callers_and_callees_still_work(self):
        # path 加进来没动老的两条路：查完链，老查询原样
        self.tool.run(function="middle", direction="path", target="bottom")

        self.assertEqual(
            self.tool.run(function="bottom", direction="callers"),
            "bottom 的调用者（共 2 个）：\n"
            "1. chain.py:8-9  middle  [function]\n"
            "2. chain.py:16-18  shortcut  [function]",
        )
        self.assertEqual(
            self.tool.run(function="shortcut", direction="callees"),
            "shortcut 调用了谁（共 2 个）：\n"
            "1. chain.py:4-5  bottom  [function]\n"
            "2. chain.py:8-9  middle  [function]",
        )


class RegistrationTest(ToolCase):
    def test_it_registers_under_its_name(self):
        registry = ToolRegistry([CallGraphTool(root=self.root)])

        self.assertIn("call_graph", registry)
        self.assertIsInstance(registry.get("call_graph"), CallGraphTool)
        self.assertEqual(registry.get("call_graph").name, "call_graph")

    def test_the_project_registry_has_it(self):
        # main.build_registry() 里漏了就红——和提示词对账测试是同一条防线
        found = [
            tool for tool in main.build_registry().list()
            if isinstance(tool, CallGraphTool)
        ]

        self.assertEqual(len(found), 1)


class ExecutorCompatibilityTest(ToolCase):
    """走既有的执行层：工具报错变成 ToolResult，不会把 Agent 带崩。"""

    def setUp(self):
        super().setUp()
        self.write("app.py", APP)
        self.executor = ToolExecutor(
            ToolRegistry([CallGraphTool(root=self.root)])
        )

    def test_a_normal_query_succeeds(self):
        result = self.executor.execute(
            "call_graph", function="helper", direction="callers"
        )

        self.assertTrue(result.success)
        self.assertIn("Runner.run", result.content)

    def test_a_missing_function_is_an_answer_not_a_failure(self):
        # 查不到是正常结果，工具照常返回文本
        result = self.executor.execute(
            "call_graph", function="nope", direction="callees"
        )

        self.assertTrue(result.success)
        self.assertIn("没有找到", result.content)

    def test_a_bad_argument_becomes_a_failed_result(self):
        result = self.executor.execute(
            "call_graph", function="helper", direction="upwards"
        )

        self.assertFalse(result.success)
        self.assertIn("direction", result.error)

    def test_a_missing_argument_becomes_a_failed_result(self):
        result = self.executor.execute("call_graph", function="helper")

        self.assertFalse(result.success)
        self.assertIn("direction", result.error)

    def test_a_path_query_succeeds(self):
        result = self.executor.execute(
            "call_graph", function="go", direction="path", target="helper"
        )

        self.assertTrue(result.success)
        self.assertIn("Alpha.go → helper", result.content)

    def test_a_path_that_finds_nothing_is_still_a_result(self):
        # 找不着链不是失败：Agent 拿到的是文本，自己决定换名字还是收手
        result = self.executor.execute(
            "call_graph", function="helper", direction="path", target="nope"
        )

        self.assertTrue(result.success)
        self.assertIn("没有找到", result.content)

    def test_path_without_a_target_becomes_a_failed_result(self):
        result = self.executor.execute(
            "call_graph", function="helper", direction="path"
        )

        self.assertFalse(result.success)
        self.assertIn("target", result.error)


class AgentWorkflowTest(ToolCase):
    """真接一次 Agent（模型是脚本替身）：call_graph → read_file，全离线。"""

    def setUp(self):
        super().setUp()
        self.write("app.py", APP)

    def build_agent(self, replies) -> Agent:
        self.client = FakeLLMClient(replies)
        return Agent(
            client=self.client,
            tools=ToolRegistry(
                [CallGraphTool(root=self.root), ReadFileTool(root=self.root)]
            ),
        )

    def tool_messages(self, agent: Agent) -> list[str]:
        return [m["content"] for m in agent.messages if m["role"] == "tool"]

    def test_call_graph_then_read_file(self):
        # 任务名是通用词：模型是脚本化的，输入文本本来就不影响结果
        agent = self.build_agent(
            [
                tool_call_reply(
                    ("call_graph", '{"function": "helper", "direction": "callers"}')
                ),
                tool_call_reply(("read_file", '{"path": "app.py"}')),
                "看完了。",
            ]
        )

        answer = agent.run("任务甲")

        self.assertEqual(
            [event.tool for event in agent.get_trace()], ["call_graph", "read_file"]
        )

        found, body = self.tool_messages(agent)[0:2]

        # 第一步只给位置：调用方是谁、在哪个文件的哪几行——正文留给下一步
        self.assertIn("Runner.run", found)
        self.assertIn("app.py:27-29", found)
        self.assertNotIn("def ", found)

        # 第二步照着位置读回正文：第 27 行正是 call_graph 指的那一行
        self.assertIn("27|    def run(self):", body)
        self.assertIn("28|        helper()", body)

        self.assertTrue(answer.endswith("看完了。"))

    def test_the_agent_can_follow_a_chain(self):
        # 多级调用：call_graph 给出整条链，再照着链上的位置去 read_file
        self.write("chain.py", CHAIN)
        agent = self.build_agent(
            [
                tool_call_reply(
                    (
                        "call_graph",
                        '{"function": "top", "direction": "path", "target": "bottom"}',
                    )
                ),
                tool_call_reply(("read_file", '{"path": "chain.py"}')),
                "链走通了。",
            ]
        )

        answer = agent.run("任务丙")

        self.assertEqual(
            [event.tool for event in agent.get_trace()], ["call_graph", "read_file"]
        )

        found, body = self.tool_messages(agent)[0:2]

        self.assertIn("top → middle → bottom", found)
        self.assertIn("chain.py:8-9", found)
        self.assertNotIn("def ", found)

        # read_file 读回正文：第 8 行正是链上中间那一步
        self.assertIn("8|def middle():", body)
        self.assertIn("9|    return bottom()", body)

        self.assertTrue(answer.endswith("链走通了。"))

    def test_a_missing_function_does_not_stop_the_agent(self):
        agent = self.build_agent(
            [
                tool_call_reply(
                    ("call_graph", '{"function": "nope", "direction": "callers"}')
                ),
                "没找到这个函数，我换个名字再试。",
            ]
        )

        answer = agent.run("任务乙")

        events = agent.get_trace()
        self.assertEqual([event.tool for event in events], ["call_graph"])
        self.assertTrue(events[0].success)
        self.assertIn("没有找到", self.tool_messages(agent)[0])
        self.assertTrue(answer.endswith("我换个名字再试。"))


if __name__ == "__main__":
    unittest.main()
