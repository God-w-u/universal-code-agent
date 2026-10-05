"""RepairLoop 测试：验不过就接着修，修到什么时候为止、什么时候必须停。

这里测的不是「验证准不准」（那是 test_verify 的事），而是**循环本身**：
什么时候算一轮、什么时候接着修、什么时候停，以及停下来的时候代码和报告
各是什么状态。所以假替身还是那些：假的 compile / build / run_tests 按剧本
返回文本，假的客户端按剧本回话，只有最后一条是真的（装了编译器才跑）——
真编译两轮：第一次没修完、第二次修完，退出码 0。

全部离线，不联网。
"""

import json
import tempfile
import unittest
from pathlib import Path

from agent import DEFAULT_MAX_REPAIR_ROUNDS, Agent
from agent.repair import RepairLoop, _digest, _edit_summary, _fingerprint
from agent.verify import COMPILE, EDIT, RUN_TESTS, Verifier
from tests.test_verify import FakeVerifyTool, ran
from tools import (
    CompileTool,
    EditFileTool,
    TestTool,
    ToolExecutor,
    ToolRegistry,
)
from tools.compile import find_compiler

# 一个编译不过的小文件：第 2 行少个分号
BAD = "int main() {\n    int answer = 42\n    return 0;\n}\n"

# edit_file 返回的那种结果文本。指纹和改动摘要都从「位置：」那一行取东西
RESULT = (
    "结果：成功\n"
    "文件：main.cpp\n"
    "位置：第 2 行（原文替换，命中 1 处）\n"
    "修改前：\n  int answer = 42\n"
    "修改后：\n  int answer = 42;"
)


class StubClient:
    """按剧本回话的假客户端。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.seen = []  # 每次请求收到的消息

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


def tool_call_reply(name, arguments, call_id="c1"):
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


def final_reply(content="好的"):
    return {"role": "assistant", "content": content}


class RepairTestBase(unittest.TestCase):
    """一个临时项目目录，外加装 Agent 的几个快捷方式。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.write("main.cpp", BAD)

    def write(self, name, text, encoding="utf-8"):
        # newline="" 是必须的：Windows 上文本模式会把 \n 翻成 \r\n
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding, newline="")
        return path

    def read(self, name="main.cpp"):
        return (self.root / name).read_text("utf-8")

    def fake(self, name, replies):
        return FakeVerifyTool(name, replies, root=self.root)

    def edit(self, old_text="    int answer = 42", new_text="    int answer = 42;", **kwargs):
        arguments = {"path": "main.cpp", "old_text": old_text, "new_text": new_text}
        arguments.update(kwargs)
        return tool_call_reply(EDIT, json.dumps(arguments))

    def build_agent(self, replies, *tools, **kwargs):
        self.client = StubClient(replies)
        return Agent(
            client=self.client,
            tools=ToolRegistry([EditFileTool(root=self.root), *tools]),
            **kwargs,
        )

    def tool_messages(self, agent):
        return [m["content"] for m in agent.messages if m["role"] == "tool"]


# ---------------------------------------------------------------------------
# 一轮：验一次没过，改了就好了
# ---------------------------------------------------------------------------


class OneRoundTest(RepairTestBase):
    def test_a_failure_then_a_passing_fix_is_one_round(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                final_reply("修好了"),
            ],
            compile_tool,
        )

        answer = agent.run("编译不过，改一下")

        # 验证过了：模型正常给最终回答，不是被报告顶掉——
        # 前面多一句摘要（7.6），它后面还是模型自己那句话
        self.assertEqual(
            answer, "修复 1 轮；第 1 轮编译通过；最终成功。\n\n修好了"
        )
        self.assertIsNone(agent.repair.stopped)
        self.assertFalse(agent.repair.pending)

        rounds = agent.repair.rounds
        self.assertEqual(len(rounds), 1)
        self.assertTrue(rounds[0].passed)
        # 先坏后修的那种不是「一轮通过」：状态还是卡在哪一关的老说法（7.8）
        self.assertFalse(rounds[0].one_shot)
        self.assertEqual(rounds[0].state, "编译通过")
        self.assertEqual(rounds[0].number, 1)
        self.assertEqual(rounds[0].verifier, "compile main.cpp")
        self.assertEqual(rounds[0].exit_code, 0)
        self.assertEqual(rounds[0].path, "main.cpp")
        self.assertIn("第 2 行", rounds[0].edit)

        self.assertIn("第 1 轮通过", self.tool_messages(agent)[-1])

    def test_an_edit_that_passes_first_time_is_one_round(self):
        # 模型没先编译、直接改，改完第一次验就过：7.8 起也算一轮（一轮通过）。
        # 7.5–7.7 这种什么都不记，于是记录里会出现「明明改好了，却是 0 轮」
        compile_tool = self.fake(COMPILE, [ran("g++ main.cpp", 0)])
        agent = self.build_agent([self.edit(), final_reply("好了")], compile_tool)

        answer = agent.run("改一下")

        # 摘要说「第 1 轮就通过」，后面还是模型自己那句话
        self.assertEqual(answer, "修复 1 轮；第 1 轮就通过；最终成功。\n\n好了")

        rounds = agent.repair.rounds
        self.assertEqual(len(rounds), 1)
        self.assertTrue(rounds[0].one_shot)
        self.assertTrue(rounds[0].passed)
        self.assertEqual(rounds[0].state, "一轮通过")
        # 轮次该有的东西一样不少
        self.assertEqual(rounds[0].path, "main.cpp")
        self.assertIn("第 2 行", rounds[0].edit)
        self.assertEqual(rounds[0].verifier, "compile main.cpp")
        self.assertEqual(rounds[0].exit_code, 0)
        self.assertEqual(len(rounds[0].fingerprint), 8)
        self.assertEqual(agent.repair.summary(), "修复 1 轮；第 1 轮就通过；最终成功。")
        self.assertIn("第 1 轮通过", self.tool_messages(agent)[-1])
        self.assertIsNone(agent.repair.stopped)
        self.assertEqual(len(self.client.seen), 2)  # 没有被多问一轮

    def test_every_passing_edit_is_its_own_round(self):
        # 一次 run 里改了两处、两处都直接过：一次 edit_file 就是一轮，两轮
        compile_tool = self.fake(
            COMPILE, [ran("g++ main.cpp", 0), ran("g++ main.cpp", 0)]
        )
        agent = self.build_agent(
            [
                self.edit(),
                self.edit(
                    old_text="    int answer = 42;",
                    new_text="    int answer = 42; // 又改了一次",
                ),
                final_reply("好了"),
            ],
            compile_tool,
        )

        answer = agent.run("改两处")

        self.assertEqual(
            answer, "修复 2 轮；第 1 轮就通过；第 2 轮就通过；最终成功。\n\n好了"
        )
        self.assertEqual([r.one_shot for r in agent.repair.rounds], [True, True])
        self.assertEqual([r.number for r in agent.repair.rounds], [1, 2])


# ---------------------------------------------------------------------------
# 接着修：模型自己继续，或者被按回去继续
# ---------------------------------------------------------------------------


class ContinueTest(RepairTestBase):
    def test_the_model_edits_again_on_its_own(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                final_reply("这回真好了"),
            ],
            compile_tool,
        )

        answer = agent.run("编译不过，改一下")

        self.assertTrue(answer.endswith("这回真好了"))
        self.assertIn("修复 2 轮；第 1 轮编译没过；第 2 轮编译通过；最终成功。", answer)
        self.assertEqual([r.passed for r in agent.repair.rounds], [False, True])
        self.assertEqual([r.number for r in agent.repair.rounds], [1, 2])
        self.assertFalse(agent.repair.pending)

    def test_a_model_that_wants_to_stop_is_pushed_back(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                self.edit(),  # 改完自动验证没过
                final_reply("应该好了"),  # 模型想收尾——不行
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                final_reply("这次过了"),
            ],
            compile_tool,
        )

        answer = agent.run("改一下")

        self.assertTrue(answer.endswith("这次过了"))
        users = [m["content"] for m in agent.messages if m["role"] == "user"]
        self.assertEqual(len(users), 2)  # 用户那句，加上自动修复催的那句
        self.assertIn("先别收尾", users[-1])
        self.assertIn("1/5 轮", users[-1])
        self.assertEqual([r.passed for r in agent.repair.rounds], [False, True])

    def test_the_nudge_request_carries_the_new_diagnostics(self):
        # 下一轮模型上下文里必须看得到：已消失 / 仍在 / 新增
        compile_tool = self.fake(
            COMPILE,
            [
                # 模型先自己编了一遍：两条错
                ran(
                    "g++ main.cpp",
                    2,
                    err=(
                        "main.cpp:2:15: error: expected ';'\n"
                        "main.cpp:3:12: error: 'x' was not declared"
                    ),
                ),
                # 第一轮改完：';' 没了，冒出来一条新的
                ran(
                    "g++ main.cpp",
                    2,
                    err=(
                        "main.cpp:3:12: error: 'x' was not declared\n"
                        "main.cpp:4:9: error: 'y' was not declared"
                    ),
                ),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                final_reply("我觉得好了"),  # 想收尾——会被催一句
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                final_reply("好了"),
            ],
            compile_tool,
        )

        agent.run("改一下")

        # 被催之后的那次请求里，前面那条工具结果带着完整的对比
        nudged = self.client.seen[3]
        tool_text = [m["content"] for m in nudged if m["role"] == "tool"][-1]
        self.assertIn("已消失", tool_text)
        self.assertIn("仍在", tool_text)
        self.assertIn("新增", tool_text)
        self.assertIn("'x' was not declared", tool_text)

    def test_a_new_error_shows_up_in_the_diff(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran(
                    "g++ main.cpp",
                    2,
                    err=(
                        "main.cpp:2:15: error: expected ';'\n"
                        "main.cpp:3:12: error: 'x' was not declared"
                    ),
                ),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                self.edit(),
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                self.edit(old_text="int main() {", new_text="int main() { // ok"),
                final_reply("好了"),
            ],
            compile_tool,
        )

        agent.run("改一下")

        added = agent.repair.rounds[1].diff.added
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].line, 3)
        self.assertEqual(len(agent.repair.rounds[1].diff.still), 1)

    def test_a_model_that_never_edits_again_stops_with_its_own_words(self):
        compile_tool = self.fake(
            COMPILE, [ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")]
        )
        agent = self.build_agent(
            [self.edit(), final_reply("修不动"), final_reply("真的修不动")],
            compile_tool,
        )

        answer = agent.run("改一下")

        self.assertIn("自动修复停止", answer)
        self.assertIn("模型没有再修改代码", answer)
        self.assertIn("真的修不动", answer)
        self.assertEqual(len(agent.repair.rounds), 1)


# ---------------------------------------------------------------------------
# 停下来：到上限、原地打转、改不动
# ---------------------------------------------------------------------------


class StopTest(RepairTestBase):
    """三条停止条件各来一条；附带着确认「停下 = 保留现状 + 说清状态」。"""

    def three_edits(self):
        return [
            self.edit(),
            self.edit(old_text="    return 0;", new_text="    return 0; // a"),
            self.edit(old_text="int main() {", new_text="int main() { // b"),
        ]

    def test_the_cap_stops_the_loop_at_once(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 2, err="main.cpp:3:1: error: A"),
                ran("g++ main.cpp", 2, err="main.cpp:4:1: error: B"),
            ],
        )
        agent = self.build_agent(
            self.three_edits(), compile_tool, max_repair_rounds=3
        )

        answer = agent.run("改一下")

        self.assertIn("自动修复停止", answer)
        self.assertIn("上限 3 轮", answer)
        self.assertIn("各轮记录", answer)
        self.assertIn("第 1 轮", answer)
        self.assertIn("第 3 轮", answer)

        self.assertEqual(len(agent.repair.rounds), 3)
        self.assertEqual([r.passed for r in agent.repair.rounds], [False] * 3)
        # 到上限就地停：不再问模型下一次该干什么
        self.assertEqual(len(self.client.seen), 3)
        # 最后一次修改原样保留，没有回滚
        self.assertIn("// b", self.read())

    def test_two_identical_rounds_stop_the_loop(self):
        same = ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")
        compile_tool = self.fake(COMPILE, [same, same, same])
        agent = self.build_agent(self.three_edits(), compile_tool)

        answer = agent.run("改一下")

        self.assertIn("自动修复停止", answer)
        self.assertIn("诊断完全相同", answer)
        # 前两轮：第一轮没有基线可比，第二轮一比，一模一样
        self.assertEqual(len(agent.repair.rounds), 2)
        self.assertEqual(len(self.client.seen), 2)

    def test_an_edit_that_changes_nothing_stops_the_loop(self):
        failing = ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")
        compile_tool = self.fake(COMPILE, [failing, failing])
        agent = self.build_agent(
            [
                self.edit(),
                # 第二次：原文和目标一模一样。edit_file 会照常报成功、文件也被
                # 重写了一遍——可内容一个字节没变，诊断当然也不会变
                self.edit(old_text="    int answer = 42;", new_text="    int answer = 42;"),
            ],
            compile_tool,
        )

        answer = agent.run("改一下")

        self.assertIn("自动修复停止", answer)
        self.assertIn("一模一样", answer)
        self.assertEqual(len(agent.repair.rounds), 2)

    def test_a_refused_edit_during_repair_stops_the_loop(self):
        compile_tool = self.fake(
            COMPILE, [ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")]
        )
        agent = self.build_agent(
            [
                self.edit(),
                tool_call_reply(
                    EDIT,
                    json.dumps(
                        {
                            "path": "main.cpp",
                            "old_text": "这段原文根本不在文件里",
                            "new_text": "x",
                        }
                    ),
                ),
            ],
            compile_tool,
        )

        answer = agent.run("改一下")

        self.assertIn("没有改成", answer)
        self.assertEqual(len(agent.repair.rounds), 1)
        # 第一次的修改还在，第二次没改成
        self.assertIn("    int answer = 42;", self.read())

    def test_running_out_of_steps_reports_instead_of_rolling_back(self):
        # 修复还欠着的时候撞上 max_steps：收尾报状态，而不是抛 AgentError
        # 把这一轮整个回滚掉（改了半天的文件不能就这么没了）
        compile_tool = self.fake(
            COMPILE, [ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")]
        )
        agent = self.build_agent(
            [self.edit(), final_reply("先歇会儿")], compile_tool, max_steps=2
        )

        answer = agent.run("改一下")

        self.assertIn("用满 2 轮", answer)
        self.assertIn("    int answer = 42;", self.read())
        # 报告也进了历史：下一轮模型知道上一轮停在哪
        self.assertEqual(agent.messages[-1]["content"], answer)


# ---------------------------------------------------------------------------
# 两关：编译过了还得看测试（7.6）
# ---------------------------------------------------------------------------


class TestStageTest(RepairTestBase):
    """编译通过只是第一关。这一组盯的是第二关：测试。

    「这个项目有测试」的标志是 tests 包在（run_tests 跑的那条 discover 只认它），
    所以这一组每个用例都先把它放上——不然测试那一关会整个跳过。
    """

    FAILED_OUTPUT = (
        "F\n"
        "======================================================================\n"
        "FAIL: test_it_prints_five (tests.test_program.ProgramTest)\n"
        "----------------------------------------------------------------------\n"
        "AssertionError: '-1' != '5'\n"
        "\n"
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.003s\n"
        "\n"
        "FAILED (failures=1)"
    )

    def setUp(self):
        super().setUp()
        # 有它才算「这个项目有测试」——光有目录不算，discover 会直接报
        # 「Start directory is not importable」
        self.write("tests/__init__.py", "")

    def failing_tests(self):
        output = self.FAILED_OUTPUT
        return self.fake(RUN_TESTS, [ran("python -m unittest", 1, out=output)])

    def test_compile_passing_with_a_failing_test_is_not_a_success(self):
        # 编得过、测试全红：那不叫修好
        agent = self.build_agent(
            [self.edit()],
            self.fake(COMPILE, [ran("g++ main.cpp", 0)]),
            self.failing_tests(),
            max_repair_rounds=1,
        )

        answer = agent.run("改一下")

        round_ = agent.repair.rounds[0]
        self.assertTrue(round_.compiled)  # 编译这一关过了
        self.assertFalse(round_.test_passed)  # 测试那一关没过
        self.assertFalse(round_.passed)  # 所以这轮不算修好
        self.assertEqual(round_.state, "测试没过")
        self.assertEqual(round_.test_exit_code, 1)
        self.assertIn("run_tests 没通过（退出码 1）", round_.test)

        # 测试红在哪儿，模型看得到
        self.assertIn("测试输出", self.tool_messages(agent)[0])
        self.assertIn("AssertionError", self.tool_messages(agent)[0])

        self.assertIn("最终没修好", answer)
        self.assertIn("上限 1 轮", answer)

    def test_compile_and_test_both_passing_is_a_success(self):
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 0),
                ],
            ),
            self.fake(RUN_TESTS, [ran("python -m unittest", 0, out="OK")]),
        )

        answer = agent.run("编译不过，改一下")

        round_ = agent.repair.rounds[0]
        self.assertTrue(round_.passed)
        self.assertEqual(round_.state, "编译通过、测试通过")
        self.assertEqual(round_.test_exit_code, 0)
        self.assertIn("run_tests 通过（退出码 0）", round_.test)

        self.assertIn("自动修复：第 1 轮通过（compile main.cpp；run_tests 通过", self.tool_messages(agent)[-1])
        self.assertIn("修复 1 轮；第 1 轮编译通过、测试通过；最终成功。", answer)

    def test_a_failing_test_is_fixed_in_a_second_round(self):
        agent = self.build_agent(
            [
                self.edit(),
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                final_reply("这回真好了"),
            ],
            self.fake(COMPILE, [ran("g++ main.cpp", 0), ran("g++ main.cpp", 0)]),
            self.fake(
                RUN_TESTS,
                [
                    ran("python -m unittest", 1, out=self.FAILED_OUTPUT),
                    ran("python -m unittest", 0, out="OK"),
                ],
            ),
        )

        answer = agent.run("测试没过，改一下")

        self.assertEqual([r.compiled for r in agent.repair.rounds], [True, True])
        self.assertEqual([r.test_passed for r in agent.repair.rounds], [False, True])
        self.assertEqual([r.passed for r in agent.repair.rounds], [False, True])
        self.assertFalse(agent.repair.pending)
        # 第一轮之后那句「接着修」说清楚卡在哪一关：编译过了，是测试没过
        self.assertIn("卡在测试上", self.tool_messages(agent)[0])
        self.assertIn(
            "修复 2 轮；第 1 轮测试没过；第 2 轮编译通过、测试通过；最终成功。", answer
        )
        self.assertTrue(answer.endswith("这回真好了"))

    def test_a_cap_of_test_failures_stops_the_loop(self):
        # 测试一直红：到上限就停，不无限修下去
        agent = self.build_agent(
            [
                self.edit(),
                self.edit(old_text="    return 0;", new_text="    return 0; // a"),
            ],
            self.fake(COMPILE, [ran("g++ main.cpp", 0)]),
            self.fake(
                RUN_TESTS,
                [
                    ran("python -m unittest", 1, out=self.FAILED_OUTPUT),
                    ran("python -m unittest", 1, out="FAILED (errors=2)：另一个错"),
                ],
            ),
            max_repair_rounds=2,
        )

        answer = agent.run("测试没过，改一下")

        self.assertEqual(len(agent.repair.rounds), 2)
        self.assertEqual([r.test_passed for r in agent.repair.rounds], [False, False])
        self.assertIn("上限 2 轮", answer)
        self.assertIn("// a", self.read())  # 最后一次修改留着，没回滚

    def test_the_same_test_failure_twice_stops_the_loop(self):
        # 两轮的测试输出只有耗时那一行不一样（0.003s / 0.418s）：那还是「一模一样」
        agent = self.build_agent(
            [
                self.edit(),
                self.edit(old_text="    return 0;", new_text="    return 0; // a"),
            ],
            self.fake(COMPILE, [ran("g++ main.cpp", 0)]),
            self.fake(
                RUN_TESTS,
                [
                    ran("python -m unittest", 1, out=self.FAILED_OUTPUT),
                    ran(
                        "python -m unittest",
                        1,
                        out=self.FAILED_OUTPUT.replace("0.003s", "0.418s"),
                    ),
                ],
            ),
        )

        answer = agent.run("测试没过，改一下")

        self.assertIn("连着两轮的测试输出完全相同", answer)
        self.assertEqual(len(agent.repair.rounds), 2)

    def test_a_project_without_tests_only_has_the_compile_stage(self):
        # 项目里没有能跑的测试：跳过第二关，但要说一声，别让「没跑」看着像「过了」
        (self.root / "tests" / "__init__.py").unlink()
        test_tool = self.failing_tests()
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 0),
                ],
            ),
            test_tool,
        )

        answer = agent.run("编译不过，改一下")

        self.assertEqual(test_tool.calls, [])  # 一次都没跑
        round_ = agent.repair.rounds[0]
        self.assertIsNone(round_.test_passed)
        self.assertEqual(round_.state, "编译通过")
        self.assertIn("测试：这个项目里没有能跑的测试", self.tool_messages(agent)[-1])
        self.assertIn("第 1 轮编译通过；最终成功", answer)

    def test_a_failed_compile_does_not_run_the_tests(self):
        # 编都编不过，产物还是改之前那个：跑测试测的不是这次的代码
        test_tool = self.failing_tests()
        agent = self.build_agent(
            [self.edit()],
            self.fake(COMPILE, [ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")]),
            test_tool,
            max_repair_rounds=1,
        )

        agent.run("改一下")

        self.assertEqual(test_tool.calls, [])
        self.assertIsNone(agent.repair.rounds[0].test_passed)
        self.assertEqual(agent.repair.rounds[0].state, "编译没过")

    def test_the_two_stages_show_up_in_the_trace(self):
        test_tool = self.fake(RUN_TESTS, [ran("python -m unittest", 0, out="OK")])
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 0),
                ],
            ),
            test_tool,
        )

        agent.run("编译不过，改一下")

        self.assertEqual(
            [event.tool for event in agent.get_trace()],
            [COMPILE, EDIT, COMPILE, RUN_TESTS],
        )


# ---------------------------------------------------------------------------
# 同一处改来改去：指纹重复就停（7.6）
# ---------------------------------------------------------------------------


class FingerprintTest(RepairTestBase):
    def three_edits_back_and_forth(self):
        """A 改成 B、B 改回 A、A 又改成 B：第三次的指纹和第一次重合。"""
        return [
            self.edit(old_text="    int answer = 42", new_text="    int answer = 42;"),
            self.edit(old_text="    int answer = 42;", new_text="    int answer = 42"),
            self.edit(old_text="    int answer = 42", new_text="    int answer = 42;"),
        ]

    def test_a_repeated_edit_fingerprint_stops_the_loop(self):
        # 头两轮的诊断故意不一样：不然「连着两轮诊断相同」会先一步把它拦下，
        # 就走不到第三轮那个重复指纹上了
        agent = self.build_agent(
            self.three_edits_back_and_forth(),
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 2, err="main.cpp:3:1: error: 另一个错"),
                    ran("g++ main.cpp", 2, err="main.cpp:3:1: error: 另一个错"),
                ],
            ),
        )

        answer = agent.run("改一下")

        self.assertIn("检测到重复修改", answer)
        self.assertIn("停止继续尝试", answer)
        self.assertEqual(len(agent.repair.rounds), 3)
        # 头一轮和第三轮的指纹是同一个：同一处、改成同样的内容
        self.assertEqual(
            agent.repair.rounds[2].fingerprint, agent.repair.rounds[0].fingerprint
        )
        self.assertNotEqual(
            agent.repair.rounds[1].fingerprint, agent.repair.rounds[0].fingerprint
        )

    def test_a_second_place_in_the_same_file_keeps_going(self):
        # 同一个文件改了好几处地方不算「重复修改」——那两个错本来就该改两次
        agent = self.build_agent(
            [
                self.edit(old_text="    int answer = 42", new_text="    int answer = 42;"),
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 0),
                ],
            ),
        )

        answer = agent.run("改一下")

        self.assertIsNone(agent.repair.stopped)
        self.assertEqual(len(agent.repair.rounds), 2)
        self.assertNotEqual(
            agent.repair.rounds[0].fingerprint, agent.repair.rounds[1].fingerprint
        )
        self.assertTrue(answer.endswith("好了"))


# ---------------------------------------------------------------------------
# 一轮里都记了什么
# ---------------------------------------------------------------------------


class RoundRecordTest(RepairTestBase):
    def setUp(self):
        super().setUp()
        self.write("tests/__init__.py", "")

    def test_a_round_carries_both_stages(self):
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 0),
                ],
            ),
            self.fake(RUN_TESTS, [ran("python -m unittest", 0, out="OK")]),
        )

        agent.run("编译不过，改一下")

        round_ = agent.repair.rounds[0]
        self.assertEqual(round_.number, 1)
        self.assertEqual(round_.path, "main.cpp")
        self.assertIn("第 2 行", round_.edit)
        self.assertEqual(len(round_.fingerprint), 8)
        self.assertEqual(round_.verifier, "compile main.cpp")
        self.assertEqual(round_.exit_code, 0)
        self.assertEqual(round_.diff.still, ())  # 上一轮那条错没了
        self.assertTrue(round_.compiled)
        self.assertTrue(round_.test_passed)
        self.assertEqual(round_.test_exit_code, 0)
        self.assertTrue(round_.passed)
        self.assertEqual(round_.state, "编译通过、测试通过")
        self.assertIn("指纹", round_.as_line())
        self.assertIn("run_tests 通过", round_.as_line())

    def test_a_round_without_a_second_stage_says_so(self):
        # 连编译都没过：测试那一关没跑，轮次记录里写明白「测试没跑」，
        # 而不是留空让人以为是过了
        agent = self.build_agent(
            [self.edit()],
            self.fake(COMPILE, [ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")]),
            max_repair_rounds=1,
        )

        agent.run("改一下")

        self.assertEqual(agent.repair.rounds[0].test, "测试没跑")
        self.assertIn("测试没跑", agent.repair.rounds[0].as_line())


# ---------------------------------------------------------------------------
# 验不了就该别动：验证工具自己没跑起来
# ---------------------------------------------------------------------------


class UnverifiableTest(RepairTestBase):
    def test_a_verification_that_cannot_run_does_not_start_the_loop(self):
        class Broken(FakeVerifyTool):
            def execute(self, **kwargs):
                raise RuntimeError("这台机器上没有 C++ 编译器")

        agent = self.build_agent(
            [self.edit(), final_reply("知道了")],
            Broken(COMPILE, [""], root=self.root),
        )

        answer = agent.run("改一下")

        self.assertEqual(answer, "知道了")
        self.assertEqual(agent.repair.rounds, [])
        self.assertIsNone(agent.repair.stopped)
        self.assertFalse(agent.repair.pending)
        self.assertIn("没跑成", self.tool_messages(agent)[-1])

    def test_an_agent_without_verification_tools_is_left_alone(self):
        # 只装了 edit_file：验不了，谈不上自动修
        client = StubClient([self.edit(), final_reply("好")])
        agent = Agent(
            client=client,
            tools=ToolRegistry([EditFileTool(root=self.root)]),
        )

        self.assertEqual(agent.run("改一下"), "好")
        self.assertEqual(agent.repair.rounds, [])
        self.assertEqual(len(client.seen), 2)


# ---------------------------------------------------------------------------
# Trace：整条修复链都看得见
# ---------------------------------------------------------------------------


class TraceTest(RepairTestBase):
    def test_the_whole_repair_shows_up_in_the_trace(self):
        compile_tool = self.fake(
            COMPILE,
            [
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                ran("g++ main.cpp", 0),
            ],
        )
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                self.edit(old_text="    return 0;", new_text="    return 0; // ok"),
                final_reply("好了"),
            ],
            compile_tool,
        )

        agent.run("编译不过，改一下")

        events = agent.get_trace()
        # 模型自己编译 -> 改 -> 自动验证（没过）-> 再改 -> 自动验证（过了）
        self.assertEqual(
            [event.tool for event in events], [COMPILE, EDIT, COMPILE, EDIT, COMPILE]
        )
        self.assertTrue(all(event.success for event in events))
        self.assertEqual(events[2].arguments, {"source": "main.cpp"})


# ---------------------------------------------------------------------------
# RepairLoop 本身：参数、算轮的小工具
# ---------------------------------------------------------------------------


class RepairLoopUnitTest(unittest.TestCase):
    def verifier(self):
        registry = ToolRegistry([])
        return Verifier(registry, ToolExecutor(registry))

    def test_the_default_cap_is_five_rounds(self):
        self.assertEqual(DEFAULT_MAX_REPAIR_ROUNDS, 5)
        self.assertEqual(RepairLoop(self.verifier()).max_rounds, 5)

    def test_a_cap_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            RepairLoop(self.verifier(), max_rounds=0)

    def test_the_agent_refuses_a_cap_below_one(self):
        with self.assertRaises(ValueError):
            Agent(client=StubClient([]), max_repair_rounds=0)

    def test_the_repair_loop_uses_the_agents_verifier(self):
        agent = Agent(client=StubClient([]))
        self.assertIs(agent.repair.verifier, agent.verifier)


class HelperTest(RepairTestBase):
    def test_the_edit_summary_is_the_position_line(self):
        self.assertEqual(_edit_summary(RESULT), "第 2 行（原文替换，命中 1 处）")

    def test_an_unreadable_file_has_no_digest(self):
        self.assertEqual(_digest(self.root, "没有这个文件.cpp"), "")

    def test_the_fingerprint_is_the_place_plus_what_went_in(self):
        first = _fingerprint("main.cpp", RESULT, {"new_text": "    int answer = 42;"})
        # 同一个地方、同样的内容：换一种写法（old_text 带的上下文不一样）还是同一次
        again = _fingerprint(
            "main.cpp", RESULT, {"new_text": "    int answer = 42;", "old_text": "…"}
        )
        self.assertEqual(first, again)

        # 内容不一样、位置不一样、文件不一样：都不是同一次修改
        self.assertNotEqual(
            first, _fingerprint("main.cpp", RESULT, {"new_text": "    int answer = 43;"})
        )
        self.assertNotEqual(
            first,
            _fingerprint(
                "main.cpp",
                RESULT.replace("第 2 行", "第 9 行"),
                {"new_text": "    int answer = 42;"},
            ),
        )
        self.assertNotEqual(
            first, _fingerprint("other.cpp", RESULT, {"new_text": "    int answer = 42;"})
        )

    def test_the_fingerprint_does_not_mind_which_newline_style(self):
        # 模型抄回来的原文是 \n，文件里可能是 \r\n：同一次改动还是同一个指纹
        lf = _fingerprint("main.cpp", RESULT, {"new_text": "a\nb"})
        crlf = _fingerprint("main.cpp", RESULT, {"new_text": "a\r\nb"})
        self.assertEqual(lf, crlf)

    def test_a_file_that_did_not_change_has_the_same_digest(self):
        first = _digest(self.root, "main.cpp")
        self.write("main.cpp", BAD)  # 同样的内容再写一遍
        self.assertEqual(_digest(self.root, "main.cpp"), first)
        self.assertNotEqual(
            _digest(self.root, "main.cpp"), _digest(self.root, "没有这个文件.cpp")
        )


# ---------------------------------------------------------------------------
# 真的编译两轮
# ---------------------------------------------------------------------------


class RealRepairTest(RepairTestBase):
    """真编译器、真 edit_file：第一次没修完 -> 自动验证没过 -> 再修 -> 退出码 0。

    两个错各自独立：先报的（未声明的变量）在后一个错被修掉之后必须还在，
    这样第一轮才有「没修完」可言。错误顺序是故意的——少分号的那个写在后面。
    """

    BAD = (
        "#include <iostream>\n"
        "\n"
        "int square(int value) {\n"
        "    return value * value;\n"
        "}\n"
        "\n"
        "int main() {\n"
        "    int total = square(7);\n"
        "    std::cout << totl << std::endl;\n"  # totl 没声明
        "    int unused = 1\n"  # 少了分号
        "    return 0;\n"
        "}\n"
    )

    def setUp(self):
        super().setUp()
        if find_compiler() is None:
            raise unittest.SkipTest("这台机器上没有 C++ 编译器")
        self.write("main.cpp", self.BAD)

    def test_a_real_two_error_file_is_fixed_over_two_rounds(self):
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                # 第一处：补上分号（还有一个错没修）
                self.edit(old_text="    int unused = 1\n", new_text="    int unused = 1;\n"),
                # 第二处：把拼错的变量名改对
                self.edit(old_text="std::cout << totl", new_text="std::cout << total"),
                final_reply("两处都修好了"),
            ],
            CompileTool(root=self.root),
        )

        answer = agent.run("main.cpp 编译不过，改一下")

        self.assertTrue(answer.endswith("两处都修好了"))
        self.assertIn("修复 2 轮；第 1 轮编译没过；第 2 轮编译通过；最终成功。", answer)

        # 第一轮：真编译没过，而且看得出「有的消失了、有的还在」
        first_round = agent.repair.rounds[0]
        self.assertFalse(first_round.passed)
        self.assertNotEqual(first_round.exit_code, 0)
        self.assertTrue(first_round.diff.gone)
        self.assertTrue(first_round.diff.still)
        self.assertIn("没通过", self.tool_messages(agent)[1])

        # 第二轮：退出码 0，修复结束
        second_round = agent.repair.rounds[1]
        self.assertTrue(second_round.passed)
        self.assertEqual(second_round.exit_code, 0)
        self.assertIn("通过（退出码 0）", self.tool_messages(agent)[2])

        # 真的没问题了：文件里那两个错都不在了
        fixed = self.read()
        self.assertIn("int unused = 1;", fixed)
        self.assertIn("std::cout << total", fixed)
        self.assertNotIn("totl", fixed)


class RealTwoStageTest(RepairTestBase):
    """真编译器 + 真 run_tests：编译过了、测试没过 -> 接着修 -> 两关都过。

    项目里埋了两个错，正好一关一个：

    - 少个分号：编译就过不了，第一关拦下；
    - `a - b`（该是 `a + b`）：**编得过**，但算出来的数是错的——只有第二关
      （跑起来比输出）才看得见。

    所以第一轮改完分号，编译这关过了，测试一定红；模型得照测试输出再改一轮。
    """

    BAD = (
        "#include <iostream>\n"
        "\n"
        "int add(int a, int b) {\n"
        "    return a - b\n"  # 少分号；就算补上，a - b 也是错的
        "}\n"
        "\n"
        "int main() {\n"
        "    std::cout << add(2, 3) << std::endl;\n"
        "    return 0;\n"
        "}\n"
    )

    TEST = (
        "import subprocess\n"
        "import unittest\n"
        "from pathlib import Path\n"
        "\n"
        "ROOT = Path(__file__).resolve().parent.parent\n"
        "\n"
        "\n"
        "class ProgramTest(unittest.TestCase):\n"
        "    def test_it_prints_five(self):\n"
        "        exe = next(\n"
        '            (p for p in (ROOT / "main.exe", ROOT / "main") if p.is_file()),\n'
        '            None,\n'
        "        )\n"
        '        self.assertIsNotNone(exe, "还没编出可执行文件")\n'
        "        done = subprocess.run(\n"
        "            [str(exe)], capture_output=True, text=True, timeout=30\n"
        "        )\n"
        '        self.assertEqual(done.stdout.strip(), "5")\n'
    )

    def setUp(self):
        super().setUp()
        if find_compiler() is None:
            raise unittest.SkipTest("这台机器上没有 C++ 编译器")
        self.write("main.cpp", self.BAD)
        # run_tests 跑的是 discover -s tests -t .：tests 得是个包
        self.write("tests/__init__.py", "")
        self.write("tests/test_program.py", self.TEST)

    def test_a_logic_bug_caught_by_the_tests_is_fixed_in_a_second_round(self):
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                # 第一处：补上分号。编得过，但 add 还是算错的
                self.edit(old_text="    return a - b\n", new_text="    return a - b;\n"),
                # 第二处：照测试输出把逻辑改对
                self.edit(old_text="    return a - b;", new_text="    return a + b;"),
                final_reply("改好了，测试也过了"),
            ],
            CompileTool(root=self.root),
            TestTool(root=self.root),
        )

        answer = agent.run("main.cpp 有个错，帮我修好")

        # 第一轮：编译过了，测试没过——7.5 那会儿到这儿就算「修好了」
        first = agent.repair.rounds[0]
        self.assertTrue(first.compiled)
        self.assertEqual(first.exit_code, 0)
        self.assertFalse(first.test_passed)
        self.assertFalse(first.passed)
        self.assertNotEqual(first.test_exit_code, 0)
        self.assertEqual(first.state, "测试没过")
        self.assertIn("AssertionError", self.tool_messages(agent)[1])

        # 第二轮：两关都过了
        second = agent.repair.rounds[1]
        self.assertTrue(second.compiled)
        self.assertTrue(second.test_passed)
        self.assertEqual(second.test_exit_code, 0)
        self.assertTrue(second.passed)

        self.assertIn("第 1 轮测试没过；第 2 轮编译通过、测试通过；最终成功。", answer)
        self.assertTrue(answer.endswith("改好了，测试也过了"))

        # 真的改对了：文件里是加号，跑起来真的打印 5
        self.assertIn("return a + b;", self.read())
        self.assertNotIn("a - b", self.read())

        # 每一步都是真跑出来的，trace 里一条不少
        self.assertEqual(
            [event.tool for event in agent.get_trace()],
            [COMPILE, EDIT, COMPILE, RUN_TESTS, EDIT, COMPILE, RUN_TESTS],
        )


if __name__ == "__main__":
    unittest.main()
