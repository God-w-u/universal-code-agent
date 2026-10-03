"""Repair History 测试：每次 run() 落一份 JSON。

这里测的是**记录本身**——文件写没写出来、写成什么样、写不出来会不会把任务
带崩。修复怎么跑是 test_repair 的事，所以这里直接用那套假替身把流程跑出来，
再看落盘的那份 JSON 对不对：完整修复、停在半路、没修、重复指纹停下，
各走一遍。

「写失败」这条最要紧：历史是附带的事，它失败绝不能把用户的任务变成失败。

全部离线，不联网。
"""

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent, RepairHistory
from agent.history import FIXED, NO_REPAIR, UNFIXED
from agent.verify import COMPILE, EDIT, RUN_TESTS
from tests.test_repair import RepairTestBase, StubClient, final_reply, tool_call_reply
from tests.test_verify import ran
from tools import EditFileTool, ToolRegistry


class HistoryTestBase(RepairTestBase):
    """在 RepairTestBase 上再挂一个临时历史目录。

    目录**故意不建**：建目录是 RepairHistory 自己的活，测的时候让它真干一次。
    """

    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.history_dir = Path(temp.name).resolve() / "repair_history"
        self.history = RepairHistory(self.history_dir)

    def build_agent(self, replies, *tools, **kwargs):
        kwargs.setdefault("history", self.history)
        return super().build_agent(replies, *tools, **kwargs)

    def records(self):
        """已经落盘的每一份记录，按文件名排（也就是按时间先后）。"""
        return [
            json.loads(path.read_text("utf-8"))
            for path in sorted(self.history_dir.glob("*.json"))
        ]


# ---------------------------------------------------------------------------
# 正常跑完的几次 run()，各记成了什么样
# ---------------------------------------------------------------------------


class RecordedRunTest(HistoryTestBase):
    def test_a_fixed_run_is_written_down_whole(self):
        # 两轮：第一轮编译没过，第二轮编译过了、测试也过了
        self.write("tests/__init__.py", "")
        agent = self.build_agent(
            [
                tool_call_reply(COMPILE, '{"source": "main.cpp"}'),
                self.edit(),
                self.edit(
                    old_text="    int answer = 42;",
                    new_text="    int answer = 42; // 又改了一次",
                ),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 0),
                ],
            ),
            self.fake(RUN_TESTS, [ran("python -m unittest", 0, out="OK")]),
        )

        answer = agent.run("编译不过，改一下")

        # 先确认这一轮本身没被记录这件事改样
        self.assertIn("修复 2 轮", answer)
        self.assertTrue(answer.endswith("好了"))

        # 目录是它自己建的，一次 run() 就是一份文件
        self.assertTrue(self.history_dir.is_dir())
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(len(self.history.paths), 1)
        self.assertIsNone(self.history.last_error)

        record = self.records()[0]
        self.assertRegex(record["time"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        self.assertEqual(record["request"], "编译不过，改一下")
        self.assertEqual(record["status"], FIXED)
        self.assertIsNone(record["stopped"])
        self.assertIn("修复 2 轮", record["summary"])
        self.assertEqual(record["round_count"], 2)
        self.assertEqual(len(record["rounds"]), 2)
        # 修好发生在第 2 轮——第 1 轮是没过的那个，不算
        self.assertEqual(record["fixed_round"], 2)

        first, second = record["rounds"]
        self.assertEqual(first["round"], 1)
        self.assertEqual(first["file"], "main.cpp")
        self.assertIn("第 2 行", first["edit"])
        self.assertEqual(len(first["fingerprint"]), 8)
        self.assertEqual(first["state"], "编译没过")
        self.assertEqual(first["compile"]["command"], "compile main.cpp")
        self.assertFalse(first["compile"]["passed"])
        self.assertEqual(first["compile"]["exit_code"], 2)
        self.assertIsNone(first["test"])  # 编都没过，测试那一关没跑
        # 诊断变化：老那条还在，没有消失、也没有新增
        self.assertEqual(first["diagnostics"]["gone"], [])
        self.assertEqual(
            first["diagnostics"]["still"],
            ["main.cpp:2:15: error: expected ';'"],
        )
        self.assertEqual(first["diagnostics"]["added"], [])

        self.assertEqual(second["round"], 2)
        self.assertEqual(second["state"], "编译通过、测试通过")
        self.assertTrue(second["compile"]["passed"])
        self.assertTrue(second["test"]["passed"])
        self.assertEqual(second["test"]["exit_code"], 0)
        self.assertIn("run_tests 通过", second["test"]["detail"])
        # 这一轮那条老诊断没了
        self.assertEqual(
            second["diagnostics"]["gone"],
            ["main.cpp:2:15: error: expected ';'"],
        )
        self.assertEqual(second["diagnostics"]["still"], [])
        self.assertEqual(second["diagnostics"]["added"], [])
        # 两次改动不一样，指纹自然不一样
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        # 目录里只有 JSON，没有写了半截的临时文件
        self.assertTrue(all(p.suffix == ".json" for p in self.history_dir.iterdir()))

    def test_a_stopped_repair_records_why_it_stopped(self):
        agent = self.build_agent(
            [self.edit()],
            self.fake(
                COMPILE,
                [ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'")],
            ),
            max_repair_rounds=1,
        )

        answer = agent.run("改一下")

        record = self.records()[0]
        self.assertEqual(record["status"], UNFIXED)
        self.assertIsNone(record["fixed_round"])  # 没修好，就没有「第几轮修好的」
        self.assertIn("上限 1 轮", record["stopped"])
        self.assertIn("最终没修好", record["summary"])
        self.assertEqual(record["round_count"], 1)
        self.assertEqual(len(record["rounds"]), 1)
        # 给用户的那段报告里就写着为什么停，记录里也是同一个原因
        self.assertIn(record["stopped"], answer)

    def test_only_the_round_that_fixed_it_counts(self):
        # 第 1 轮没过、第 2 轮没过、第 3 轮才过：记的是 3——不是 1，也不是 2。
        # 三轮的诊断/改动各不相同，免得先撞上「连着两轮一样」那条停止规矩
        agent = self.build_agent(
            [
                self.edit(old_text="    int answer = 42", new_text="    int answer = 42;"),
                self.edit(
                    old_text="    int answer = 42;",
                    new_text="    int answer = 42; // 第一次修",
                ),
                self.edit(
                    old_text="    int answer = 42; // 第一次修",
                    new_text="    int answer = 42; // 第二次修",
                ),
                final_reply("好了"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: 第一个错"),
                    ran("g++ main.cpp", 2, err="main.cpp:3:1: error: 第二个错"),
                    ran("g++ main.cpp", 0),
                ],
            ),
        )

        agent.run("改一下")

        record = self.records()[0]
        self.assertEqual(record["round_count"], 3)
        self.assertEqual(record["fixed_round"], 3)
        self.assertEqual(record["status"], FIXED)
        # 三个字段说的是同一件事，对得上
        self.assertEqual(record["fixed_round"], record["rounds"][-1]["round"])
        self.assertEqual(record["summary"], "修复 3 轮；第 1 轮编译没过；第 2 轮编译没过；第 3 轮编译通过；最终成功。")
        self.assertIn(f"修复 {record['round_count']} 轮", record["summary"])
        # 前两轮老老实实写着没过，没被当成修好的那一轮
        self.assertEqual(
            [item["state"] for item in record["rounds"]],
            ["编译没过", "编译没过", "编译通过"],
        )

    def test_a_repeated_fingerprint_stop_is_recorded(self):
        # A 改成 B、B 改回 A、A 又改成 B：第三次的指纹和第一次重合
        agent = self.build_agent(
            [
                self.edit(old_text="    int answer = 42", new_text="    int answer = 42;"),
                self.edit(old_text="    int answer = 42;", new_text="    int answer = 42"),
                self.edit(old_text="    int answer = 42", new_text="    int answer = 42;"),
            ],
            self.fake(
                COMPILE,
                [
                    ran("g++ main.cpp", 2, err="main.cpp:2:15: error: expected ';'"),
                    ran("g++ main.cpp", 2, err="main.cpp:3:1: error: 另一个错"),
                    ran("g++ main.cpp", 2, err="main.cpp:3:1: error: 另一个错"),
                ],
            ),
        )

        agent.run("改一下")

        record = self.records()[0]
        self.assertEqual(record["status"], UNFIXED)
        self.assertIsNone(record["fixed_round"])
        self.assertIn("检测到重复修改", record["stopped"])
        self.assertEqual(record["round_count"], 3)
        # 头一轮和第三轮的指纹一模一样，正是它触发的停止
        self.assertEqual(
            record["rounds"][2]["fingerprint"], record["rounds"][0]["fingerprint"]
        )

    def test_a_passing_edit_is_recorded_as_one_round(self):
        # 改完第一次验就过（模型没先编译）：7.8 起也算一轮，状态是「一轮通过」。
        # JSON 里的轮数和摘要里那句「修复 N 轮」说的是同一个数
        agent = self.build_agent(
            [self.edit(), final_reply("好了")],
            self.fake(COMPILE, [ran("g++ main.cpp", 0)]),
        )

        answer = agent.run("把这里改一下")

        record = self.records()[0]
        self.assertEqual(record["status"], FIXED)
        self.assertEqual(record["round_count"], 1)
        self.assertEqual(len(record["rounds"]), 1)
        self.assertEqual(record["fixed_round"], 1)  # 一轮就修好了
        self.assertEqual(record["summary"], "修复 1 轮；第 1 轮就通过；最终成功。")
        self.assertIn(record["summary"], answer)  # 最终回答里就是这一句
        self.assertTrue(
            record["summary"].startswith(f"修复 {record['round_count']} 轮")
        )

        round_ = record["rounds"][0]
        self.assertEqual(round_["state"], "一轮通过")
        self.assertEqual(round_["round"], 1)
        self.assertEqual(round_["file"], "main.cpp")
        self.assertIn("第 2 行", round_["edit"])
        self.assertEqual(len(round_["fingerprint"]), 8)
        self.assertTrue(round_["compile"]["passed"])
        self.assertEqual(round_["compile"]["exit_code"], 0)

    def test_a_plain_answer_is_recorded_too(self):
        # 一个工具都没有的一问一答，也留一笔：翻记录时看得到「这轮没干活」。
        # 没有工具时走的是普通 chat，客户端返回的就是一段文本（不是消息字典）
        agent = Agent(
            client=StubClient(["你好"]),
            tools=ToolRegistry([]),
            history=self.history,
        )

        self.assertEqual(agent.run("你好"), "你好")

        record = self.records()[0]
        self.assertEqual(record["request"], "你好")
        self.assertEqual(record["status"], NO_REPAIR)
        self.assertEqual(record["rounds"], [])
        self.assertIsNone(record["fixed_round"])  # 没改过代码，谈不上第几轮修好


# ---------------------------------------------------------------------------
# 一份一个文件：不覆盖、不留半个
# ---------------------------------------------------------------------------


class SeparateFilesTest(HistoryTestBase):
    def test_every_run_gets_its_own_file(self):
        agent = Agent(
            client=StubClient([final_reply("一"), final_reply("二")]),
            tools=ToolRegistry([EditFileTool(root=self.root)]),
            history=self.history,
        )

        self.assertEqual(agent.run("第一件事"), "一")
        self.assertEqual(agent.run("第二件事"), "二")

        files = sorted(self.history_dir.glob("*.json"))
        self.assertEqual(len(files), 2)
        self.assertNotEqual(files[0].name, files[1].name)
        for path in files:
            # 时间 + 随机数：人看得出先后，机器不会重名
            self.assertRegex(path.name, r"^\d{8}-\d{6}-[0-9a-f]{6}\.json$")
        # 第一份没被第二份顶掉：两份都在，各自记着自己那一轮。
        # 两次跑在同一秒里，文件名的时间那截一样，谁在前只看随机那截——所以比集合
        self.assertEqual(
            {json.loads(path.read_text("utf-8"))["request"] for path in files},
            {"第一件事", "第二件事"},
        )
        self.assertEqual(len(self.history.paths), 2)
        # 目录里干干净净：只有两份 JSON
        self.assertEqual(len(list(self.history_dir.iterdir())), 2)


# ---------------------------------------------------------------------------
# 写不进去：只留一句话，不许把任务带崩
# ---------------------------------------------------------------------------


class WriteFailureTest(HistoryTestBase):
    def test_a_write_failure_is_a_warning_not_a_failure(self):
        # 历史目录的位置上蹲着一个同名文件：目录建不出来
        self.history_dir.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.write_text("占着位置", encoding="utf-8")

        agent = self.build_agent(
            [self.edit(), final_reply("好了")],
            self.fake(COMPILE, [ran("g++ main.cpp", 0)]),
        )

        answer = agent.run("改一下")

        # 任务一点没受影响：回答照给（摘要照加），验证照跑、照附在工具结果后面
        self.assertTrue(answer.endswith("好了"))
        self.assertIn("验证（compile main.cpp）：通过", self.tool_messages(agent)[-1])
        self.assertIsNone(agent.repair.stopped)
        # 但这事得说出来，不能悄悄算了
        self.assertIsNone(self.history.last_path)
        self.assertEqual(self.history.paths, [])
        self.assertIsNotNone(self.history.last_error)
        self.assertIn(str(self.history_dir), self.history.last_error)

    def test_a_later_success_clears_the_old_warning(self):
        # 失败一次之后再写成了，last_error 要清掉（不然会一直报着上一次的错）
        self.history_dir.parent.mkdir(parents=True, exist_ok=True)
        self.history_dir.write_text("占着位置", encoding="utf-8")
        agent = self.build_agent(
            [self.edit(), final_reply("好了"), final_reply("又好了")],
            self.fake(COMPILE, [ran("g++ main.cpp", 0)]),
        )

        agent.run("改一下")
        self.assertIsNotNone(self.history.last_error)

        # 把占位置的那个文件挪开，再跑一次：这次该写成
        self.history_dir.unlink()
        agent.run("再改一下")

        self.assertIsNone(self.history.last_error)
        self.assertIsNotNone(self.history.last_path)
        self.assertEqual(len(self.history.paths), 1)


if __name__ == "__main__":
    unittest.main()
