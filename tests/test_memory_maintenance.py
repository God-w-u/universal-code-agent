"""Memory 生命周期与存储管理（Phase 11.3）测试：上限、去重、清理、统计。

- **上限**：条数超了自动丢最旧的，文件大小超了丢到装得下（至少留最新
  一条）；没超就不动文件。显式的 prune / compact 各管一摊：prune 按上限
  丢，compact 只把文件重写干净（坏行、残句、重复都消失）。
- **去重**：task + outcome + summary 完全一样只留最新；task 相同但结果
  不同的两条都留着。
- **安全**：重写走临时文件 + 换名；换名失败时原文件一个字节没变、临时
  文件清掉；自动清理失败不影响 Agent 这一轮。
- **统计**：total / success / failure / oldest / newest，坏行不算。
- **兼容**：旧文件（CRLF、坏行、残句）照读；默认构造的 store 还是
  「懒建、一行一条」；带着上限跑 Agent，一轮任务还是一条记录。
- **配置**：两个上限可配、默认值和 memory 那边一致、拼错报错。

全部在临时目录里跑，不联网、不碰默认路径上的真文件。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from agent import Agent
from config import (
    ConfigError,
    MEMORY_DEFAULT_MAX_BYTES,
    MEMORY_DEFAULT_MAX_RECORDS,
    MemoryConfig,
    load_memory_config,
)
from memory import (
    DEFAULT_LIMITS,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_RECORDS,
    FAILURE,
    SUCCESS,
    LocalMemoryStore,
    MemoryContext,
    MemoryLimits,
    MemoryRecord,
    MemoryRetriever,
)
from memory.maintenance import dedupe, plan_prune, record_size, stats_of
from tests.test_agent import FakeLLMClient


class MemoryMaintenanceTestBase(unittest.TestCase):
    """一个临时目录里的文件，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.path = self.root / "memory.jsonl"
        # 重写时用的临时文件名（和 store 里的规则一致，见 _rewrite）
        self.scratch = self.path.with_name(self.path.name + ".tmp")

    def plain(self) -> LocalMemoryStore:
        """默认上限的 store（500 条 / 2 MiB）。"""
        return LocalMemoryStore(self.path)

    def raw(self) -> LocalMemoryStore:
        """只加一行的 store（limits=None）。

        造「脏」文件用：重复、超限的历史得先绕过自动清理才写得进去。
        """
        return LocalMemoryStore(self.path, limits=None)

    def limited(
        self,
        max_records: int = DEFAULT_MAX_RECORDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> LocalMemoryStore:
        return LocalMemoryStore(
            self.path,
            limits=MemoryLimits(max_records=max_records, max_bytes=max_bytes),
        )

    def record(self, task, summary="", outcome=SUCCESS, timestamp=None):
        return MemoryRecord.create(
            task=task, summary=summary, outcome=outcome, timestamp=timestamp
        )

    def fill(self, store, count, task=lambda index: f"缓存任务{index:02d}"):
        """往 store 里连写 count 条等长的记录，返回写进去的那批。"""
        records = [self.record(task(index)) for index in range(count)]
        for record in records:
            store.append(record)
        return records

    def line(self, task) -> str:
        """一条完整记录的 JSON 文本（造坏文件 / 手写文件的素材）。"""
        return json.dumps(self.record(task).to_dict(), ensure_ascii=False)


class MemoryLimitsTest(unittest.TestCase):
    """上限本身：默认值是正的，0 / 负数不接受。"""

    def test_the_defaults_are_positive(self):
        self.assertGreater(DEFAULT_MAX_RECORDS, 0)
        self.assertGreater(DEFAULT_MAX_BYTES, 0)
        self.assertEqual(
            DEFAULT_LIMITS,
            MemoryLimits(
                max_records=DEFAULT_MAX_RECORDS, max_bytes=DEFAULT_MAX_BYTES
            ),
        )

    def test_a_non_positive_record_limit_is_refused(self):
        with self.assertRaises(ValueError):
            MemoryLimits(max_records=0)
        with self.assertRaises(ValueError):
            MemoryLimits(max_records=-1)

    def test_a_non_positive_byte_limit_is_refused(self):
        with self.assertRaises(ValueError):
            MemoryLimits(max_bytes=0)


class RecordSizeTest(MemoryMaintenanceTestBase):
    """大小上限的账要和真实文件对得上（换行、编码）。"""

    def test_the_size_matches_what_append_writes(self):
        record = self.record("缓存任务")

        self.raw().append(record)

        self.assertEqual(self.path.stat().st_size, record_size(record))

    def test_files_are_written_with_plain_newlines(self):
        # 写的时候 newline=""：Windows 上也不会变成 \r\n，大小账才准
        self.raw().append(self.record("任务甲"))
        self.raw().append(self.record("任务乙"))

        self.assertNotIn(b"\r\n", self.path.read_bytes())


class DedupeTest(unittest.TestCase):
    """去重：三元组完全一样只留最新；只差 summary / outcome 的不算重复。"""

    def record(self, task, summary="", outcome=SUCCESS, timestamp="t"):
        return MemoryRecord.create(
            task=task, summary=summary, outcome=outcome, timestamp=timestamp
        )

    def test_an_exact_duplicate_keeps_only_the_newest(self):
        first = self.record("任务甲", "同摘要", timestamp="2026-01-01T00:00:00+00:00")
        second = self.record("任务甲", "同摘要", timestamp="2026-01-02T00:00:00+00:00")

        kept = dedupe([first, second])

        self.assertEqual([record.id for record in kept], [second.id])

    def test_the_other_records_keep_their_order(self):
        first = self.record("任务甲")
        middle = self.record("任务乙")
        again = self.record("任务甲")

        kept = dedupe([first, middle, again])

        self.assertEqual([record.task for record in kept], ["任务乙", "任务甲"])
        self.assertEqual(kept[-1].id, again.id)  # 留下的还是最新那条

    def test_a_same_task_with_another_summary_is_not_a_duplicate(self):
        first = self.record("任务甲", "第一次的结果")
        second = self.record("任务甲", "第二次的结果")

        kept = dedupe([first, second])

        self.assertEqual(len(kept), 2)

    def test_a_same_task_with_another_outcome_is_not_a_duplicate(self):
        done = self.record("任务甲", "同样的摘要", SUCCESS)
        failed = self.record("任务甲", "同样的摘要", FAILURE)

        kept = dedupe([done, failed])

        self.assertEqual([record.outcome for record in kept], [SUCCESS, FAILURE])

    def test_nothing_to_dedupe_returns_everything(self):
        records = [self.record("任务甲"), self.record("任务乙")]

        self.assertEqual(dedupe(records), records)

    def test_an_empty_list_stays_empty(self):
        self.assertEqual(dedupe([]), [])


class PlanPruneTest(unittest.TestCase):
    """算账逻辑（不动文件）：去重在前，然后条数、再大小。"""

    def records(self, count):
        return [
            MemoryRecord.create(
                task=f"缓存任务{index:02d}", summary="", outcome=SUCCESS
            )
            for index in range(count)
        ]

    def test_no_limits_means_no_dropping(self):
        records = self.records(6)

        result = plan_prune(records, None)

        self.assertEqual(result.after, 6)
        self.assertEqual(result.over_count, 0)
        self.assertEqual(result.over_bytes, 0)

    def test_the_record_limit_keeps_the_newest(self):
        records = self.records(5)

        result = plan_prune(
            records, MemoryLimits(max_records=2, max_bytes=10_000_000)
        )

        self.assertEqual(result.before, 5)
        self.assertEqual(result.after, 2)
        self.assertEqual(result.over_count, 3)
        self.assertEqual(
            [record.task for record in result.kept], ["缓存任务03", "缓存任务04"]
        )

    def test_the_byte_limit_counts_the_serialized_size(self):
        records = self.records(2)
        newest_size = record_size(records[1])

        result = plan_prune(
            records, MemoryLimits(max_records=99, max_bytes=newest_size)
        )

        self.assertEqual([record.id for record in result.kept], [records[1].id])
        self.assertEqual(result.over_bytes, 1)

    def test_dedupe_happens_before_the_limits(self):
        first = MemoryRecord.create(task="任务甲", summary="", outcome=SUCCESS)
        again = MemoryRecord.create(task="任务甲", summary="", outcome=SUCCESS)
        other = MemoryRecord.create(task="任务乙", summary="", outcome=SUCCESS)

        result = plan_prune(
            [first, again, other], MemoryLimits(max_records=2, max_bytes=10_000_000)
        )

        self.assertEqual(result.deduped, 1)
        self.assertEqual(result.over_count, 0)  # 去重后就剩两条，没超
        self.assertEqual(result.after, 2)


class StatsTest(MemoryMaintenanceTestBase):
    """统计：总数 / 成 / 败 / 最早最晚（按写入顺序），坏行不算。"""

    def test_an_empty_history_counts_zero(self):
        stats = self.plain().stats()

        self.assertEqual((stats.total, stats.success, stats.failure), (0, 0, 0))
        self.assertIsNone(stats.oldest)
        self.assertIsNone(stats.newest)

    def test_outcomes_are_counted(self):
        store = self.raw()
        store.append(self.record("任务甲", outcome=SUCCESS))
        store.append(self.record("任务乙", outcome=FAILURE))
        store.append(self.record("任务丙", outcome=FAILURE))

        stats = store.stats()

        self.assertEqual((stats.total, stats.success, stats.failure), (3, 1, 2))

    def test_oldest_and_newest_follow_the_write_order(self):
        store = self.raw()
        first = self.record("任务甲", timestamp="2026-01-01T00:00:00+00:00")
        middle = self.record("任务乙", timestamp="2026-01-02T00:00:00+00:00")
        last = self.record("任务丙", timestamp="2026-01-03T00:00:00+00:00")
        for record in (first, middle, last):
            store.append(record)

        stats = store.stats()

        self.assertEqual(stats.oldest, first.timestamp)
        self.assertEqual(stats.newest, last.timestamp)

    def test_stats_count_only_readable_records(self):
        self.path.write_text(
            self.line("任务甲") + "\n坏行\n" + self.line("任务乙") + "\n",
            encoding="utf-8",
        )

        self.assertEqual(self.raw().stats().total, 2)

    def test_stats_of_an_empty_list_has_no_timestamps(self):
        stats = stats_of([])

        self.assertEqual(stats.total, 0)
        self.assertIsNone(stats.oldest)
        self.assertIsNone(stats.newest)


class AutoPruneOnAppendTest(MemoryMaintenanceTestBase):
    """append 之后自动看一眼上限：超了丢最旧的，没超什么都不动。"""

    def test_over_the_record_limit_keeps_the_newest(self):
        store = self.limited(max_records=3, max_bytes=10_000_000)

        self.fill(store, 5, task=lambda index: f"任务{index}")

        self.assertEqual(
            [record.task for record in store.load()],
            ["任务2", "任务3", "任务4"],
        )

    def test_over_the_byte_limit_drops_until_it_fits(self):
        self.fill(self.raw(), 5)
        line_size = self.path.stat().st_size // 5  # 每条一样长
        store = self.limited(max_records=99, max_bytes=2 * line_size)

        store.append(self.record("缓存任务05"))

        self.assertEqual(
            [record.task for record in store.load()],
            ["缓存任务04", "缓存任务05"],
        )
        self.assertLessEqual(self.path.stat().st_size, 2 * line_size)

    def test_one_record_bigger_than_the_limit_is_still_kept(self):
        # 单条就超上限时留它一条：上限是防增长，不是把历史抹干净
        store = self.limited(max_records=99, max_bytes=10)

        self.fill(store, 2, task=lambda index: f"任务{index}")

        self.assertEqual([record.task for record in store.load()], ["任务1"])
        self.assertGreater(self.path.stat().st_size, 10)

    def test_nothing_over_the_limit_leaves_the_file_alone(self):
        store = self.limited(max_records=10, max_bytes=10_000_000)

        with patch.object(LocalMemoryStore, "_rewrite") as rewrite:
            store.append(self.record("任务甲"))
            store.append(self.record("任务乙"))

        self.assertEqual(rewrite.call_count, 0)  # 没超限就一次都没重写
        self.assertIsNone(store.last_error)

    def test_a_duplicate_is_dropped_on_append(self):
        store = self.limited(max_records=10, max_bytes=10_000_000)
        first = self.record("同一件事", "同样的结果")
        second = self.record("同一件事", "同样的结果")  # id / 时间戳不同，内容一样

        store.append(first)
        store.append(second)

        records = store.load()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].id, second.id)  # 留的是最新那条

    def test_a_same_task_with_another_result_survives(self):
        store = self.limited(max_records=10, max_bytes=10_000_000)

        store.append(self.record("同一件事", "第一次的结果"))
        store.append(self.record("同一件事", "第二次的结果"))

        self.assertEqual(len(store.load()), 2)

    def test_the_limit_check_does_not_read_when_limits_are_off(self):
        # limits=None：append 退回成「只加一行」，连加载都不做
        store = self.raw()

        with patch.object(LocalMemoryStore, "load", side_effect=AssertionError):
            store.append(self.record("任务甲"))

        self.assertEqual(len(LocalMemoryStore(self.path).load()), 1)


class PruneTest(MemoryMaintenanceTestBase):
    """显式 prune：按上限丢最旧的；没变化就不动文件。"""

    def test_prune_drops_the_oldest_over_the_record_limit(self):
        self.fill(self.raw(), 4, task=lambda index: f"任务{index}")
        store = self.limited(max_records=2, max_bytes=10_000_000)

        result = store.prune()

        self.assertEqual(result.before, 4)
        self.assertEqual(result.after, 2)
        self.assertEqual(result.over_count, 2)
        self.assertTrue(result.rewritten)
        self.assertEqual(
            [record.task for record in store.load()], ["任务2", "任务3"]
        )

    def test_prune_does_nothing_when_within_the_limits(self):
        store = self.limited(max_records=5, max_bytes=10_000_000)
        store.append(self.record("任务甲"))
        before = self.path.read_bytes()

        result = store.prune()

        self.assertFalse(result.rewritten)
        self.assertFalse(result.deduped)
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_history_is_still_readable_after_a_prune(self):
        self.fill(self.raw(), 4, task=lambda index: f"缓存任务{index}")
        store = self.limited(max_records=2, max_bytes=10_000_000)

        store.prune()

        records = store.load()
        self.assertEqual([record.task for record in records], ["缓存任务2", "缓存任务3"])
        hits = MemoryRetriever(store).retrieve("缓存")
        self.assertEqual(hits[0].record.task, "缓存任务3")  # 同分新的在前

    def test_prune_reports_the_dedupe_count(self):
        store = self.raw()
        store.append(self.record("任务甲", "同摘要"))
        store.append(self.record("任务甲", "同摘要"))
        store.append(self.record("任务乙"))

        result = store.prune()

        self.assertEqual((result.before, result.after, result.deduped), (3, 2, 1))

    def test_prune_without_limits_only_dedupes(self):
        store = self.raw()
        for _ in range(6):
            store.append(self.record("同一件事", "同样的结果"))

        result = store.prune()

        self.assertEqual(result.after, 1)  # limits=None：不因为条数丢
        self.assertEqual(result.deduped, 5)

    def test_a_prune_that_rewrites_also_drops_bad_lines(self):
        self.path.write_text(
            "坏行一\n"
            + self.line("任务甲")
            + "\n坏行二\n"
            + self.line("任务乙")
            + "\n",
            encoding="utf-8",
        )
        store = self.limited(max_records=1, max_bytes=10_000_000)

        result = store.prune()

        self.assertTrue(result.rewritten)
        self.assertEqual(self.path.read_text(encoding="utf-8").count("\n"), 1)
        self.assertEqual([record.task for record in store.load()], ["任务乙"])


class CompactTest(MemoryMaintenanceTestBase):
    """显式 compact：把文件重写干净，不按上限丢历史。"""

    def test_compact_drops_bad_lines_and_keeps_the_good_records(self):
        # line() 每次生成新 id / 时间戳，所以先造好再拼文件、最后原样比对
        first = self.line("任务甲")
        second = self.line("任务乙")
        self.path.write_text(
            "坏行一\n" + first + "\n" + second + "\n坏行二\n", encoding="utf-8"
        )

        result = self.raw().compact()

        self.assertEqual((result.before, result.after), (2, 2))
        self.assertTrue(result.rewritten)
        self.assertEqual(
            self.path.read_text(encoding="utf-8").splitlines(),
            [first, second],
        )

    def test_compact_dedupes_and_keeps_the_newest(self):
        store = self.raw()
        store.append(self.record("任务甲", "同摘要"))
        store.append(self.record("任务甲", "同摘要"))
        store.append(self.record("任务乙"))

        result = store.compact()

        self.assertEqual((result.before, result.after, result.deduped), (3, 2, 1))
        self.assertEqual([record.task for record in store.load()], ["任务甲", "任务乙"])

    def test_compact_does_not_apply_the_limits(self):
        # compact 管「文件脏不脏」，不管「太多太大」——超限的历史不动
        self.fill(self.raw(), 4, task=lambda index: f"任务{index}")
        store = self.limited(max_records=1, max_bytes=10)

        result = store.compact()

        self.assertEqual(result.after, 4)
        self.assertEqual(result.over_count, 0)

    def test_compact_cleans_a_torn_tail(self):
        self.path.write_text(
            self.line("任务甲") + "\n任务乙的半行", encoding="utf-8"
        )

        self.raw().compact()

        self.assertEqual(
            [record.task for record in self.raw().load()], ["任务甲"]
        )
        self.assertTrue(self.path.read_text(encoding="utf-8").endswith("\n"))

    def test_compact_on_a_missing_file_does_not_create_it(self):
        result = self.plain().compact()

        self.assertFalse(self.path.exists())
        self.assertFalse(result.rewritten)
        self.assertEqual(result.after, 0)


class CorruptionCompatTest(MemoryMaintenanceTestBase):
    """旧文件 / 脏文件照读；新写的文件是干净的 LF。"""

    def test_an_old_file_with_crlf_line_endings_still_works(self):
        content = (self.line("任务甲") + "\r\n" + self.line("任务乙") + "\r\n")
        self.path.write_bytes(content.encode("utf-8"))
        store = self.raw()

        self.assertEqual(
            [record.task for record in store.load()], ["任务甲", "任务乙"]
        )

        store.append(self.record("任务丙"))

        self.assertEqual(len(store.load()), 3)

    def test_prune_keeps_working_on_a_file_with_bad_lines(self):
        self.path.write_text(
            "坏行\n" + self.line("任务甲") + "\n",
            encoding="utf-8",
        )
        store = self.limited(max_records=5, max_bytes=10_000_000)

        result = store.prune()

        self.assertFalse(result.rewritten)  # 没超限：坏行也留在原地
        self.assertEqual([record.task for record in store.load()], ["任务甲"])

    def test_a_broken_line_does_not_stop_the_automatic_cleanup(self):
        self.fill(self.raw(), 3, task=lambda index: f"任务{index}")
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text("坏行\n" + text, encoding="utf-8")
        store = self.limited(max_records=2, max_bytes=10_000_000)

        store.append(self.record("任务9"))

        self.assertEqual(
            [record.task for record in store.load()], ["任务2", "任务9"]
        )


class RewriteSafetyTest(MemoryMaintenanceTestBase):
    """重写走临时文件 + 换名：失败不破坏原文件，也不影响任务。"""

    def test_a_failed_prune_leaves_the_original_file_alone(self):
        self.fill(self.raw(), 3, task=lambda index: f"任务{index}")
        before = self.path.read_bytes()
        store = self.limited(max_records=2, max_bytes=10_000_000)

        with patch.object(Path, "replace", side_effect=OSError("换名失败（测试）")):
            with self.assertRaises(OSError):
                store.prune()

        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.scratch.exists())  # 临时文件清掉了
        self.assertEqual(len(store.load()), 3)

    def test_a_failed_compact_leaves_the_original_file_alone(self):
        store = self.raw()
        store.append(self.record("任务甲", "同摘要"))
        store.append(self.record("任务甲", "同摘要"))
        before = self.path.read_bytes()

        with patch.object(Path, "replace", side_effect=OSError("换名失败（测试）")):
            with self.assertRaises(OSError):
                store.compact()

        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.scratch.exists())

    def test_a_failed_automatic_cleanup_still_writes_the_record(self):
        store = self.limited(max_records=1, max_bytes=10_000_000)
        store.append(self.record("任务甲"))

        with patch.object(Path, "replace", side_effect=OSError("换名失败（测试）")):
            store.append(self.record("任务乙"))  # 触发自动清理，清理失败

        self.assertIn("自动清理", store.last_error)
        self.assertFalse(self.scratch.exists())
        # 记录已经写进去了（清理是附带的一步），而且读得出来
        self.assertEqual(
            [record.task for record in store.load()], ["任务甲", "任务乙"]
        )

    def test_a_failing_cleanup_does_not_change_the_answer(self):
        store = self.limited(max_records=1, max_bytes=10_000_000)
        store.append(self.record("旧任务"))
        agent = Agent(
            client=FakeLLMClient(["回答"]), memory=MemoryContext(store=store)
        )

        with patch.object(Path, "replace", side_effect=OSError("换名失败（测试）")):
            answer = agent.run("新任务")

        self.assertEqual(answer, "回答")
        self.assertEqual(len(store.load()), 2)  # 旧 + 新，清理没成但都在

    def test_a_successful_cleanup_clears_the_error(self):
        store = self.limited(max_records=1, max_bytes=10_000_000)
        store.append(self.record("任务甲"))
        with patch.object(Path, "replace", side_effect=OSError("换名失败（测试）")):
            store.append(self.record("任务乙"))
        self.assertIsNotNone(store.last_error)

        store.append(self.record("任务丙"))  # 这次换名是好的

        self.assertIsNone(store.last_error)


class MemoryLimitsConfigTest(unittest.TestCase):
    """配置里的两个上限：默认值、可配、拼错报错。"""

    def load(self, **env):
        cleared = {
            key: value
            for key, value in os.environ.items()
            if key
            not in (
                "AGENT_MEMORY",
                "AGENT_MEMORY_PATH",
                "AGENT_MEMORY_MAX_RECORDS",
                "AGENT_MEMORY_MAX_BYTES",
            )
        }
        cleared.update(env)
        with patch.dict(os.environ, cleared, clear=True):
            return load_memory_config()

    def test_the_defaults_come_from_the_memory_module(self):
        config = self.load()

        self.assertEqual(config.max_records, DEFAULT_MAX_RECORDS)
        self.assertEqual(config.max_bytes, DEFAULT_MAX_BYTES)

    def test_the_duplicated_defaults_stay_in_step(self):
        # config 和 memory 两处各写了一份默认值（有意的重复，见 config.py）：
        # 往一边改别忘了另一边
        self.assertEqual(MEMORY_DEFAULT_MAX_RECORDS, DEFAULT_MAX_RECORDS)
        self.assertEqual(MEMORY_DEFAULT_MAX_BYTES, DEFAULT_MAX_BYTES)

    def test_both_limits_can_be_configured(self):
        config = self.load(
            AGENT_MEMORY_MAX_RECORDS="7", AGENT_MEMORY_MAX_BYTES="2048"
        )

        self.assertEqual(config.max_records, 7)
        self.assertEqual(config.max_bytes, 2048)

    def test_a_non_integer_limit_is_refused(self):
        for value in ("五百", "500条", "1.5"):
            with self.assertRaises(ConfigError) as caught:
                self.load(AGENT_MEMORY_MAX_RECORDS=value)

            self.assertIn("AGENT_MEMORY_MAX_RECORDS", str(caught.exception))

    def test_a_non_positive_limit_is_refused(self):
        with self.assertRaises(ConfigError):
            self.load(AGENT_MEMORY_MAX_RECORDS="0")
        with self.assertRaises(ConfigError):
            self.load(AGENT_MEMORY_MAX_BYTES="-3")

    def test_an_empty_value_falls_back_to_the_default(self):
        config = self.load(AGENT_MEMORY_MAX_RECORDS="  ")

        self.assertEqual(config.max_records, DEFAULT_MAX_RECORDS)


class BuildMemoryLimitsTest(unittest.TestCase):
    """main.build_memory：把配置里的上限装进 store。"""

    def test_the_limits_reach_the_store(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name).resolve() / "memory.jsonl"

        memory = main.build_memory(
            MemoryConfig(enabled=True, path=path, max_records=7, max_bytes=999)
        )

        self.assertEqual(
            memory.store.limits, MemoryLimits(max_records=7, max_bytes=999)
        )


class AgentCompatTest(MemoryMaintenanceTestBase):
    """11.1 / 11.2 的行为不变：带上限的 store 上，一轮任务照旧一条记录。"""

    def test_a_run_still_writes_exactly_one_record(self):
        store = self.limited(max_records=5, max_bytes=10_000_000)
        agent = Agent(
            client=FakeLLMClient(["改好了"]), memory=MemoryContext(store=store)
        )

        agent.run("修一下缓存过期的问题")

        records = store.load()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].task, "修一下缓存过期的问题")
        self.assertEqual(records[0].summary, "改好了")
        self.assertEqual(records[0].outcome, SUCCESS)

    def test_a_default_store_keeps_the_default_limits(self):
        # 默认构造还是「懒建、一行一条」，同时也带着那两条默认上限
        store = LocalMemoryStore(self.path)

        self.assertEqual(store.limits, DEFAULT_LIMITS)
        self.assertEqual(store.load(), [])
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
