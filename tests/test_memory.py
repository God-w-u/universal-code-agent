"""Memory（Phase 11.1）测试：记录结构、JSONL 存取、轻量检索。

- **结构**：MemoryRecord 的字段、校验、序列化往返、摘要裁剪。
- **存取**：append / load、多条、重新加载后还在、目录不存在时自建、
  文件不存在当空历史、坏行跳过（一条坏不连累其余的）、半写残句不
  连累下一条、坏行留在原地。
- **检索**：token 命中（汉字 bigram、英文大小写不敏感）、打分排序
  （task 比 summary 重、同分新的在前、确定性）、top_k（默认 + 单次
  覆盖）、无匹配返回空、summary 上限（只裁命中结果，不动原文）。
- **边界**：默认文件在项目目录之外（运行数据不进仓库）；MemoryStore
  是抽象类。

全部在临时目录里跑，不联网、不碰默认路径上的真文件。
"""

import dataclasses
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from config import PROJECT_ROOT
from memory import (
    DEFAULT_FILENAME,
    DEFAULT_TOP_K,
    FAILURE,
    MAX_SUMMARY_CHARS,
    SUCCESS,
    LocalMemoryStore,
    MemoryRecord,
    MemoryRetriever,
    MemoryStore,
    default_memory_path,
    tokenize,
)
from memory.retriever import SCORE_SUMMARY, SCORE_TASK
from tools.base import ToolError


class MemoryTestBase(unittest.TestCase):
    """一个临时目录里的 store，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.path = self.root / "memory.jsonl"
        self.store = LocalMemoryStore(self.path)

    def record(self, task, summary="", outcome=SUCCESS, metadata=None):
        return MemoryRecord.create(
            task=task, summary=summary, outcome=outcome, metadata=metadata
        )

    def add(self, task, summary="", outcome=SUCCESS, metadata=None):
        """造一条并写进去，返回那条记录。"""
        record = self.record(task, summary, outcome, metadata)
        self.store.append(record)
        return record

    def line(self, task) -> str:
        """一条完整记录的 JSON 文本（造坏文件的素材）。"""
        return json.dumps(self.record(task).to_dict(), ensure_ascii=False)

    def raw_lines(self):
        return self.path.read_text(encoding="utf-8").splitlines()


class MemoryRecordTest(unittest.TestCase):
    """记录本身：字段、校验、序列化、裁剪。"""

    def test_the_six_fields_are_there(self):
        record = MemoryRecord.create(task="任务甲", summary="摘要", outcome=SUCCESS)

        self.assertEqual(
            [field.name for field in dataclasses.fields(record)],
            ["id", "timestamp", "task", "summary", "outcome", "metadata"],
        )

    def test_metadata_defaults_to_an_empty_dict(self):
        record = MemoryRecord.create(task="任务甲", summary="", outcome=FAILURE)

        self.assertEqual(record.metadata, {})

    def test_a_record_is_frozen(self):
        record = MemoryRecord.create(task="任务甲", summary="", outcome=SUCCESS)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            record.task = "改掉"

    def test_create_fills_a_fresh_id_and_timestamp(self):
        first = MemoryRecord.create(task="任务甲", summary="", outcome=SUCCESS)
        second = MemoryRecord.create(task="任务甲", summary="", outcome=SUCCESS)

        self.assertTrue(first.id.strip())
        self.assertNotEqual(first.id, second.id)
        datetime.fromisoformat(first.timestamp)  # 能解析，不是随手拼的串

    def test_an_explicit_id_and_timestamp_are_kept(self):
        record = MemoryRecord.create(
            task="任务甲",
            summary="",
            outcome=SUCCESS,
            id="fixed-1",
            timestamp="2026-01-01T00:00:00+00:00",
        )

        self.assertEqual(record.id, "fixed-1")
        self.assertEqual(record.timestamp, "2026-01-01T00:00:00+00:00")

    def test_outcome_only_accepts_the_two_values(self):
        with self.assertRaises(ValueError) as caught:
            MemoryRecord.create(task="任务甲", summary="", outcome="maybe")

        self.assertIn("maybe", str(caught.exception))

    def test_an_empty_task_is_rejected(self):
        with self.assertRaises(ValueError):
            MemoryRecord.create(task="   ", summary="", outcome=SUCCESS)

    def test_metadata_must_be_string_pairs(self):
        with self.assertRaises(ValueError):
            MemoryRecord.create(
                task="任务甲", summary="", outcome=SUCCESS, metadata={"轮次": 3}
            )

    def test_a_dict_round_trip_keeps_every_field(self):
        record = MemoryRecord.create(
            task="任务甲",
            summary="摘要",
            outcome=FAILURE,
            metadata={"模型": "假模型", "轮次": "2"},
        )

        self.assertEqual(MemoryRecord.from_dict(record.to_dict()), record)

    def test_from_dict_rejects_broken_shapes(self):
        for bad in ([], "task", 42, {"task": "只有 task"}, {}):
            with self.assertRaises(ValueError):
                MemoryRecord.from_dict(bad)

    def test_from_dict_turns_metadata_values_into_text(self):
        data = {
            "id": "1",
            "timestamp": "t",
            "task": "T",
            "summary": "",
            "outcome": SUCCESS,
            "metadata": {"轮次": 2},
        }

        self.assertEqual(MemoryRecord.from_dict(data).metadata, {"轮次": "2"})

    def test_clipped_shortens_the_summary_without_touching_the_original(self):
        record = MemoryRecord.create(
            task="任务甲", summary="一二三四五六七八九十", outcome=SUCCESS
        )

        short = record.clipped(5)

        self.assertEqual(short.summary, "一二三四…")
        self.assertEqual(len(short.summary), 5)
        self.assertEqual(record.summary, "一二三四五六七八九十")
        self.assertEqual(short.id, record.id)  # 还是同一条历史

    def test_clipped_leaves_a_short_summary_alone(self):
        record = MemoryRecord.create(task="任务甲", summary="短", outcome=SUCCESS)

        self.assertIs(record.clipped(50), record)

    def test_clipped_refuses_a_non_positive_limit(self):
        record = MemoryRecord.create(task="任务甲", summary="", outcome=SUCCESS)

        with self.assertRaises(ValueError):
            record.clipped(0)


class MemoryStoreShapeTest(MemoryTestBase):
    """store 的形状：抽象类、默认路径（项目之外）、懒建文件。"""

    def test_the_store_interface_is_abstract(self):
        with self.assertRaises(TypeError):
            MemoryStore()

    def test_the_default_path_is_outside_the_project(self):
        # 运行数据不进仓库：默认文件必须在项目目录之外
        path = default_memory_path()

        self.assertFalse(path.is_relative_to(PROJECT_ROOT))
        self.assertEqual(path.name, DEFAULT_FILENAME)

    def test_the_default_path_is_under_the_user_home(self):
        self.assertTrue(default_memory_path().is_relative_to(Path.home()))

    def test_a_store_without_a_path_uses_the_default(self):
        self.assertEqual(LocalMemoryStore().path, default_memory_path())

    def test_a_path_object_and_a_string_both_work(self):
        self.assertEqual(LocalMemoryStore(self.path).path, self.path)
        self.assertEqual(LocalMemoryStore(str(self.path)).path, self.path)

    def test_loading_an_empty_store_neither_writes_nor_fails(self):
        # 新建 store：文件不存在是正常状态，读出来是空历史，也不落盘
        self.assertEqual(self.store.load(), [])
        self.assertFalse(self.path.exists())

    def test_append_needs_a_record(self):
        with self.assertRaises(TypeError):
            self.store.append({"task": "假的"})


class MemoryStoreTest(MemoryTestBase):
    """存取：追加、多条、重载、落盘格式。"""

    def test_one_record_round_trips(self):
        record = self.add(task="任务甲", summary="摘要甲")

        self.assertEqual(self.store.load(), [record])

    def test_several_records_keep_the_write_order(self):
        first = self.add(task="任务甲")
        second = self.add(task="任务乙")
        third = self.add(task="任务丙")

        self.assertEqual(self.store.load(), [first, second, third])

    def test_history_survives_a_reload(self):
        self.add(task="任务甲", summary="摘要甲", outcome=FAILURE)

        again = LocalMemoryStore(self.path)  # 模拟下次运行 / 另一个实例

        loaded = again.load()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].task, "任务甲")
        self.assertEqual(loaded[0].outcome, FAILURE)

    def test_it_creates_the_directory_on_first_write(self):
        nested = self.root / "a" / "b" / "memory.jsonl"
        store = LocalMemoryStore(nested)

        store.append(self.record(task="任务甲"))

        self.assertTrue(nested.exists())

    def test_the_file_is_one_json_object_per_line(self):
        self.add(task="任务甲", summary="摘要甲")
        self.add(task="任务乙", summary="摘要乙")

        lines = self.raw_lines()

        self.assertEqual(len(lines), 2)
        self.assertEqual([json.loads(line)["task"] for line in lines],
                         ["任务甲", "任务乙"])
        self.assertTrue(self.path.read_text(encoding="utf-8").endswith("\n"))

    def test_chinese_is_written_readable_not_escaped(self):
        self.add(task="任务甲", summary="摘要甲")

        text = self.path.read_text(encoding="utf-8")

        self.assertIn("任务甲", text)


class MemoryStoreCorruptionTest(MemoryTestBase):
    """坏记录容错：一条坏不连累其余；半写残句不连累下一条。"""

    def test_a_broken_line_does_not_take_the_others_down(self):
        self.path.write_text(
            "\n".join(
                [self.line("任务甲"), "这不是 JSON", self.line("任务乙")]
            )
            + "\n",
            encoding="utf-8",
        )

        self.assertEqual(
            [record.task for record in self.store.load()], ["任务甲", "任务乙"]
        )

    def test_broken_lines_of_every_shape_are_skipped(self):
        broken = [
            "{不是合法 JSON",  # 解析不过去
            "[1, 2, 3]",  # 是 JSON，但不是对象
            '"一个字符串"',  # 同上
            '{"task": "缺字段"}',  # 字段不全
            '{"id": 1, "timestamp": "t", "task": "T", "summary": "", '
            '"outcome": "success"}',  # id 不是字符串
            '{"id": "1", "timestamp": "t", "task": "T", "summary": "", '
            '"outcome": "maybe"}',  # outcome 不认识
            '{"id": "1", "timestamp": "t", "task": "  ", "summary": "", '
            '"outcome": "success"}',  # 空 task
        ]
        self.path.write_text(
            "\n".join([self.line("任务甲")] + broken + [self.line("任务乙")])
            + "\n",
            encoding="utf-8",
        )

        self.assertEqual(
            [record.task for record in self.store.load()], ["任务甲", "任务乙"]
        )

    def test_blank_lines_are_not_records(self):
        self.path.write_text("\n\n" + self.line("任务甲") + "\n\n", encoding="utf-8")

        self.assertEqual([record.task for record in self.store.load()], ["任务甲"])

    def test_a_file_with_only_garbage_reads_as_empty(self):
        self.path.write_text("完全不是记录\n也不是\n", encoding="utf-8")

        self.assertEqual(self.store.load(), [])

    def test_invalid_bytes_do_not_crash_the_read(self):
        good = self.line("任务甲").encode("utf-8")
        self.path.write_bytes(b"\xff\xfe\x00 bad\n" + good + b"\n")

        self.assertEqual([record.task for record in self.store.load()], ["任务甲"])

    def test_a_broken_line_stays_where_it_is(self):
        # 读是只读的：坏行留在原地，不重写文件（重写才有丢数据的风险）
        content = self.line("任务甲") + "\n坏行\n"
        self.path.write_text(content, encoding="utf-8")

        self.store.load()

        self.assertEqual(self.path.read_text(encoding="utf-8"), content)

    def test_appending_after_a_broken_line_still_works(self):
        self.path.write_text("坏行\n", encoding="utf-8")

        self.add(task="任务乙")

        self.assertEqual([record.task for record in self.store.load()], ["任务乙"])

    def test_a_torn_tail_does_not_swallow_the_next_record(self):
        # 模拟上次写到一半：一行没有换行结尾的残句
        self.path.write_text("任务甲的半行", encoding="utf-8")

        self.add(task="任务乙")

        # 残句自己是一行（跳过），新记录是完整的一行（能读）
        self.assertEqual(len(self.raw_lines()), 2)
        self.assertEqual([record.task for record in self.store.load()], ["任务乙"])

    def test_a_missing_final_newline_still_reads_as_a_record(self):
        # 残句判定不能误伤：最后一行是完整记录、只是差个换行
        self.path.write_text(self.line("任务甲"), encoding="utf-8")

        self.assertEqual([record.task for record in self.store.load()], ["任务甲"])

        self.add(task="任务乙")

        self.assertEqual(
            [record.task for record in self.store.load()], ["任务甲", "任务乙"]
        )


class MemoryTokenizeTest(unittest.TestCase):
    """拆词规则：英文整词、汉字 bigram、去重、标点是分隔符。"""

    def test_ascii_runs_stay_whole_and_lowercase(self):
        self.assertEqual(tokenize("Fix Cache_Timeout"), ("fix", "cache_timeout"))

    def test_chinese_runs_become_bigrams(self):
        self.assertEqual(tokenize("修复缓存"), ("修复", "复缓", "缓存"))

    def test_tokens_are_deduped_in_order(self):
        self.assertEqual(tokenize("缓存 缓存 修复"), ("缓存", "修复"))

    def test_a_single_chinese_character_stays(self):
        self.assertEqual(tokenize("看"), ("看",))

    def test_punctuation_separates_chinese_runs(self):
        # 全角逗号是分隔符，不许把两个词粘成一个「存超」
        self.assertEqual(tokenize("缓存，超时！"), ("缓存", "超时"))

    def test_mixed_text_keeps_both_kinds_in_order(self):
        self.assertEqual(tokenize("缓存 cache"), ("缓存", "cache"))

    def test_an_empty_text_has_no_tokens(self):
        self.assertEqual(tokenize(""), ())
        self.assertEqual(tokenize("！？……"), ())


class MemoryRetrieverShapeTest(MemoryTestBase):
    """retriever 的形状：默认值、构造校验、store 留得住。"""

    def test_the_defaults_are_positive(self):
        self.assertGreater(DEFAULT_TOP_K, 0)
        self.assertGreater(MAX_SUMMARY_CHARS, 0)

    def test_the_store_is_kept(self):
        self.assertIs(MemoryRetriever(self.store).store, self.store)

    def test_a_non_positive_top_k_is_refused(self):
        with self.assertRaises(ValueError):
            MemoryRetriever(self.store, top_k=0)

    def test_a_non_positive_summary_limit_is_refused(self):
        with self.assertRaises(ValueError):
            MemoryRetriever(self.store, summary_chars=0)


class MemoryRetrievalTest(MemoryTestBase):
    """检索：命中、排序、top_k、无匹配、summary 上限。"""

    def setUp(self):
        super().setUp()
        self.retriever = MemoryRetriever(self.store)

    def tasks(self, hits):
        return [hit.record.task for hit in hits]

    def test_a_task_word_matches(self):
        self.add(task="修复缓存过期")

        self.assertEqual(self.tasks(self.retriever.retrieve("缓存")), ["修复缓存过期"])

    def test_a_summary_word_matches_too(self):
        self.add(task="任务甲", summary="顺带提到缓存")

        hits = self.retriever.retrieve("缓存")

        self.assertEqual(self.tasks(hits), ["任务甲"])
        self.assertEqual(hits[0].score, SCORE_SUMMARY)

    def test_matching_is_case_insensitive(self):
        self.add(task="Fix CacheTimeout")

        self.assertEqual(len(self.retriever.retrieve("cachetimeout")), 1)
        self.assertEqual(len(self.retriever.retrieve("CACHETIMEOUT")), 1)

    def test_records_without_a_hit_are_left_out(self):
        self.add(task="修复缓存过期")
        self.add(task="写文档")

        self.assertEqual(self.tasks(self.retriever.retrieve("缓存")), ["修复缓存过期"])

    def test_an_empty_memory_comes_back_empty(self):
        self.assertEqual(self.retriever.retrieve("缓存"), [])

    def test_an_empty_query_comes_back_empty(self):
        self.add(task="修复缓存过期")

        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   "), [])

    def test_a_query_with_nothing_in_common_comes_back_empty(self):
        self.add(task="修复缓存过期")

        self.assertEqual(self.retriever.retrieve("写一篇文档"), [])

    def test_a_task_hit_outranks_a_summary_hit(self):
        self.add(task="任务甲", summary="提到缓存")
        self.add(task="缓存相关任务")

        hits = self.retriever.retrieve("缓存")

        self.assertEqual(self.tasks(hits), ["缓存相关任务", "任务甲"])
        self.assertEqual(hits[0].score, SCORE_TASK)
        self.assertEqual(hits[1].score, SCORE_SUMMARY)

    def test_more_matching_words_rank_higher(self):
        self.add(task="缓存超时的问题")
        self.add(task="只有缓存")

        hits = self.retriever.retrieve("缓存 超时")

        self.assertEqual(hits[0].record.task, "缓存超时的问题")
        self.assertEqual(hits[0].score, 2 * SCORE_TASK)

    def test_a_word_in_both_task_and_summary_counts_twice(self):
        self.add(task="缓存的问题", summary="缓存还没修")

        self.assertEqual(
            self.retriever.retrieve("缓存")[0].score, SCORE_TASK + SCORE_SUMMARY
        )

    def test_a_tie_puts_the_newer_record_first(self):
        self.add(task="缓存甲")
        self.add(task="缓存乙")

        self.assertEqual(self.tasks(self.retriever.retrieve("缓存")), ["缓存乙", "缓存甲"])

    def test_the_order_is_deterministic(self):
        self.add(task="缓存甲")
        self.add(task="缓存乙")
        self.add(task="缓存丙")

        first = self.retriever.retrieve("缓存")
        again = self.retriever.retrieve("缓存")
        fresh = MemoryRetriever(LocalMemoryStore(self.path)).retrieve("缓存")

        expected = [hit.record.id for hit in first]
        self.assertEqual([hit.record.id for hit in again], expected)
        self.assertEqual([hit.record.id for hit in fresh], expected)

    def test_the_default_top_k_caps_the_results(self):
        for index in range(DEFAULT_TOP_K + 3):
            self.add(task=f"缓存第{index}条")

        self.assertEqual(len(self.retriever.retrieve("缓存")), DEFAULT_TOP_K)

    def test_top_k_can_be_overridden_per_call(self):
        for index in range(4):
            self.add(task=f"缓存第{index}条")

        self.assertEqual(len(self.retriever.retrieve("缓存", top_k=2)), 2)

    def test_a_bigger_top_k_does_not_pad_the_results(self):
        self.add(task="缓存甲")

        self.assertEqual(len(self.retriever.retrieve("缓存", top_k=50)), 1)

    def test_a_non_positive_top_k_is_refused_at_call_time(self):
        with self.assertRaises(ToolError):
            self.retriever.retrieve("缓存", top_k=0)

    def test_a_long_summary_is_clipped_in_the_hit_only(self):
        long_summary = "摘" * (MAX_SUMMARY_CHARS * 2)
        record = self.add(task="缓存", summary=long_summary)

        hit = self.retriever.retrieve("缓存")[0]

        self.assertLessEqual(len(hit.record.summary), MAX_SUMMARY_CHARS)
        self.assertTrue(hit.record.summary.endswith("…"))
        self.assertEqual(hit.record.id, record.id)  # 还是同一条历史
        # store 里的原文一个字没少（裁的是副本）
        self.assertEqual(self.store.load()[0].summary, long_summary)

    def test_the_summary_limit_can_be_configured(self):
        self.add(task="缓存", summary="一二三四五六七八九十")

        hit = MemoryRetriever(self.store, summary_chars=5).retrieve("缓存")[0]

        self.assertEqual(len(hit.record.summary), 5)

    def test_a_short_summary_is_left_alone(self):
        self.add(task="缓存", summary="短摘要")

        hit = self.retriever.retrieve("缓存")[0]

        self.assertEqual(hit.record.summary, "短摘要")

    def test_history_is_still_searchable_after_a_reload(self):
        self.add(task="修复缓存过期", summary="改了缓存 key")

        hits = MemoryRetriever(LocalMemoryStore(self.path)).retrieve("缓存")

        self.assertEqual(self.tasks(hits), ["修复缓存过期"])


if __name__ == "__main__":
    unittest.main()
