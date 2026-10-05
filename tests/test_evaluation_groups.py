"""9.5 评估集扩充：元数据、分组统计、多语料、自带的 case 文件。

四块：

- **元数据**（`EvaluationCase` 的几个新字段）：能读、取值写错会报错、
  老 case 文件（没这些字段）照常能读——loader 向后兼容。
- **分组统计**（`EvaluationReport.group_by / tally / grouped_summary`）：
  按语言 / 查询语言 / 查询类型 / 语料切，反例单列；没标元数据又没有反例时
  整段不印（老输出逐字不变）。
- **多语料**（`Evaluator`）：case 的 `corpus` 指到哪份索引就走哪份；指了
  不存在的语料直接报错，不当成「没命中」。
- **评估入口**：`--no-segment` 的形状；分组结果与 `[segment]` 诊断行的打印。

全部离线：小工程是临时目录里现写的，索引现建，不联网、不碰真模型。
这个文件里**不写自带的 case 查询原句**（项目把自己也索引了）。
"""

import json
import tempfile
import unittest
from pathlib import Path

from index import (
    DEFAULT_CORPUS,
    SEGMENT_STRATEGY,
    ChunkIndex,
    Embedder,
    EmbeddingScorer,
    EmbeddingUnavailable,
    EvaluationCase,
    EvaluationReport,
    EvaluationResult,
    Evaluator,
    FakeEmbedder,
    Retriever,
    load_cases,
    namespace_of,
)
from index.__main__ import (
    CPP_FIXTURE_CORPUS,
    CPP_FIXTURE_ROOT,
    build_corpora,
    overflow_line,
    run_reports,
    variants,
)
from tools.base import ToolError

STORE = '''\
"""存东西。"""


def zebra_sighting():
    """记一笔斑马目击。"""
    return 1
'''

OTHER = '''\
"""另一份语料。"""


def kite_flying():
    """放风筝。"""
    return 2
'''


def write_project(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8", newline="")
    return path


class ReportTestBase(unittest.TestCase):
    def result(
        self,
        query: str,
        rank: int | None,
        *,
        hit_time: float = 0.01,
        **case_fields,
    ) -> EvaluationResult:
        """手搓一条结果：分组统计只看 case 上的元数据和 rank。"""
        case = EvaluationCase(query=query, expected_symbol="target", **case_fields)
        return EvaluationResult(case, results=(), rank=rank, elapsed_time=hit_time)

    def report(self, *results: EvaluationResult) -> EvaluationReport:
        return EvaluationReport(tuple(results))


# ---------------------------------------------------------------------------
# 1. 元数据与 loader
# ---------------------------------------------------------------------------


class MetadataTest(ReportTestBase):
    def test_the_metadata_fields_default_to_empty(self):
        case = EvaluationCase(query="zebra", expected_symbol="zebra")

        self.assertEqual(case.language, "")
        self.assertEqual(case.query_lang, "")
        self.assertEqual(case.query_type, "")
        self.assertIsNone(case.corpus)

    def test_a_misspelled_value_is_rejected(self):
        with self.assertRaises(ToolError):
            EvaluationCase(
                query="zebra", expected_symbol="zebra", query_type="semantik"
            )
        with self.assertRaises(ToolError):
            EvaluationCase(query="zebra", expected_symbol="zebra", language="py")

    def test_an_empty_corpus_name_is_rejected(self):
        with self.assertRaises(ToolError):
            EvaluationCase(query="zebra", expected_symbol="zebra", corpus="  ")


class LoaderTest(unittest.TestCase):
    def write_cases(self, payload) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name).resolve() / "cases.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_the_new_fields_load(self):
        path = self.write_cases(
            [
                {
                    "query": "zebra",
                    "expected_symbol": "zebra_sighting",
                    "language": "python",
                    "query_lang": "en",
                    "query_type": "semantic",
                    "corpus": "other",
                }
            ]
        )

        case = load_cases(path)[0]

        self.assertEqual(case.language, "python")
        self.assertEqual(case.query_lang, "en")
        self.assertEqual(case.query_type, "semantic")
        self.assertEqual(case.corpus, "other")

    def test_an_old_case_file_still_loads(self):
        # 9.4 及以前的形状：只有 query + expected_*
        path = self.write_cases(
            [
                {"query": "zebra", "expected_symbol": "zebra_sighting"},
                {"query": "nowhere", "expected_nothing": True},
            ]
        )

        cases = load_cases(path)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].language, "")

    def test_a_bad_value_is_reported_with_the_path(self):
        path = self.write_cases(
            [{"query": "zebra", "expected_symbol": "s", "query_lang": "zh-CN"}]
        )

        with self.assertRaises(ToolError) as caught:
            load_cases(path)

        self.assertIn("query_lang", str(caught.exception))

    def test_an_unknown_field_is_still_rejected(self):
        path = self.write_cases(
            [{"query": "zebra", "expected_symbol": "s", "difficulty": "hard"}]
        )

        with self.assertRaises(ToolError):
            load_cases(path)


class ShippedCasesTest(unittest.TestCase):
    """自带的 case 文件：规模、覆盖和元数据都得立得住。"""

    def setUp(self):
        self.cases = load_cases()

    def test_the_set_is_big_enough_to_measure_something(self):
        positives = [case for case in self.cases if not case.expected_nothing]
        negatives = [case for case in self.cases if case.expected_nothing]

        self.assertGreaterEqual(len(positives), 20)
        self.assertGreaterEqual(len(negatives), 5)

    def test_every_positive_carries_its_metadata(self):
        for case in self.cases:
            if case.expected_nothing:
                continue
            self.assertTrue(case.language, case.query)
            self.assertTrue(case.query_lang, case.query)
            self.assertTrue(case.query_type, case.query)

    def test_both_languages_and_both_query_types_are_covered(self):
        positives = [case for case in self.cases if not case.expected_nothing]

        self.assertEqual({case.language for case in positives}, {"python", "cpp"})
        self.assertEqual({case.query_lang for case in positives}, {"zh", "en"})
        self.assertEqual(
            {case.query_type for case in positives}, {"lexical", "semantic"}
        )

    def test_the_cpp_cases_point_at_the_fixture_corpus(self):
        cpp = [case for case in self.cases if case.language == "cpp"]

        self.assertTrue(cpp)
        self.assertEqual({case.corpus for case in cpp}, {CPP_FIXTURE_CORPUS})
        self.assertTrue(CPP_FIXTURE_ROOT.is_dir())

    def test_no_test_file_quotes_a_semantic_query(self):
        """测试文件里不能写 case 的查询原句（8.4 立下的规矩）。

        项目把自己也索引了：查询原句写进 `tests/**.py`，那条 case 就会命中
        测试文件自己——命中率好看，其实是在考自己。词法查询（符号名、文件名
        这些）本来就该在源码里出现，不查它们；JSON case 文件不产 chunk，
        写在那儿没关系。
        """
        root = Path(__file__).resolve().parent
        sources = sorted(root.rglob("*.py"))

        for case in self.cases:
            if case.query_type != "semantic":
                continue
            for path in sources:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if case.query in text:
                    self.fail(f"case 查询 {case.query!r} 出现在 {path.name} 里")


# ---------------------------------------------------------------------------
# 2. 分组统计
# ---------------------------------------------------------------------------


class GroupByTest(ReportTestBase):
    def sample(self) -> EvaluationReport:
        return self.report(
            self.result("a", 1, language="python", query_lang="zh", query_type="semantic"),
            self.result("b", 4, language="python", query_lang="zh", query_type="lexical"),
            self.result("c", None, language="cpp", query_lang="en", query_type="semantic"),
            EvaluationResult(
                EvaluationCase(query="d", expected_nothing=True),
                results=(),
                rank=None,
                elapsed_time=0.01,
            ),
        )

    def test_it_slices_by_language(self):
        groups = self.sample().group_by("language")

        self.assertEqual(set(groups), {"python", "cpp"})
        self.assertEqual(len(groups["python"].answerable), 2)
        self.assertEqual(groups["python"].hit1, 0.5)
        self.assertEqual(groups["cpp"].hit1, 0.0)

    def test_negatives_never_land_in_a_group(self):
        groups = self.sample().group_by("query_type")

        counted = sum(len(report.results) for report in groups.values())
        self.assertEqual(counted, 3)  # 反例不在里面

    def test_unlabelled_cases_are_left_out(self):
        report = self.report(self.result("a", 1, language="python"), self.result("b", 1))

        groups = report.group_by("language")

        self.assertEqual(list(groups), ["python"])
        self.assertEqual(groups["python"].answerable[0].query, "a")

    def test_the_overall_hit_rates_are_untouched(self):
        report = self.sample()

        self.assertEqual(report.hit1, 1 / 3)
        self.assertEqual(report.hit3, 1 / 3)
        self.assertEqual(report.hit5, 2 / 3)

    def test_the_tally_reuses_the_same_rates(self):
        report = self.sample()

        line = report.tally()

        self.assertIn("该找到 3 条", line)
        self.assertIn("Hit@1 1/3（33.3%）", line)
        self.assertIn("Hit@5 2/3（66.7%）", line)
        self.assertIn("反例 1 条、1 条确实没查到", line)

    def test_the_grouped_summary_lists_every_group(self):
        text = self.sample().grouped_summary()

        self.assertIn("整体：", text)
        self.assertIn("语言=python：", text)
        self.assertIn("语言=cpp：", text)
        self.assertIn("查询语言=zh：", text)
        self.assertIn("查询类型=semantic：", text)
        self.assertIn("negative：反例 1 条、1 条确实没查到、假阳性 0 条", text)

    def test_an_old_style_report_prints_nothing(self):
        report = self.report(self.result("a", 1), self.result("b", 2))

        self.assertEqual(report.grouped_summary(), "")

    def test_a_false_positive_is_counted(self):
        report = self.report(
            self.result("a", 1, language="python"),
            EvaluationResult(
                EvaluationCase(query="b", expected_nothing=True),
                results=(object(),),
                rank=None,
                elapsed_time=0.01,
            ),
        )

        self.assertIn("假阳性 1 条", report.grouped_summary())


# ---------------------------------------------------------------------------
# 3. 多语料
# ---------------------------------------------------------------------------


class CorpusRoutingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        first = self.root / "one"
        second = self.root / "two"
        first.mkdir()
        second.mkdir()
        write_project(first, "store.py", STORE)
        write_project(second, "other.py", OTHER)
        self.one = ChunkIndex(root=first)
        self.two = ChunkIndex(root=second)
        self.two.build()

    def evaluator(self) -> Evaluator:
        return Evaluator(
            {
                DEFAULT_CORPUS: Retriever(index=self.one),
                "other": Retriever(index=self.two),
            }
        )

    def test_a_case_without_a_corpus_uses_the_default(self):
        report = self.evaluator().run(
            (EvaluationCase(query="zebra", expected_symbol="zebra_sighting"),)
        )

        self.assertEqual(report.hit1, 1.0)

    def test_a_case_can_point_at_another_corpus(self):
        report = self.evaluator().run(
            (
                EvaluationCase(
                    query="kite", expected_symbol="kite_flying", corpus="other"
                ),
            )
        )

        self.assertEqual(report.hit1, 1.0)

    def test_an_unknown_corpus_is_an_error_not_a_miss(self):
        with self.assertRaises(ToolError) as caught:
            self.evaluator().run(
                (EvaluationCase(query="kite", expected_symbol="k", corpus="nope"),)
            )

        self.assertIn("nope", str(caught.exception))

    def test_a_bare_retriever_is_still_accepted(self):
        report = Evaluator(Retriever(index=self.one)).run(
            (EvaluationCase(query="zebra", expected_symbol="zebra_sighting"),)
        )

        self.assertEqual(report.hit1, 1.0)

    def test_an_empty_corpus_table_is_rejected(self):
        with self.assertRaises(ToolError):
            Evaluator({})


class BuildCorporaTest(unittest.TestCase):
    def test_it_builds_the_project_and_the_fixture(self):
        table = build_corpora()

        self.assertIn(DEFAULT_CORPUS, table)
        self.assertIn(CPP_FIXTURE_CORPUS, table)
        self.assertTrue(table[DEFAULT_CORPUS].built)
        symbols = {chunk.symbol for chunk in table[CPP_FIXTURE_CORPUS].chunks()}
        self.assertIn("RingBuffer.push", symbols)

    def test_the_project_corpus_has_no_fixture_chunks(self):
        table = build_corpora()

        paths = {chunk.path for chunk in table[DEFAULT_CORPUS].chunks()}
        self.assertFalse([path for path in paths if "fixtures" in path])


# ---------------------------------------------------------------------------
# 4. 评估入口：开关与诊断行
# ---------------------------------------------------------------------------


class WindowEmbedder(Embedder):
    """声明窗口 + 真词器的假 embedder：给入口的形状测试用。"""

    def __init__(self, window: int = 8, dimension: int = 2) -> None:
        self._window = window
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> int | None:
        return self._window

    def count_tokens(self, text: str) -> int:
        return len(text)  # 只求「有个能数的东西」，不模拟真分词

    def embed(self, texts):
        return tuple((float(len(text)), 1.0) for text in texts)


class EntryTestBase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        write_project(self.root, "store.py", STORE)
        self.index = ChunkIndex(root=self.root)
        self.index.build()
        self.cases = (
            EvaluationCase(
                query="zebra",
                expected_symbol="zebra_sighting",
                language="python",
                query_lang="en",
                query_type="lexical",
            ),
        )


class NoSegmentSwitchTest(EntryTestBase):
    def test_the_switch_only_changes_the_hybrid_groups(self):
        plain = variants(self.index, ["--hybrid"])[1]
        cut = variants(self.index, ["--hybrid", "--no-segment"])[1]

        self.assertNotIn("不分段", plain[0])
        self.assertIn("不分段", cut[0])

    def test_the_segmenting_switch_reaches_the_scorer(self):
        embedder = WindowEmbedder(window=64)

        cut = variants(self.index, ["--hybrid"], local_embedder=embedder)[1]
        off = variants(
            self.index, ["--hybrid", "--no-segment"], local_embedder=embedder
        )[1]

        # 用假 embedder 那一组测：它没声明窗口，两个方向都是不分段
        self.assertFalse(cut[1][DEFAULT_CORPUS].scorers[-1].segmented)
        self.assertFalse(off[1][DEFAULT_CORPUS].scorers[-1].segmented)

    def test_a_windowed_embedder_segments_unless_told_not_to(self):
        embedder = WindowEmbedder(window=64)

        cut = variants(self.index, ["--local"], local_embedder=embedder)[2]
        off = variants(
            self.index, ["--local", "--no-segment"], local_embedder=embedder
        )[2]

        self.assertTrue(cut[1][DEFAULT_CORPUS].scorers[-1].segmented)
        self.assertFalse(off[1][DEFAULT_CORPUS].scorers[-1].segmented)


class CacheNamespaceEntryTest(EntryTestBase):
    def setUp(self):
        super().setUp()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.cache_dir = Path(temp.name).resolve()

    def namespace(self, argv) -> str:
        embedder = WindowEmbedder(window=64)
        groups = variants(
            self.index, argv, local_embedder=embedder, cache_dir=self.cache_dir
        )
        return groups[2][1][DEFAULT_CORPUS].scorers[-1].cache.namespace

    def test_the_two_switches_use_different_files(self):
        cut = self.namespace(["--local"])
        off = self.namespace(["--local", "--no-segment"])

        self.assertIn(SEGMENT_STRATEGY, cut)
        self.assertNotIn(SEGMENT_STRATEGY, off)
        self.assertIn(namespace_of(WindowEmbedder(window=64)), cut)


class OverflowLineTest(EntryTestBase):
    def corpus(self) -> dict[str, ChunkIndex]:
        return {DEFAULT_CORPUS: self.index}

    def test_it_reports_the_segmented_side(self):
        scorer = EmbeddingScorer(WindowEmbedder(window=8), chunks=self.index.chunks)

        line = overflow_line(scorer, self.corpus())

        self.assertIn("分段后仍有", line)
        self.assertIn("窗口 8 token", line)

    def test_it_reports_the_whole_text_side_when_not_segmenting(self):
        scorer = EmbeddingScorer(
            WindowEmbedder(window=8), chunks=self.index.chunks, segment=False
        )

        line = overflow_line(scorer, self.corpus())

        self.assertIn("未分段仍有", line)

    def test_it_is_skipped_without_a_window(self):
        class NoWindow(WindowEmbedder):
            @property
            def max_tokens(self) -> int | None:
                return None

        scorer = EmbeddingScorer(NoWindow(window=8), chunks=self.index.chunks)

        self.assertIsNone(overflow_line(scorer, self.corpus()))

    def test_it_is_skipped_without_a_tokenizer(self):
        scorer = EmbeddingScorer(FakeEmbedder(), chunks=self.index.chunks)

        self.assertIsNone(overflow_line(scorer, self.corpus()))

    def test_a_broken_counter_only_costs_the_line(self):
        class Broken(WindowEmbedder):
            def count_tokens(self, text: str) -> int:
                raise EmbeddingUnavailable("模型没加载")

        scorer = EmbeddingScorer(Broken(window=8), chunks=self.index.chunks)

        self.assertIn("真实分词诊断跳过", overflow_line(scorer, self.corpus()))


class ReportsWithMetadataTest(EntryTestBase):
    def lines(self, argv, **kwargs) -> list[str]:
        lines: list[str] = []
        code = run_reports(self.cases, self.index, argv, out=lines.append, **kwargs)
        self.assertEqual(code, 0)
        return lines

    def test_the_grouped_lines_are_printed_for_labelled_cases(self):
        text = "\n".join(self.lines(["--hybrid"]))

        self.assertIn("分组结果：", text)
        self.assertIn("整体：", text)
        self.assertIn("语言=python：", text)
        self.assertIn("查询类型=lexical：", text)

    def test_the_segment_line_is_printed_for_a_windowed_embedder(self):
        lines = self.lines(["--local"], local_embedder=WindowEmbedder(window=8))
        text = "\n".join(lines)

        self.assertIn("[segment] 分段后仍有", text)

    def test_the_no_segment_run_reports_the_whole_text(self):
        lines = self.lines(
            ["--local", "--no-segment"], local_embedder=WindowEmbedder(window=8)
        )

        self.assertIn("[segment] 未分段仍有", "\n".join(lines))

    def test_the_fake_group_prints_no_segment_line(self):
        text = "\n".join(self.lines(["--hybrid"]))

        self.assertNotIn("[segment]", text)

    def test_the_segment_line_is_printed_once_for_two_corpora(self):
        """每个语料配一个 scorer，诊断行是同一份语料表数出来的——只印一次。"""
        other_root = self.root / "second_corpus"
        other_root.mkdir()
        write_project(other_root, "kite.py", OTHER)
        other = ChunkIndex(root=other_root)
        other.build()
        lines: list[str] = []

        code = run_reports(
            self.cases,
            {DEFAULT_CORPUS: self.index, "second": other},
            ["--local"],
            out=lines.append,
            local_embedder=WindowEmbedder(window=8),
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            len([line for line in lines if line.startswith("[segment]")]), 1
        )


if __name__ == "__main__":
    unittest.main()
