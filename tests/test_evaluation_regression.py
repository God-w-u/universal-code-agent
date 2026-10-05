"""效果回归（13.6）：把自带的 43 条 RAG case 当成一把固定的尺子。

和功能测试的分工，一句话：**功能测试问「对不对」，这个文件问「有没有变差」。**

- 功能测试（test_evaluation.py / test_evaluation_groups.py / test_evaluation_entry.py）
  钉的是尺子本身准不准：名次怎么算、指标怎么汇总、case 怎么读、分组怎么切、
  入口怎么选组。那儿用的是可控的小项目，断言的是精确的数字。
- **效果回归（这个文件）** 拿真语料跑真 case 集（43 条：项目 36 条 + C++ fixture
  7 条），盯的是「这一版和上一版比，检回能力有没有塌」。数字是**快照 + 下限**：
  快了不报警，掉了超过余量才红——语料里加一个文件、改一句注释本来就会动名次，
  把当期数字写死只会天天假红（test_evaluation.py 的模块头也是这么说的）。

四组：

1. `FrozenCaseSetTest`：case 集是**冻结**的——总数、正反例、语言 / 查询语言 /
   查询类型 / 语料的分布都数一遍。以后要加 case、改 case，这边的数字得跟着
   有意识地更新一次，而不是不知不觉漂走。
2. `BaselineRegressionTest`：字面 baseline 的 Hit@1 / Hit@3 / Hit@5，整体、
   python / cpp、中文 / 英文、词法 / 语义各一组下限；反例必须一条都查不到。
3. `FakeEmbeddingRegressionTest`：假 embedding 的 hybrid 那一组——**离线、常跑**，
   验的是「整条语义管线接上之后，字面结果不被打下去、跑得稳」。它不背语义
   质量（假货的向量本来就是假的）。
4. `RealModelRegressionTest`：local / multilingual 两组真模型——默认**跳过**
   （要装库、要下模型、跑一次几十秒），设 `RAG_REAL_MODELS=1` 才跑；模型
   不可用时跳到 skip，不判失败。

不联网、不装东西、不打真模型（除非显式开）。**不改任何 case、阈值和评分逻辑**：
这里的下限是从当前实测值往下留了余量写死的，尺子本身一个字没动。
"""

import os
import unittest

from index import (
    DEFAULT_CORPUS,
    DEMO_ALIASES,
    EmbeddingScorer,
    Evaluator,
    FakeEmbedder,
    Retriever,
    create_embedder,
    hybrid_scorers,
    load_cases,
)
from index.__main__ import baseline_retrievers, build_corpora

# 真模型那两组的开关：默认关着——全量测试跑得快、离线、不依赖模型文件。
REAL_MODELS = os.environ.get("RAG_REAL_MODELS") == "1"

# 冻结快照（2026-10-05，字面 baseline，43 条 case / 37 条正例，不带缓存）：
#
#   整体     hit1 = 10/37   hit3 = 12/37   hit5 = 12/37
#   python   hit5 =  9/30   cpp  hit5 =  3/7
#   中文     hit5 =  2/20   en   hit5 = 10/17
#   词法     hit5 = 10/12   语义 hit5 =  2/25
#   反例     6 条，假阳性 0
#
# 语义那 25 条字面打分够不着是数据集**故意**的（那正是 embedding 要补的位置），
# 所以它的下限只是「别整片塌掉」。每条下限都比实测低 1~3 条：语料自然漂移
# 不翻红，整组掉下去才报警。
OVERALL_HIT1_FLOOR = 8 / 37
OVERALL_HIT3_FLOOR = 10 / 37
OVERALL_HIT5_FLOOR = 10 / 37
PYTHON_HIT5_FLOOR = 7 / 30
CPP_HIT5_FLOOR = 2 / 7
ZH_HIT5_FLOOR = 1 / 20
EN_HIT5_FLOOR = 8 / 17
LEXICAL_HIT5_FLOOR = 0.75  # 和 test_evaluation.py 的那条报警线同一个数
SEMANTIC_HIT5_FLOOR = 1 / 25

# 假 embedding 那一组允许的反例假阳性上限。实测是 1：FakeEmbedder 的向量是
# 词表哈希凑出来的，跟文本意思没关系，个别反例蹭到分是它的正常表现——不是
# 回归，但**再多就是整条管线的行为变了**，得回头看。
FAKE_FALSE_POSITIVE_LIMIT = 1

_CACHE: dict[str, object] = {}


def cases():
    """自带的那 43 条 case（整个模块共用一份）。"""
    if "cases" not in _CACHE:
        _CACHE["cases"] = load_cases()
    return _CACHE["cases"]


def corpora():
    """真项目 + C++ fixture 两份语料（建一次索引要一两秒，全模块共用）。"""
    if "corpora" not in _CACHE:
        _CACHE["corpora"] = build_corpora()
    return _CACHE["corpora"]


def baseline_report():
    """字面 baseline 跑一遍（共用一份，省得每类重跑）。"""
    if "baseline" not in _CACHE:
        _CACHE["baseline"] = Evaluator(baseline_retrievers(corpora())).run(cases())
    return _CACHE["baseline"]


def fake_retrievers() -> dict[str, Retriever]:
    """每个语料配一份挂假 embedding 的 retriever（不预热）。"""
    return {
        name: Retriever(
            index=corpus,
            scorers=hybrid_scorers(FakeEmbedder(DEMO_ALIASES), chunks=corpus.chunks),
        )
        for name, corpus in corpora().items()
    }


def fake_report():
    """假 embedding 的 hybrid 跑一遍（离线）。跑过的那批 scorer 留着看统计。"""
    if "fake" not in _CACHE:
        retrievers = fake_retrievers()
        scorers = [
            scorer
            for retriever in retrievers.values()
            for scorer in retriever.scorers
            if isinstance(scorer, EmbeddingScorer)
        ]
        for scorer in scorers:
            scorer.prefetch()
        _CACHE["fake_scorers"] = {
            name: scorer
            for name, retriever in retrievers.items()
            for scorer in retriever.scorers
            if isinstance(scorer, EmbeddingScorer)
        }
        _CACHE["fake"] = Evaluator(retrievers).run(cases())
    return _CACHE["fake"]


def fake_scorers() -> dict[str, EmbeddingScorer]:
    """`fake_report()` 那次真跑过的语义 scorer（语料名 → scorer）。"""
    return _CACHE["fake_scorers"]


# ---------------------------------------------------------------------------
# 1. case 集是冻结的
# ---------------------------------------------------------------------------


class FrozenCaseSetTest(unittest.TestCase):
    """43 条 case 的形状：加一条、删一条、改一处标注都会在这儿翻红。"""

    def setUp(self):
        self.cases = cases()
        self.positive = [item for item in self.cases if not item.expected_nothing]
        self.negative = [item for item in self.cases if item.expected_nothing]

    def test_the_set_has_the_frozen_size(self):
        self.assertEqual(len(self.cases), 43)
        self.assertEqual(len(self.positive), 37)
        self.assertEqual(len(self.negative), 6)

    def test_no_query_appears_twice(self):
        self.assertEqual(len({item.query for item in self.cases}), len(self.cases))

    def test_the_language_split_is_frozen(self):
        # 37 条正例全标了语言；反例不标（反例没有「该命中哪种语言的代码」）
        counts = _counts(self.positive, "language")
        self.assertEqual(counts, {"python": 30, "cpp": 7})
        self.assertEqual(_counts(self.negative, "language"), {"": 6})

    def test_the_query_language_split_is_frozen(self):
        self.assertEqual(_counts(self.positive, "query_lang"), {"zh": 20, "en": 17})
        self.assertEqual(_counts(self.negative, "query_lang"), {"zh": 4, "en": 2})

    def test_the_query_type_split_is_frozen(self):
        counts = _counts(self.positive, "query_type")
        self.assertEqual(counts, {"lexical": 12, "semantic": 25})
        self.assertEqual(_counts(self.negative, "query_type"), {"": 6})

    def test_the_corpus_split_is_frozen(self):
        self.assertEqual(_counts(self.positive, "corpus"), {"": 30, "cpp_fixture": 7})
        # 反例没指语料：本来就是「哪儿都查不到」的查询，走默认语料
        self.assertTrue(all(item.corpus is None for item in self.negative))


def _counts(items, field: str) -> dict[str, int]:
    """按某个元数据字段数一遍（没标的算空串）。"""
    counted: dict[str, int] = {}
    for item in items:
        key = getattr(item, field) or ""
        counted[key] = counted.get(key, 0) + 1
    return counted


# ---------------------------------------------------------------------------
# 2. 字面 baseline：Hit@1 / Hit@3 / Hit@5 的回归线
# ---------------------------------------------------------------------------


class BaselineRegressionTest(unittest.TestCase):
    """字面 baseline 不许掉出下限；反例不许出现假阳性。"""

    @classmethod
    def setUpClass(cls):
        cls.report = baseline_report()

    def test_every_case_has_a_result(self):
        self.assertEqual(len(self.report.results), len(cases()))

    def test_the_rates_stay_above_the_line(self):
        self.assertGreaterEqual(self.report.hit1, OVERALL_HIT1_FLOOR)
        self.assertGreaterEqual(self.report.hit3, OVERALL_HIT3_FLOOR)
        self.assertGreaterEqual(self.report.hit5, OVERALL_HIT5_FLOOR)

    def test_the_rates_keep_their_shape(self):
        self.assertLessEqual(self.report.hit1, self.report.hit3)
        self.assertLessEqual(self.report.hit3, self.report.hit5)
        for rate in (self.report.hit1, self.report.hit3, self.report.hit5):
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)

    def test_the_lexical_group_still_holds(self):
        """字面打分的主场：词法查询得照旧进前 5。"""
        lexical = self.report.group_by("query_type")["lexical"]
        self.assertGreaterEqual(lexical.hit5, LEXICAL_HIT5_FLOOR)

    def test_the_semantic_group_does_not_collapse(self):
        semantic = self.report.group_by("query_type")["semantic"]
        self.assertGreaterEqual(semantic.hit5, SEMANTIC_HIT5_FLOOR)

    def test_each_language_group_stays_above_its_line(self):
        groups = self.report.group_by("language")
        self.assertGreaterEqual(groups["python"].hit5, PYTHON_HIT5_FLOOR)
        self.assertGreaterEqual(groups["cpp"].hit5, CPP_HIT5_FLOOR)

    def test_each_query_language_group_stays_above_its_line(self):
        groups = self.report.group_by("query_lang")
        self.assertGreaterEqual(groups["zh"].hit5, ZH_HIT5_FLOOR)
        self.assertGreaterEqual(groups["en"].hit5, EN_HIT5_FLOOR)

    def test_every_negative_case_stays_empty(self):
        self.assertEqual(len(self.report.nothing_cases), 6)
        for item in self.report.nothing_cases:
            self.assertTrue(
                item.found_nothing,
                f"{item.query!r} 本该查不到，却查到了 {len(item.results)} 条",
            )

    def test_a_second_run_gives_the_same_verdicts(self):
        again = Evaluator(baseline_retrievers(corpora())).run(cases())

        self.assertEqual(
            [item.verdict for item in self.report.results],
            [item.verdict for item in again.results],
        )
        self.assertEqual(self.report.hit5, again.hit5)


# ---------------------------------------------------------------------------
# 3. 假 embedding：离线常跑的那条语义管线
# ---------------------------------------------------------------------------


class FakeEmbeddingRegressionTest(unittest.TestCase):
    """整条语义管线（预热 / 打分 / 缓存）接上之后：不许比字面差，跑得要稳。

    它不背语义质量——FakeEmbedder 的向量是词表哈希凑的，跟意思无关。它背的是
    「接上之后不出事」：字面的结果没被挤下去、反例没有成片翻车、两次跑一样。
    """

    @classmethod
    def setUpClass(cls):
        cls.report = fake_report()

    def test_every_case_has_a_result(self):
        self.assertEqual(len(self.report.results), len(cases()))

    def test_it_does_not_lose_ground_to_the_baseline(self):
        self.assertGreaterEqual(self.report.hit1, baseline_report().hit1)
        self.assertGreaterEqual(self.report.hit5, baseline_report().hit5)

    def test_the_pipeline_really_ran(self):
        """不是「接了个没干活的 scorer」：整份语料一次批量嵌完，查询另算。

        每个语料一只 scorer：预热把这份语料的**每个块**一次批量嵌掉（一次
        调用，不是逐条），之后每条 case 的查询各一次。
        """
        per_corpus = fake_scorers()
        self.assertTrue(per_corpus)
        table = corpora()
        for name, scorer in per_corpus.items():
            chunks = len(table[name].chunks())
            served = sum(
                1
                for item in cases()
                if (item.corpus or DEFAULT_CORPUS) == name
            )
            self.assertEqual(scorer.texts_embedded, chunks + served)
            self.assertEqual(scorer.embed_calls, 1 + served)  # 1 批块 + 每条查询

    def test_negative_cases_do_not_fall_over_in_droves(self):
        wrong = [item for item in self.report.nothing_cases if not item.found_nothing]
        self.assertLessEqual(
            len(wrong),
            FAKE_FALSE_POSITIVE_LIMIT,
            f"反例假阳性 {len(wrong)} 条，超过上限 {FAKE_FALSE_POSITIVE_LIMIT}",
        )

    def test_a_second_run_gives_the_same_verdicts(self):
        again = Evaluator(fake_retrievers()).run(cases())

        self.assertEqual(
            [item.verdict for item in self.report.results],
            [item.verdict for item in again.results],
        )


# ---------------------------------------------------------------------------
# 4. 真模型：默认跳过，显式开才跑
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    REAL_MODELS,
    "真模型评估是可选实验（要装库、要下模型）：设 RAG_REAL_MODELS=1 才跑",
)
class RealModelRegressionTest(unittest.TestCase):
    """local / multilingual 两组真模型：跑得通、指标在合理范围。

    具体数字是实验数据（模型版本、机器都会影响），不在这儿写死；模型装不上 /
    下不动一律 skip，不判失败——这条线是「能跑就量一把」，不是全量测试的
    前置条件。
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = cases()
        cls.table = corpora()

    def report_for(self, mode: str):
        try:
            embedder = create_embedder(mode)
            retrievers = {
                name: Retriever(
                    index=corpus,
                    scorers=hybrid_scorers(embedder, chunks=corpus.chunks),
                )
                for name, corpus in self.table.items()
            }
            for retriever in retrievers.values():
                for scorer in retriever.scorers:
                    if isinstance(scorer, EmbeddingScorer):
                        scorer.prefetch()  # 模型就是在这儿加载的
        except Exception as exc:  # 模型不可用 = 跳过，不是失败
            self.skipTest(f"{mode} 模型不可用：{type(exc).__name__}: {exc}")
        return Evaluator(retrievers).run(self.cases)

    def check(self, mode: str) -> None:
        report = self.report_for(mode)

        self.assertEqual(len(report.results), len(self.cases))
        self.assertLessEqual(report.hit1, report.hit3)
        self.assertLessEqual(report.hit3, report.hit5)
        for rate in (report.hit1, report.hit3, report.hit5):
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)
        self.assertEqual(len(report.nothing_cases), 6)

    def test_the_local_model_still_runs(self):
        self.check("local")

    def test_the_multilingual_model_still_runs(self):
        self.check("multilingual")


if __name__ == "__main__":
    unittest.main()
