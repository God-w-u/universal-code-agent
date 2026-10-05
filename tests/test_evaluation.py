"""Retrieval Evaluation 测试：尺子本身准不准。

评估这层最怕的是**尺子歪了**——名次算错、命中了说没中，然后拿这份数字去比
embedding 的好坏。所以这儿用完全可控的临时项目把名次钉死：四个函数都含
「处理」，期望的目标分别排在第 1 / 2 / 4 名、以及压根不在结果里（第 5 条），
于是 Hit@1 必须是 1/4、Hit@3 是 2/4、Hit@5 是 3/4——一个数一个数对。

真项目那组只验「跑得通、结果稳、没崩」，**不背具体命中率**：语料一变
（加个文件、改句注释）名次就会动，把当期数字写死在测试里，明天就是假红。

这个文件测的是「尺子准不准」（功能）；自带 case 集在真语料上的**效果回归**
（Hit@1/3/5 的下限快照、分组、反例）在 tests/test_evaluation_regression.py。
"""

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from index import (
    DEFAULT_CORPUS,
    EvaluationCase,
    EvaluationReport,
    Evaluator,
    Retriever,
    load_cases,
)
from index.__main__ import baseline_retrievers, build_corpora
from tools.base import ToolError


# 四个函数都含「处理」，按行先后排：alpha(1) / beta(5) / gamma(9) / delta(13)。
# 名次就是这么定下来的。`other` 不含这两个字，扫不到。
HANDLERS = "\n".join(
    [
        "def handle_alpha():",  # 1
        '    """处理甲。"""',  # 2
        "    return 1",  # 3
        "",  # 4
        "def handle_beta():",  # 5
        '    """处理乙。"""',  # 6
        "    return 2",  # 7
        "",  # 8
        "def handle_gamma():",  # 9
        '    """处理丙。"""',  # 10
        "    return 3",  # 11
        "",  # 12
        "def handle_delta():",  # 13
        '    """处理丁。"""',  # 14
        "    return 4",  # 15
        "",  # 16
        "def other():",  # 17
        "    return 0",  # 18
    ]
) + "\n"

# 另一个文件，专门给「限定名 / 简单名」那几条用。
BOX = "\n".join(
    [
        "class Box:",  # 1
        "    def put(self, item):",  # 2
        "        self.item = item",  # 3
    ]
) + "\n"


class EvaluationTestBase(unittest.TestCase):
    """临时目录当 root，`make_project()` 摆好那两份源码。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def make_project(self) -> None:
        self.make("pkg/handlers.py", HANDLERS)
        self.make("pkg/box.py", BOX)

    def retriever(self) -> Retriever:
        self.make_project()
        retriever = Retriever(root=self.root)
        retriever.index.build()
        return retriever

    def report(self, cases, top_k: int = 5, retriever: Retriever | None = None):
        return Evaluator(retriever or self.retriever(), top_k=top_k).run(cases)

    def chunk(self, symbol: str, path: str = "pkg/handlers.py"):
        """fixture 里某条 chunk，拿来单独验 matches()。"""
        found = self.retriever().index.chunks(path=path, symbol=symbol)
        self.assertEqual(len(found), 1, f"{path}:{symbol} 该正好一条 chunk")
        return found[0]

    def write_cases(self, text: str) -> Path:
        path = self.root / "cases.json"
        path.write_text(text, encoding="utf-8", newline="")
        return path

    def load(self, text: str):
        return load_cases(self.write_cases(text))


def case(query: str, symbol: str, scenario: str = "") -> EvaluationCase:
    return EvaluationCase(query, expected_symbol=symbol, scenario=scenario)


ALPHA = case("处理", "handle_alpha")  # 第 1 名
BETA = case("处理", "handle_beta")  # 第 2 名
DELTA = case("处理", "handle_delta")  # 第 4 名
OTHER = case("处理", "other")  # 前 5 条里没有
NOTHING = EvaluationCase("根本没有这个词", expected_nothing=True)


class CaseTest(EvaluationTestBase):
    """EvaluationCase：字段、校验、怎么算命中。"""

    def test_fields_are_stored(self):
        item = EvaluationCase(
            "查点什么",
            expected_path="a/b.py",
            expected_symbol="A.b",
            scenario="symbol",
            note="说明",
        )
        self.assertEqual(item.query, "查点什么")
        self.assertEqual(item.expected_path, "a/b.py")
        self.assertEqual(item.expected_symbol, "A.b")
        self.assertFalse(item.expected_nothing)
        self.assertEqual(item.scenario, "symbol")
        self.assertEqual(item.note, "说明")

    def test_defaults(self):
        item = EvaluationCase("x", expected_symbol="y")
        self.assertIsNone(item.expected_path)
        self.assertFalse(item.expected_nothing)
        self.assertEqual(item.scenario, "")
        self.assertEqual(item.note, "")

    def test_query_cannot_be_empty(self):
        for query in ("", "   ", "\t\n"):
            with self.assertRaises(ToolError):
                EvaluationCase(query, expected_symbol="y")

    def test_needs_something_to_expect(self):
        with self.assertRaises(ToolError):
            EvaluationCase("说了等于没说")

    def test_nothing_excludes_expectations(self):
        with self.assertRaises(ToolError):
            EvaluationCase("x", expected_symbol="y", expected_nothing=True)
        with self.assertRaises(ToolError):
            EvaluationCase("x", expected_path="a.py", expected_nothing=True)
        EvaluationCase("x", expected_nothing=True)  # 单独给没问题

    def test_target_reads_well(self):
        self.assertEqual(EvaluationCase("x", expected_symbol="A.b").target, "A.b")
        self.assertEqual(EvaluationCase("x", expected_path="a/b.py").target, "a/b.py")
        self.assertEqual(
            EvaluationCase("x", expected_path="a/b.py", expected_symbol="A.b").target,
            "a/b.py:A.b",
        )
        self.assertIn("查不到", EvaluationCase("x", expected_nothing=True).target)

    def test_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            ALPHA.query = "别的"

    def test_matches_by_path(self):
        chunk = self.chunk("handle_alpha")
        self.assertTrue(EvaluationCase("x", expected_path="pkg/handlers.py").matches(chunk))
        self.assertFalse(EvaluationCase("x", expected_path="pkg/box.py").matches(chunk))

    def test_matches_by_symbol(self):
        chunk = self.chunk("handle_alpha")
        self.assertTrue(case("x", "handle_alpha").matches(chunk))
        self.assertFalse(case("x", "handle_beta").matches(chunk))

    def test_simple_name_matches_qualified_symbol(self):
        """期望写 `put`，`Box.put` 也算中——查名字时简单名一向都认。"""
        chunk = self.chunk("Box.put", path="pkg/box.py")
        self.assertTrue(case("x", "put").matches(chunk))
        self.assertTrue(case("x", "Box.put").matches(chunk))

    def test_qualified_name_is_strict(self):
        """期望写 `Other.put` 就不算中：限定名是拿来区分同名方法的。"""
        chunk = self.chunk("Box.put", path="pkg/box.py")
        self.assertFalse(case("x", "Other.put").matches(chunk))

    def test_both_conditions_must_hold(self):
        chunk = self.chunk("handle_alpha")
        self.assertTrue(
            EvaluationCase(
                "x", expected_path="pkg/handlers.py", expected_symbol="handle_alpha"
            ).matches(chunk)
        )
        self.assertFalse(
            EvaluationCase(
                "x", expected_path="pkg/handlers.py", expected_symbol="handle_beta"
            ).matches(chunk)
        )
        self.assertFalse(
            EvaluationCase(
                "x", expected_path="pkg/box.py", expected_symbol="handle_alpha"
            ).matches(chunk)
        )


class ResultTest(EvaluationTestBase):
    """EvaluationResult：名次、命中标记、耗时、为什么。"""

    def test_rank_one(self):
        item = self.report([ALPHA]).results[0]
        self.assertEqual(item.rank, 1)
        self.assertTrue(item.hit1)
        self.assertTrue(item.hit3)
        self.assertTrue(item.hit5)
        self.assertEqual(item.query, "处理")
        self.assertEqual(len(item.results), 4)  # 四个都含「处理」
        self.assertGreater(item.elapsed_time, 0.0)
        self.assertFalse(item.found_nothing)
        self.assertIn("第 1 名", item.verdict)

    def test_rank_two(self):
        item = self.report([BETA]).results[0]
        self.assertEqual(item.rank, 2)
        self.assertFalse(item.hit1)
        self.assertTrue(item.hit3)
        self.assertTrue(item.hit5)

    def test_rank_four(self):
        item = self.report([DELTA]).results[0]
        self.assertEqual(item.rank, 4)
        self.assertFalse(item.hit1)
        self.assertFalse(item.hit3)
        self.assertTrue(item.hit5)

    def test_not_in_top_k(self):
        item = self.report([OTHER]).results[0]
        self.assertIsNone(item.rank)
        self.assertFalse(item.hit1)
        self.assertFalse(item.hit3)
        self.assertFalse(item.hit5)
        self.assertFalse(item.found_nothing)  # 有结果，只是没一个对
        self.assertIn("没命中", item.verdict)

    def test_query_with_no_results(self):
        item = self.report([case("根本没有这个词", "handle_alpha")]).results[0]
        self.assertEqual(item.results, ())
        self.assertIsNone(item.rank)
        self.assertTrue(item.found_nothing)
        self.assertFalse(item.hit1)
        self.assertIn("一条都没查到", item.verdict)

    def test_nothing_case_verdicts(self):
        ok = self.report([NOTHING]).results[0]
        self.assertTrue(ok.found_nothing)
        self.assertEqual(ok.verdict, "确实没查到")

        bad = self.report([EvaluationCase("处理", expected_nothing=True)]).results[0]
        self.assertFalse(bad.found_nothing)
        self.assertIn("不该查到", bad.verdict)

    def test_top_k_limits_what_is_fetched(self):
        """只取 2 条时，第 4 名的目标根本不在视野里——Hit@5 自然不成立。"""
        item = self.report([DELTA], top_k=2).results[0]
        self.assertEqual(len(item.results), 2)
        self.assertIsNone(item.rank)
        self.assertFalse(item.hit5)

    def test_is_frozen(self):
        item = self.report([ALPHA]).results[0]
        with self.assertRaises(FrozenInstanceError):
            item.rank = 9


class MetricsTest(EvaluationTestBase):
    """Hit@1 / Hit@3 / Hit@5 和平均耗时。"""

    def test_hit_rates(self):
        report = self.report([ALPHA, BETA, DELTA, OTHER])
        self.assertAlmostEqual(report.hit1, 0.25)  # 只有 ALPHA 在第 1 名
        self.assertAlmostEqual(report.hit3, 0.5)
        self.assertAlmostEqual(report.hit5, 0.75)  # DELTA 在第 4 名，赶上了

    def test_rates_are_monotonic(self):
        report = self.report([ALPHA, BETA, DELTA, OTHER])
        self.assertLessEqual(report.hit1, report.hit3)
        self.assertLessEqual(report.hit3, report.hit5)

    def test_nothing_cases_stay_out_of_hit_rates(self):
        """反例不是「该找到」的，命中率里不能算它——不然分越查越低。"""
        with_nothing = self.report([ALPHA, BETA, DELTA, OTHER, NOTHING])
        without = self.report([ALPHA, BETA, DELTA, OTHER])
        self.assertEqual(with_nothing.hit1, without.hit1)
        self.assertEqual(with_nothing.hit3, without.hit3)
        self.assertEqual(with_nothing.hit5, without.hit5)
        self.assertEqual(len(with_nothing.answerable), 4)
        self.assertEqual(len(with_nothing.nothing_cases), 1)

    def test_a_wrong_nothing_case_is_recorded(self):
        """反例写错了（或者检索变瞎了）：查到东西就是不过，但也不进命中率。"""
        report = self.report([EvaluationCase("处理", expected_nothing=True)])
        self.assertEqual(len(report.nothing_cases), 1)
        self.assertFalse(report.nothing_cases[0].found_nothing)
        self.assertEqual(report.answerable, ())
        self.assertEqual(report.hit1, 0.0)

    def test_misses_lists_what_was_not_found(self):
        report = self.report([ALPHA, BETA, DELTA, OTHER, NOTHING])
        self.assertEqual([item.case for item in report.misses], [OTHER])

    def test_average_time_is_the_mean(self):
        report = self.report([ALPHA, BETA, DELTA, OTHER, NOTHING])
        times = [item.elapsed_time for item in report.results]
        self.assertAlmostEqual(report.average_time, sum(times) / len(times))
        self.assertGreater(report.average_time, 0.0)

    def test_empty_case_list(self):
        report = self.report([])
        self.assertEqual(report.results, ())
        self.assertEqual(report.hit1, 0.0)
        self.assertEqual(report.hit3, 0.0)
        self.assertEqual(report.hit5, 0.0)
        self.assertEqual(report.average_time, 0.0)
        self.assertIsInstance(report.summary(), str)

    def test_case_order_is_kept(self):
        report = self.report([DELTA, ALPHA, BETA])
        self.assertEqual([item.case for item in report.results], [DELTA, ALPHA, BETA])

    def test_hit_flags_agree_with_rank(self):
        report = self.report([ALPHA, BETA, DELTA, OTHER])
        for item in report.results:
            self.assertEqual(item.hit1, item.rank == 1)
            self.assertEqual(item.hit3, item.rank is not None and item.rank <= 3)
            self.assertEqual(item.hit5, item.rank is not None and item.rank <= 5)

    def test_summary_reads_well(self):
        text = self.report([ALPHA, BETA, DELTA, OTHER, NOTHING]).summary()
        self.assertIn("共 5 条 case", text)
        self.assertIn("Hit@1 1/4", text)
        self.assertIn("Hit@3 2/4", text)
        self.assertIn("Hit@5 3/4", text)
        self.assertIn("确实没查到", text)
        self.assertIn("处理", text)
        self.assertIn("ms", text)

    def test_report_is_frozen(self):
        report = self.report([ALPHA])
        with self.assertRaises(FrozenInstanceError):
            report.results = ()
        self.assertIsInstance(report, EvaluationReport)


class StabilityTest(EvaluationTestBase):
    """同一组 case 跑两遍，除了耗时，别的都得一样。"""

    def test_same_ranks_and_order(self):
        cases = [ALPHA, BETA, DELTA, OTHER, NOTHING]
        first = self.report(cases)
        second = self.report(cases)
        self.assertEqual(
            [item.rank for item in first.results],
            [item.rank for item in second.results],
        )
        self.assertEqual(
            [item.verdict for item in first.results],
            [item.verdict for item in second.results],
        )
        self.assertEqual(
            [[one.chunk.id for one in item.results] for item in first.results],
            [[one.chunk.id for one in item.results] for item in second.results],
        )
        self.assertEqual(first.hit1, second.hit1)
        self.assertEqual(first.hit5, second.hit5)

    def test_evaluation_does_not_touch_the_index(self):
        retriever = self.retriever()
        before = retriever.index.chunks()
        self.report([ALPHA, BETA], retriever=retriever)
        self.assertEqual(retriever.index.chunks(), before)


class LoaderTest(EvaluationTestBase):
    """load_cases：从 JSON 读 case，写错要吵出来。"""

    def test_default_cases_load(self):
        cases = load_cases()
        self.assertTrue(cases)
        for item in cases:
            self.assertTrue(item.query.strip())
            self.assertTrue(item.expected_nothing or item.expected_path or item.expected_symbol)

    def test_default_cases_cover_the_scenarios(self):
        scenarios = {item.scenario for item in load_cases()}
        self.assertTrue(
            {"file_name", "symbol", "class", "function", "description", "keyword", "nothing"}
            <= scenarios,
            f"自带 case 没覆盖全场景：{sorted(scenarios)}",
        )

    def test_default_cases_have_no_duplicates(self):
        queries = [item.query for item in load_cases()]
        self.assertEqual(len(set(queries)), len(queries), "同一句查询写了两次")

    def test_load_from_a_file(self):
        path = self.write_cases(json.dumps([
            {"query": "查点什么", "expected_path": "a/b.py", "scenario": "file_name", "note": "说明"}
        ], ensure_ascii=False))
        cases = load_cases(path)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].query, "查点什么")
        self.assertEqual(cases[0].expected_path, "a/b.py")
        self.assertEqual(cases[0].scenario, "file_name")

    def test_unknown_field_is_an_error(self):
        """字段名拼错不能静默忽略：那条 case 会永远不命中，看着像检索不行。"""
        with self.assertRaises(ToolError):
            self.load('[{"query": "x", "expected_symobl": "y"}]')

    def test_wrong_type_is_an_error(self):
        with self.assertRaises(ToolError):
            self.load('[{"query": "x", "expected_nothing": "true"}]')
        with self.assertRaises(ToolError):
            self.load('[{"query": 3, "expected_symbol": "y"}]')

    def test_missing_query_is_an_error(self):
        with self.assertRaises(ToolError):
            self.load('[{"expected_symbol": "y"}]')

    def test_no_expectation_is_an_error(self):
        with self.assertRaises(ToolError):
            self.load('[{"query": "x"}]')

    def test_top_level_must_be_a_list(self):
        with self.assertRaises(ToolError):
            self.load('{"query": "x", "expected_symbol": "y"}')

    def test_broken_json_is_an_error(self):
        with self.assertRaises(ToolError):
            self.load("[{")

    def test_missing_file_is_an_error(self):
        with self.assertRaises(ToolError):
            load_cases(self.root / "没有这个文件.json")


class RealProjectEvaluationTest(unittest.TestCase):
    """拿真项目跑自带 case 集：验的是「能跑、稳、不崩」，不是具体分数。"""

    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        # 9.5 起自带 case 分两份语料（项目 + C++ fixture），各配一个 retriever
        cls.corpora = build_corpora()
        cls.retriever = baseline_retrievers(cls.corpora)[DEFAULT_CORPUS]
        cls.report = Evaluator(baseline_retrievers(cls.corpora)).run(cls.cases)

    def test_every_case_has_a_result(self):
        self.assertEqual(len(self.report.results), len(self.cases))
        self.assertEqual(
            [item.case for item in self.report.results], list(self.cases)
        )

    def test_rates_are_sane(self):
        for rate in (self.report.hit1, self.report.hit3, self.report.hit5):
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)
        self.assertLessEqual(self.report.hit1, self.report.hit3)
        self.assertLessEqual(self.report.hit3, self.report.hit5)

    def test_it_still_finds_what_literal_scoring_is_for(self):
        """回归报警线：字面打分的主场——词法查询——还得照旧进前 5。

        9.5 把 case 集扩到 37 条正例，其中 25 条是**语义**查询：字面打分
        够不着它们是数据集故意的（那正是 embedding 要补的位置），拿整体
        hit5 当报警线只会天天误报（实测 32.4%）。词法那 12 条（名字 /
        路径 / 关键词）是字面打分该做到的，掉下去才是回归。
        """
        lexical = self.report.group_by("query_type")["lexical"]
        self.assertGreaterEqual(lexical.hit5, 0.75)

    def test_a_nothing_case_stays_empty(self):
        self.assertTrue(self.report.nothing_cases)
        for item in self.report.nothing_cases:
            self.assertTrue(
                item.found_nothing,
                f"{item.query!r} 本该查不到，却查到了 {len(item.results)} 条",
            )

    def test_results_come_from_the_index(self):
        everything = set()
        for index in self.corpora.values():
            everything.update(index.chunks())
        for item in self.report.results:
            for one in item.results:
                self.assertIn(one.chunk, everything)

    def test_stable_across_runs(self):
        again = Evaluator(baseline_retrievers(self.corpora)).run(self.cases)
        self.assertEqual(
            [item.verdict for item in self.report.results],
            [item.verdict for item in again.results],
        )
        self.assertEqual(self.report.hit5, again.hit5)

    def test_summary_is_printable(self):
        text = self.report.summary()
        self.assertIn("Hit@1", text)
        self.assertIn("平均检索耗时", text)
        # 逐条 + 空行 + 总账 + 反例那行 + 耗时那行
        self.assertEqual(len(text.splitlines()), len(self.cases) + 4)


if __name__ == "__main__":
    unittest.main()
