"""Fake Embedding 测试：接口通不通、管线对不对、假货只在该说话时说话。

三组分开钉：

- **假货本身**（FakeEmbedder / cosine）：查表出向量、顺序对应、零向量不
  会除出花来、写坏的表直接报错。
- **scorer 和 hybrid**：相似度过线才给 Match、分数 = 相似度 × 权重、
  查询和 chunk 各只 embed 一次（真接模型时这条就是钱）、关掉 embedding
  检索行为和原来逐条一致。
- **评估对照**：小项目上一句 baseline 一条都查不到的人话，hybrid 排到
  第 1（这条不受语料漂移影响）；真项目上拿自带 case 集比 baseline 和
  hybrid——8.5 里被类块压住的那条 description，8.6 靠语义分提到第 1，
  8.7 起 baseline 的粒度奖励自己就能排到第 1（hybrid 至少不许弄坏），
  其它 case 不该被弄差。

注意本文件也在语料里（项目把自己也索引了）：**不许把 case 的查询原样
写进来**——写进来就成了语料的一部分，那条 case 会命中这个测试文件自己
（8.4 / 8.5 都踩过）。查询在这里统统是拼出来的。
"""

import tempfile
import unittest
from pathlib import Path

from index import (
    DEFAULT_CORPUS,
    DEFAULT_SCORERS,
    DEMO_ALIASES,
    ChunkIndex,
    CodeChunk,
    Embedder,
    EmbeddingScorer,
    EvaluationCase,
    Evaluator,
    FakeEmbedder,
    FUNCTION,
    Query,
    Retriever,
    cosine,
    hybrid_scorers,
    load_cases,
)
from index.__main__ import baseline_retrievers, build_corpora
from tools.base import ToolError


# ---------------------------------------------------------------------------
# 小项目 fixture：两句人话查询在这儿有确定的答案。
#
# 「记录有没有被人动过」这条 baseline 一条都查不到（整句没在谁正文里出现，
# 拆出来的又只有一个长词）；假货认得「记录 = row」「改过 = 被改」，于是
#   verify_row 撞上 记录 + 被改 → 相似度 2/√(2·3) ≈ 0.82，排第一。
# ---------------------------------------------------------------------------

STORE = "\n".join(
    [
        "def save_row():",  # 1
        '    """保存一条记录。"""',  # 2
        "    return 1",  # 3
        "",  # 4
        "def verify_row():",  # 5
        '    """核对记录有没有被改过。"""',  # 6
        "    return 2",  # 7
        "",  # 8
        "def unrelated():",  # 9
        '    """和这条线没关系。"""',  # 10
        "    return 3",  # 11
    ]
) + "\n"

ROW_ALIASES = (
    ("核对", "校验", "verify"),
    ("改过", "被改", "changed"),
    ("记录", "row"),
)

SEMANTIC_QUERY = "记录有没有被人改过"

# 8.5 那条 description 的查询。拼起来：本文件也在语料里，原样写进来那条
# case 就会命中这个测试文件自己（8.4 / 8.5 都踩过这个坑）。
CLASS_BLOCKED_QUERY = "把文件" + "读回来"


class EmbeddingTestBase(unittest.TestCase):
    """临时目录当 root，`retriever()` 摆好 fixture 并建好索引。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def retriever(self, scorers=DEFAULT_SCORERS) -> Retriever:
        self.make("pkg/store.py", STORE)
        retriever = Retriever(root=self.root, scorers=scorers)
        retriever.index.build()
        return retriever

    def symbols(self, results) -> list[str]:
        return [item.chunk.symbol for item in results]


class SpyEmbedder(Embedder):
    """包一层，记下每次 embed 收到的文本——数调用次数用。"""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.texts: list[str] = []

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed(self, texts):
        self.texts.extend(texts)
        return self.inner.embed(texts)


def query(text: str) -> Query:
    return Query(text=text, stripped=text.lower(), tokens=(text.lower(),))


def make_chunk(symbol: str, code: str, digest: str = "d0d0d0d0") -> CodeChunk:
    return CodeChunk(
        path="pkg/store.py",
        symbol=symbol,
        kind=FUNCTION,
        language="python",
        start_line=1,
        end_line=2,
        digest=digest,
        code=code,
    )


class CosineTest(unittest.TestCase):
    def test_same_direction_is_one(self):
        self.assertAlmostEqual(cosine((1.0, 1.0), (2.0, 2.0)), 1.0)

    def test_orthogonal_is_zero(self):
        self.assertEqual(cosine((1.0, 0.0), (0.0, 1.0)), 0.0)

    def test_zero_vector_is_zero_not_an_error(self):
        self.assertEqual(cosine((0.0, 0.0), (1.0, 1.0)), 0.0)
        self.assertEqual(cosine((0.0, 0.0), (0.0, 0.0)), 0.0)

    def test_length_mismatch_is_refused(self):
        with self.assertRaises(ToolError):
            cosine((1.0,), (1.0, 2.0))


class FakeEmbedderTest(unittest.TestCase):
    """假货的规矩：一组词一个维度，撞上就是 1，撞上几组就是几个 1。"""

    ALIASES = (("核对", "verify"), ("记录", "row"))

    def test_dimension_is_the_group_count(self):
        self.assertEqual(FakeEmbedder(self.ALIASES).dimension, 2)
        self.assertEqual(FakeEmbedder().dimension, 0)

    def test_alias_hit_is_one(self):
        embedder = FakeEmbedder(self.ALIASES)
        (vector,) = embedder.embed(("这条记录要保存",))
        self.assertEqual(vector, (0.0, 1.0))

    def test_alias_miss_is_zero(self):
        embedder = FakeEmbedder(self.ALIASES)
        (vector,) = embedder.embed(("完全无关的一句话",))
        self.assertEqual(vector, (0.0, 0.0))

    def test_matching_ignores_case(self):
        embedder = FakeEmbedder(self.ALIASES)
        (vector,) = embedder.embed(("Please VERIFY this",))
        self.assertEqual(vector, (1.0, 0.0))

    def test_cjk_matches_as_substring(self):
        """没有分词器，中文就拿表里的说法去撞，撞得上就算。"""
        embedder = FakeEmbedder(self.ALIASES)
        (vector,) = embedder.embed(("核对一下",))
        self.assertEqual(vector, (1.0, 0.0))

    def test_batch_keeps_order_and_count(self):
        embedder = FakeEmbedder(self.ALIASES)
        vectors = embedder.embed(("只有记录", "只有核对"))
        self.assertEqual(vectors, ((0.0, 1.0), (1.0, 0.0)))

    def test_same_text_same_vector(self):
        embedder = FakeEmbedder(self.ALIASES)
        self.assertEqual(embedder.embed(("记录",)), embedder.embed(("记录",)))

    def test_empty_table_gives_empty_vectors(self):
        embedder = FakeEmbedder()
        self.assertEqual(embedder.embed(("随便什么",)), ((),))

    def test_blank_alias_is_refused(self):
        with self.assertRaises(ToolError):
            FakeEmbedder((("核对", ""),))
        with self.assertRaises(ToolError):
            FakeEmbedder(((),))


class EmbeddingScorerTest(EmbeddingTestBase):
    """scorer 这层：分数、门槛、闭嘴的时机、缓存。"""

    def test_match_carries_similarity_and_weight(self):
        scorer = EmbeddingScorer(
            FakeEmbedder((("甲",), ("乙",))), weight=50.0, threshold=0.5
        )
        matches = scorer(query("甲"), make_chunk("f", "只有甲"))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].kind, "embedding")
        self.assertAlmostEqual(matches[0].score, 50.0)  # 相似度 1.0 × 50
        self.assertIn("1.00", matches[0].detail)

    def test_below_threshold_nothing_is_returned(self):
        # 查询只撞 1 组、chunk 撞 2 组（多出来的那组把相似度稀释到 0.71）。
        scorer = EmbeddingScorer(
            FakeEmbedder((("甲",), ("乙",))), weight=50.0, threshold=0.9
        )
        self.assertEqual(scorer(query("甲"), make_chunk("f", "甲和乙都写了")), [])

    def test_zero_similarity_never_matches_even_at_threshold_zero(self):
        """threshold 给到 0 是「谁有点像谁进」，但不是「谁都能进」——一点
        都对不上（零向量）的不给 0 分结果。"""
        scorer = EmbeddingScorer(
            FakeEmbedder((("甲",),)), weight=50.0, threshold=0.0
        )
        self.assertEqual(scorer(query("完全不相干"), make_chunk("f", "return 1")), [])

    def test_weight_must_be_positive(self):
        with self.assertRaises(ToolError):
            EmbeddingScorer(FakeEmbedder((("甲",),)), weight=0.0)

    def test_threshold_must_be_a_similarity(self):
        with self.assertRaises(ToolError):
            EmbeddingScorer(FakeEmbedder((("甲",),)), threshold=1.5)

    def test_everything_is_embedded_exactly_once(self):
        """一句查询 + 每条 chunk 各 embed 一次——查第二遍一条都不用重算
        （Query 按值缓存，chunk 按 id + digest）。真接模型时这一条就是钱。"""
        spy = SpyEmbedder(FakeEmbedder(ROW_ALIASES))
        retriever = self.retriever(DEFAULT_SCORERS + (EmbeddingScorer(spy),))
        retriever.search(SEMANTIC_QUERY)
        self.assertEqual(len(spy.texts), 1 + len(retriever.index.chunks()))
        retriever.search(SEMANTIC_QUERY)
        self.assertEqual(len(spy.texts), 1 + len(retriever.index.chunks()))

    def test_each_new_query_is_embedded(self):
        spy = SpyEmbedder(FakeEmbedder(ROW_ALIASES))
        retriever = self.retriever(DEFAULT_SCORERS + (EmbeddingScorer(spy),))
        retriever.search(SEMANTIC_QUERY)
        retriever.search("另一句话")
        queries = [text for text in spy.texts if text in (SEMANTIC_QUERY, "另一句话")]
        self.assertEqual(queries, [SEMANTIC_QUERY, "另一句话"])

    def test_changed_chunk_is_embedded_again(self):
        """digest 变了就是另一条 chunk——不能拿旧向量给新代码打分。"""
        spy = SpyEmbedder(FakeEmbedder(ROW_ALIASES))
        scorer = EmbeddingScorer(spy)
        scorer(query("记录"), make_chunk("f", "记录在这儿", digest="aaaa"))
        scorer(query("记录"), make_chunk("f", "记录在这儿", digest="bbbb"))
        # 两句查询内容一样（Query 按值相等，第二句是缓存命中），chunk 换了
        # digest 就得重算：1 + 1 + 1 = 3 次。
        self.assertEqual(len(spy.texts), 3)

    def test_embedded_text_includes_path_and_symbol(self):
        """正文里没有、符号名里有的词，假货也得认——embed 的文本是
        「路径 + 符号名 + 正文」，不只正文。"""
        scorer = EmbeddingScorer(FakeEmbedder(ROW_ALIASES), threshold=0.1)
        matches = scorer(query("记录"), make_chunk("row_total", "return 42"))
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0].score, 40.0)  # 相似度 1.0 × 默认权重

    def test_wrong_dimension_is_refused(self):
        class BadEmbedder(Embedder):
            @property
            def dimension(self) -> int:
                return 2

            def embed(self, texts):
                return tuple((1.0,) for _ in texts)  # 少了一维

        scorer = EmbeddingScorer(BadEmbedder())
        with self.assertRaises(ToolError):
            scorer(query("甲"), make_chunk("f", "甲"))

    def test_wrong_batch_size_is_refused(self):
        class BadEmbedder(Embedder):
            @property
            def dimension(self) -> int:
                return 1

            def embed(self, texts):
                return ()  # 一条都不回

        scorer = EmbeddingScorer(BadEmbedder())
        with self.assertRaises(ToolError):
            scorer(query("甲"), make_chunk("f", "甲"))


class HybridRetrieverTest(EmbeddingTestBase):
    """检索这层：hybrid 能捞东西、不抢原有打分器的戏、关掉就回到老样子。"""

    def hybrid(self) -> Retriever:
        return self.retriever(
            scorers=hybrid_scorers(FakeEmbedder(ROW_ALIASES))
        )

    def test_semantic_query_finds_what_keywords_cannot(self):
        self.assertEqual(self.retriever().search(SEMANTIC_QUERY), ())
        results = self.hybrid().search(SEMANTIC_QUERY)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.symbol, "verify_row")
        self.assertIn("embedding", [item.kind for item in results[0].matches])

    def test_keyword_hits_are_not_disturbed(self):
        """同一句查询，原来靠字面命中的 chunk 一条不少、分数一分不差；
        embedding 只是在旁边又添了一些候选。"""
        plain = self.retriever().search("save_row")
        hybrid = self.hybrid().search("save_row")
        by_symbol = {item.chunk.symbol: item for item in hybrid}
        self.assertTrue(plain)
        for item in plain:
            self.assertIn(item.chunk.symbol, by_symbol)
            kept = [
                match for match in by_symbol[item.chunk.symbol].matches
                if match.kind != "embedding"
            ]
            self.assertEqual(kept, list(item.matches))
        self.assertEqual(hybrid[0].chunk.symbol, "save_row")

    def test_without_embedding_nothing_changes(self):
        """embedding 不存在时，结果和 8.4 逐条一致（分数也一样）。"""
        explicit = self.retriever(scorers=DEFAULT_SCORERS).search("核对记录")
        default = self.retriever().search("核对记录")
        self.assertEqual(explicit, default)
        self.assertTrue(default)

    def test_unknown_words_stay_empty(self):
        """表里没有、代码里也没有的词，hybrid 也不许凭空给结果。"""
        nonsense = "量子" + "纠缠"  # 拼起来：本文件也在语料里，见文件头
        self.assertEqual(self.hybrid().search(nonsense), ())

    def test_hybrid_order_is_stable(self):
        first = self.hybrid().search(SEMANTIC_QUERY)
        second = self.hybrid().search(SEMANTIC_QUERY)
        self.assertEqual([item.chunk.id for item in first],
                         [item.chunk.id for item in second])
        self.assertEqual(self.symbols(first), self.symbols(second))

    def test_hybrid_scorers_keeps_the_defaults(self):
        scorers = hybrid_scorers(FakeEmbedder(ROW_ALIASES))
        self.assertEqual(scorers[: len(DEFAULT_SCORERS)], DEFAULT_SCORERS)
        self.assertEqual(len(scorers), len(DEFAULT_SCORERS) + 1)
        self.assertIsInstance(scorers[-1], EmbeddingScorer)


class FixtureComparisonTest(EmbeddingTestBase):
    """小项目上的 baseline vs hybrid：一句话从查不到变成第 1 名。

    这条不依赖真项目语料——就算 README 里那些数字哪天漂了，这条对照还在。
    """

    def test_a_human_phrase_goes_from_nothing_to_first(self):
        case = EvaluationCase(SEMANTIC_QUERY, expected_symbol="verify_row")
        self.make("pkg/store.py", STORE)
        index = ChunkIndex(root=self.root)  # 两份报告共用一份索引
        index.build()

        baseline = Evaluator(Retriever(index=index)).run((case,))
        hybrid = Evaluator(
            Retriever(index=index, scorers=hybrid_scorers(FakeEmbedder(ROW_ALIASES)))
        ).run((case,))

        self.assertIsNone(baseline.results[0].rank, "baseline 本该查不到")
        self.assertEqual(hybrid.results[0].rank, 1, "hybrid 该把它排到第 1")


class RealProjectComparisonTest(unittest.TestCase):
    """拿真项目和自带 case 集比 baseline 与 hybrid——8.6 的正脸。

    只钉两条事实：① 8.5 里被类块压住的那条 description 被提了名次；② 没有
    任何 case 被弄差、反例还是空的。具体命中率不写死（语料一变就漂），
    但下面这条 demo 口径依赖 DEMO_ALIASES——改了词表就得回来重跑。
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        # 9.5 起 case 集分两份语料（项目 + C++ fixture）：按语料表各配一个
        # retriever，baseline / hybrid 都是同一个形状
        cls.corpora = build_corpora()
        cls.index = cls.corpora[DEFAULT_CORPUS]
        cls.baseline = Evaluator(baseline_retrievers(cls.corpora)).run(cls.cases)
        cls.hybrid = Evaluator(
            {
                name: Retriever(
                    index=index, scorers=hybrid_scorers(FakeEmbedder(DEMO_ALIASES))
                )
                for name, index in cls.corpora.items()
            }
        ).run(cls.cases)
        cls.baseline_rank = {item.case.query: item.rank for item in cls.baseline.results}
        cls.hybrid_rank = {item.case.query: item.rank for item in cls.hybrid.results}

    def test_the_blocked_description_is_first_for_both(self):
        """这条曾被类块压在第 2 名（8.5 的诊断）。8.6 靠语义分把它提到第 1；
        8.7 起 baseline 的粒度奖励自己就能把它排到第 1——语义分在这条上
        不再是增量，hybrid 至少不许把它弄坏。"""
        self.assertEqual(self.baseline_rank[CLASS_BLOCKED_QUERY], 1)
        self.assertEqual(self.hybrid_rank[CLASS_BLOCKED_QUERY], 1)

    def test_no_case_gets_worse(self):
        for case in self.cases:
            before = self.baseline_rank[case.query]
            after = self.hybrid_rank[case.query]
            if before is None:
                continue  # baseline 就没查到（反例 / 那条人话），没有「弄差」可言
            self.assertIsNotNone(after, f"{case.query!r} 被弄丢了")
            self.assertLessEqual(after, before, f"{case.query!r} 名次退步了")

    def test_an_unrelated_negative_stays_empty(self):
        """真正毫无关联的反例，hybrid 也不许凭空给结果。"""
        for item in self.hybrid.nothing_cases:
            if "symbol" in item.query:
                continue  # 它含着真词，见下一条
            self.assertTrue(
                item.found_nothing,
                f"{item.query!r} 本该查不到，却查到了 {len(item.results)} 条",
            )

    def test_a_missing_identifier_matches_its_words(self):
        """那条标识符反例里有个「symbol」：假 embedding 认字，不认
        「这个标识符根本不存在」——于是捞出一堆讲符号的块。这不是 bug，是
        语义检索改写了「查不到」的定义（真模型多半也会这样）。结果必须
        **全部来自 embedding 那一条理由**：关键词一条都不许中，它确实不存在。
        给 hybrid 准备反例时得挑真正无关的词——另一条反例那种。
        """
        query = "missing_symbol" + "_zzz"  # 拼起来：本文件也在语料里
        item = next(i for i in self.hybrid.nothing_cases if i.query == query)
        self.assertFalse(item.found_nothing, "现在它会被语义捞出来")
        for one in item.results:
            kinds = {match.kind for match in one.matches}
            self.assertEqual(kinds, {"embedding"})

    def test_hit_rates_do_not_drop(self):
        self.assertGreaterEqual(self.hybrid.hit1, self.baseline.hit1)
        self.assertGreaterEqual(self.hybrid.hit3, self.baseline.hit3)
        self.assertGreaterEqual(self.hybrid.hit5, self.baseline.hit5)

    def test_results_come_from_the_index(self):
        everything = set()
        for index in self.corpora.values():
            everything.update(index.chunks())
        for item in self.hybrid.results:
            for one in item.results:
                self.assertIn(one.chunk, everything)

    def test_stable_across_runs(self):
        """同一个索引、同一个词表，跑两遍名次一样（新 scorer 从零缓存重算）。"""
        again = Evaluator(
            {
                name: Retriever(
                    index=index,
                    scorers=hybrid_scorers(FakeEmbedder(DEMO_ALIASES)),
                )
                for name, index in self.corpora.items()
            }
        ).run(self.cases)
        self.assertEqual(
            [item.rank for item in again.results],
            [item.rank for item in self.hybrid.results],
        )


if __name__ == "__main__":
    unittest.main()
