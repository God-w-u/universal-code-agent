"""9.5 分段 embedding 的离线测试。

三块：

- **切分本身**（`index/segments.py`）：估计规则、短文本一段、长文本多段、
  每段都装得进预算、每一段都带位置行、顺序和内容一个不丢、单行 / 整段超长
  的硬切、确定性、参数校验。
- **接进 scorer**：声明了窗口的 embedder 才切；一个块多个向量；**打分取
  「和查询最像的那一段」**（max 池化——这是修「后半段被截掉」的关键，
  所以专门有一条测试钉它）；9.4 的老形状（没声明窗口 / 关掉开关）逐条不变。
- **缓存与降级**：分段后 key 带段号、命名空间带切法版本、换开关就换文件、
  第二次跑不惊动模型；embedder 炸了照样退回字面检索。

全部离线：假 embedder 只有 `dimension` / `embed`（外加一个声明窗口的
`max_tokens`），不装库、不联网、不下载模型。这个文件里**不写自带的 case
查询原句**——项目把自己也索引了，写了就会命中测试文件自己（8.4 踩过）。
"""

import tempfile
import unittest
from pathlib import Path

from index import (
    SEGMENT_SAFETY_MARGIN,
    SEGMENT_STRATEGY,
    TEXT_STRATEGY,
    ChunkIndex,
    CodeChunk,
    DegradingScorer,
    Embedder,
    EmbeddingScorer,
    FakeEmbedder,
    Query,
    Retriever,
    VectorCache,
    cache_file,
    cache_for,
    chunk_key,
    combined_stats_line,
    compact_text,
    estimate_tokens,
    hybrid_scorers,
    namespace_of,
    segment_chunk,
    segment_text,
    strategy_id,
)
from tools.base import ToolError

DIGEST = "d0d0d0d0"


def make_chunk(
    code: str,
    *,
    symbol: str = "do_work",
    path: str = "pkg/store.py",
    kind: str = "function",
    digest: str = DIGEST,
    start_line: int = 1,
) -> CodeChunk:
    """手搓一条 chunk：测试要的是可控的正文，不必真去建索引。"""
    return CodeChunk(
        path=path,
        symbol=symbol,
        kind=kind,
        language="python",
        start_line=start_line,
        end_line=start_line + code.count("\n"),
        digest=digest,
        code=code,
    )


def query(text: str) -> Query:
    return Query(text=text, stripped=text.lower(), tokens=(text.lower(),))


class StubEmbedder(Embedder):
    """写死向量的假 embedder：撞上关键词给一支向量，否则给另一支。

    `window` 声明模型窗口（`None` = 没声明，不切分）；`batches` 记下每次
    `embed()` 收到的那批文本，用来钉「一批只调一次」和「顺序对得上」。
    """

    def __init__(
        self,
        window: int | None = None,
        *,
        markers: tuple[tuple[str, tuple[float, ...]], ...] = (),
        base: tuple[float, ...] = (0.1, 0.9, 0.0),
        dimension: int = 3,
    ) -> None:
        self._window = window
        self._markers = markers
        self._base = base
        self._dimension = dimension
        self.batches: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> int | None:
        return self._window

    def embed(self, texts):
        batch = tuple(texts)
        self.batches.append(batch)
        return tuple(self._vector(text) for text in batch)

    def _vector(self, text: str) -> tuple[float, ...]:
        for marker, vector in self._markers:
            if marker in text:
                return vector
        return self._base


# ---------------------------------------------------------------------------
# 1. token 估计与切分本身
# ---------------------------------------------------------------------------


class EstimateTokensTest(unittest.TestCase):
    def test_empty_text_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_a_chinese_character_counts_as_one(self):
        self.assertEqual(estimate_tokens("校验文件"), 4)

    def test_ascii_counts_two_characters_per_token(self):
        self.assertEqual(estimate_tokens("abcd"), 2)
        self.assertEqual(estimate_tokens("abcde"), 3)  # 向上取整，宁多勿少


class SegmentTextTest(unittest.TestCase):
    WINDOW = 32  # 预算 = 30

    def budget(self) -> int:
        return self.WINDOW - SEGMENT_SAFETY_MARGIN

    def test_a_short_text_comes_back_untouched(self):
        text = "pkg/store.py:do_work [function]\nprint(1)"

        segments = segment_text(text, self.WINDOW)

        self.assertEqual(segments, (text,))

    def test_a_text_exactly_at_the_budget_stays_in_one_piece(self):
        text = "x" * (self.budget() * 2)  # 都是 ASCII：正好 30 个 token

        self.assertEqual(estimate_tokens(text), self.budget())
        self.assertEqual(segment_text(text, self.WINDOW), (text,))

    def test_an_empty_text_is_still_one_segment(self):
        self.assertEqual(segment_text("", self.WINDOW), ("",))

    def test_a_long_text_is_split_into_fitting_pieces(self):
        header = "pkg/store.py:do_work [function]"
        body = "\n".join("line %02d of the body" % index for index in range(40))
        text = f"{header}\n{body}"

        segments = segment_text(text, self.WINDOW)

        self.assertGreater(len(segments), 1)
        for segment in segments:
            self.assertLessEqual(estimate_tokens(segment), self.budget())

    def test_every_segment_carries_the_location_line(self):
        header = "pkg/store.py:do_work [function]"
        text = header + "\n" + "\n".join("payload " * 4 for _ in range(20))

        segments = segment_text(text, self.WINDOW)

        for segment in segments:
            self.assertTrue(segment.startswith(header), segment)

    def test_nothing_is_lost_and_the_order_is_kept(self):
        header = "pkg/store.py:do_work [function]"
        body = "\n".join("line %02d of the body" % index for index in range(40))
        text = f"{header}\n{body}"

        segments = segment_text(text, self.WINDOW)

        # 每段开头的重复位置行去掉，剩下的拼起来该和原文一模一样
        # （只差切段处那些换行）
        rebuilt = "".join(
            segment[len(header) + 1 :] if segment.startswith(header + "\n") else segment
            for segment in segments
        )
        self.assertEqual(rebuilt, body.replace("\n", ""))

    def test_one_over_long_line_is_hard_split(self):
        header = "pkg/store.py:do_work [function]"
        line = "y" * 200
        text = f"{header}\n{line}"

        segments = segment_text(text, self.WINDOW)

        self.assertGreater(len(segments), 1)
        for segment in segments:
            self.assertLessEqual(estimate_tokens(segment), self.budget())
            self.assertTrue(segment.startswith(header))
        rebuilt = "".join(
            segment[len(header) + 1 :] for segment in segments
        )
        self.assertEqual(rebuilt, line)

    def test_a_header_that_does_not_fit_falls_back_to_a_hard_split(self):
        text = "p" * 400  # 一行超长，没有换行，位置行自己就装不下

        segments = segment_text(text, self.WINDOW)

        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segments), text)
        for segment in segments:
            self.assertLessEqual(estimate_tokens(segment), self.budget())

    def test_the_same_input_gives_the_same_segments(self):
        text = "pkg/store.py:do_work [function]\n" + "\n".join(
            "payload " * 4 for _ in range(20)
        )

        self.assertEqual(segment_text(text, self.WINDOW), segment_text(text, self.WINDOW))

    def test_a_window_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            segment_text("anything", 0)


class SegmentChunkTest(unittest.TestCase):
    def test_it_segments_the_compact_text(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n    return 1\n')

        segments = segment_chunk(chunk, 128)

        self.assertEqual(segments, (compact_text(chunk),))

    def test_the_location_line_survives_a_reasonable_window(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n    return 1\n')

        segments = segment_chunk(chunk, 128)

        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0].startswith(f"{chunk.path}:{chunk.symbol}"))

    def test_the_code_of_the_chunk_is_not_touched(self):
        code = 'def do_work():\n    """干活。"""\n    return 1\n'
        chunk = make_chunk(code)

        segment_chunk(chunk, 8)

        self.assertEqual(chunk.code, code)
        self.assertEqual(chunk.symbol, "do_work")


class StrategyIdTest(unittest.TestCase):
    def test_without_segments_it_is_the_text_strategy(self):
        self.assertEqual(strategy_id(False), TEXT_STRATEGY)

    def test_with_segments_it_carries_both_versions(self):
        value = strategy_id(True)

        self.assertIn(TEXT_STRATEGY, value)
        self.assertIn(SEGMENT_STRATEGY, value)
        self.assertNotEqual(value, TEXT_STRATEGY)


# ---------------------------------------------------------------------------
# 2. EmbeddingScorer：一个块多个向量，打分取最像的那段
# ---------------------------------------------------------------------------


class WindowTest(unittest.TestCase):
    def test_a_declared_window_turns_segmenting_on(self):
        scorer = EmbeddingScorer(StubEmbedder(window=64))

        self.assertTrue(scorer.segmented)
        self.assertEqual(scorer.max_tokens, 64)

    def test_the_fake_embedder_declares_nothing_and_stays_whole(self):
        scorer = EmbeddingScorer(FakeEmbedder())

        self.assertFalse(scorer.segmented)
        self.assertIsNone(scorer.max_tokens)

    def test_the_switch_turns_it_off_even_with_a_window(self):
        scorer = EmbeddingScorer(StubEmbedder(window=64), segment=False)

        self.assertFalse(scorer.segmented)


class SegmentsPerChunkTest(unittest.TestCase):
    def setUp(self):
        self.chunk = make_chunk('def do_work():\n    """干活。"""\n' + "    x = 1\n" * 30)
        self.window = 40

    def texts(self) -> tuple[str, ...]:
        return segment_chunk(self.chunk, self.window)

    def test_a_long_chunk_becomes_several_vectors(self):
        embedder = StubEmbedder(window=self.window)
        scorer = EmbeddingScorer(embedder, chunks=(self.chunk,))

        scorer.prefetch()

        vectors = scorer._chunks[(self.chunk.id, self.chunk.digest)]
        self.assertEqual(len(vectors), len(self.texts()))
        self.assertGreater(len(vectors), 1)

    def test_the_segments_go_to_the_model_in_one_batch(self):
        embedder = StubEmbedder(window=self.window)
        scorer = EmbeddingScorer(embedder, chunks=(self.chunk,))

        scorer.prefetch()

        self.assertEqual(len(embedder.batches), 1)
        self.assertEqual(embedder.batches[0], self.texts())

    def test_a_short_chunk_stays_one_vector(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n')
        embedder = StubEmbedder(window=self.window)

        scorer = EmbeddingScorer(embedder, chunks=(chunk,))
        scorer.prefetch()

        self.assertEqual(len(scorer._chunks[(chunk.id, chunk.digest)]), 1)
        self.assertEqual(embedder.batches[0], (compact_text(chunk),))

    def test_the_lazy_path_also_produces_every_segment(self):
        embedder = StubEmbedder(window=self.window)
        scorer = EmbeddingScorer(embedder)  # 没有 chunks=：逐条算的老路

        vectors = scorer._chunk_vectors(self.chunk)

        self.assertEqual(len(vectors), len(self.texts()))


class MaxPoolingTest(unittest.TestCase):
    """打分取「最像的那一段」——这是分段要修的核心毛病，单独钉住。"""

    TAIL = (1.0, 0.0, 0.0)
    ELSE = (0.0, 1.0, 0.0)

    def setUp(self):
        # needle 只出现在正文靠后的一行：窗口装不下整块时它落在后面的段里。
        # 行数刻意控制在 compact_text 的正文上限（400 字符）以内——那是另一层
        # 截断（9.3 的 compact-v1），掺进来就测不清分段的池化了。
        body = "\n".join("    filler_%02d = %d" % (i, i) for i in range(10))
        self.chunk = make_chunk(
            'def do_work():\n    """干活。"""\n' + body + "\n    needle = 42\n"
        )
        self.window = 40

    def scorer(self, embedder) -> EmbeddingScorer:
        return EmbeddingScorer(embedder, chunks=(self.chunk,))

    def test_the_query_matches_a_later_segment(self):
        embedder = StubEmbedder(
            window=self.window,
            markers=(("needle", self.TAIL),),
            base=self.ELSE,
        )
        scorer = self.scorer(embedder)
        scorer.prefetch()

        matches = scorer(query("needle"), self.chunk)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].kind, "embedding")
        self.assertAlmostEqual(matches[0].score, scorer.weight, places=6)  # 相似度 1

    def test_the_first_segment_alone_would_not_match(self):
        embedder = StubEmbedder(
            window=self.window,
            markers=(("needle", self.TAIL),),
            base=self.ELSE,
        )
        scorer = self.scorer(embedder)
        scorer.prefetch()
        vectors = scorer._chunks[(self.chunk.id, self.chunk.digest)]

        texts = segment_chunk(self.chunk, self.window)
        # 前提：needle 真的落在切出来的某一段里，而且**不在第一段**——
        # 不然「取最像的那段」和「只看第一段」就是一回事，测不出池化
        self.assertIn("needle", compact_text(self.chunk))
        self.assertTrue(any("needle" in text for text in texts))
        self.assertNotIn("needle", texts[0])
        self.assertEqual(vectors[0], self.ELSE)
        self.assertIn(self.TAIL, vectors)

    def test_without_segmenting_the_same_chunk_misses(self):
        # 关掉分段就是 9.4 的行为：整段文本一次进模型，模型只会看开头，
        # 假 embedder 这里按「有没有 needle」判断，所以模拟不了截断——
        # 换成按「第一段才看得到」的假 embedder 太绕，这里只钉开关本身
        embedder = StubEmbedder(
            window=self.window,
            markers=(("needle", self.TAIL),),
            base=self.ELSE,
        )
        scorer = EmbeddingScorer(embedder, chunks=(self.chunk,), segment=False)

        scorer.prefetch()

        self.assertEqual(len(scorer._chunks[(self.chunk.id, self.chunk.digest)]), 1)


# ---------------------------------------------------------------------------
# 3. 缓存：key / 命名空间 / 复用
# ---------------------------------------------------------------------------


class CacheKeyTest(unittest.TestCase):
    def setUp(self):
        self.chunk = make_chunk('def do_work():\n    """干活。"""\n' + "    x = 1\n" * 30)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name).resolve()

    def test_segmented_keys_carry_the_segment_number(self):
        scorer = EmbeddingScorer(StubEmbedder(window=40))

        self.assertEqual(
            scorer._cache_key(self.chunk, 0), f"{chunk_key(self.chunk)}#s0"
        )
        self.assertEqual(
            scorer._cache_key(self.chunk, 3), f"{chunk_key(self.chunk)}#s3"
        )

    def test_the_old_shape_is_kept_when_not_segmenting(self):
        scorer = EmbeddingScorer(StubEmbedder())

        self.assertEqual(scorer._cache_key(self.chunk, 0), chunk_key(self.chunk))

    def test_the_namespace_carries_the_segment_strategy(self):
        embedder = StubEmbedder(window=128)

        cache = cache_for(embedder, self.directory)

        self.assertIn(SEGMENT_STRATEGY, cache.namespace)
        self.assertIn(namespace_of(embedder), cache.namespace)

    def test_turning_the_switch_off_changes_the_namespace(self):
        embedder = StubEmbedder(window=128)

        cut = cache_for(embedder, self.directory)
        whole = cache_for(embedder, self.directory, segmented=False)

        self.assertNotEqual(cut.namespace, whole.namespace)
        self.assertNotEqual(cut.path, whole.path)
        self.assertNotIn(SEGMENT_STRATEGY, whole.namespace)

    def test_an_embedder_without_a_window_keeps_the_old_namespace(self):
        embedder = FakeEmbedder((("校验",),))

        cache = cache_for(embedder, self.directory)

        self.assertEqual(
            cache.namespace, f"{namespace_of(embedder)}#{TEXT_STRATEGY}"
        )
        self.assertEqual(
            cache.path, cache_file(self.directory, cache.namespace)
        )

    def test_a_second_run_reuses_the_segment_vectors(self):
        chunk = self.chunk
        cache = cache_for(StubEmbedder(window=40), self.directory)
        first = StubEmbedder(window=40)
        scorer = EmbeddingScorer(first, cache=cache, chunks=(chunk,))
        scorer.prefetch()
        cache.flush()

        second = StubEmbedder(window=40)
        again = EmbeddingScorer(
            second, cache=cache_for(second, self.directory), chunks=(chunk,)
        )
        embedded = again.prefetch()

        self.assertEqual(embedded, 0)
        self.assertEqual(second.batches, [])
        self.assertEqual(again.embed_calls, 0)
        self.assertGreater(again.cache_hits, 0)

    def test_a_new_digest_invalidates_every_segment(self):
        cache = cache_for(StubEmbedder(window=40), self.directory)
        embedder = StubEmbedder(window=40)
        scorer = EmbeddingScorer(embedder, cache=cache, chunks=(self.chunk,))
        scorer.prefetch()
        cache.flush()

        changed = make_chunk(self.chunk.code, digest="beefbeef")
        second = StubEmbedder(window=40)
        again = EmbeddingScorer(
            second, cache=cache_for(second, self.directory), chunks=(changed,)
        )
        embedded = again.prefetch()

        self.assertGreater(embedded, 0)
        self.assertEqual(again.cache_hits, 0)


# ---------------------------------------------------------------------------
# 4. 兼容与降级
# ---------------------------------------------------------------------------


class CompatibilityTest(unittest.TestCase):
    def test_the_scorer_without_a_window_behaves_like_9_4(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n')
        embedder = FakeEmbedder((("zebra",),))
        cache = VectorCache(Path(tempfile.gettempdir()) / "uac95-unused.json", "test")

        scorer = EmbeddingScorer(embedder, chunks=(chunk,), cache=cache)
        scorer.prefetch()

        self.assertEqual(scorer._cache_key(chunk, 0), chunk_key(chunk))
        self.assertEqual(len(scorer._chunks[(chunk.id, chunk.digest)]), 1)
        self.assertNotIn("分段", scorer.stats_line())

    def test_hybrid_scorers_passes_the_switch_through(self):
        embedder = StubEmbedder(window=40)

        cut = hybrid_scorers(embedder)[-1]
        whole = hybrid_scorers(embedder, segment=False)[-1]

        self.assertTrue(cut.segmented)
        self.assertFalse(whole.segmented)

    def test_the_stats_line_counts_segments(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n' + "    x = 1\n" * 30)
        scorer = EmbeddingScorer(StubEmbedder(window=40), chunks=(chunk,))

        scorer.prefetch()

        line = scorer.stats_line()
        self.assertIn("分段", line)
        self.assertIn("窗口 40 token", line)
        self.assertEqual(scorer.segment_total, len(segment_chunk(chunk, 40)))
        self.assertEqual(scorer.segmented_chunks, 1)

    def test_combined_stats_line_adds_the_corpora_up(self):
        one = EmbeddingScorer(StubEmbedder(window=40), chunks=(make_chunk("a = 1"),))
        two = EmbeddingScorer(StubEmbedder(window=40), chunks=(make_chunk("b = 2"),))
        one.prefetch()
        two.prefetch()

        line = combined_stats_line((one, two))

        self.assertIn(f"嵌入 {one.texts_embedded + two.texts_embedded} 条文本", line)
        self.assertIn(f"分段 {one.segment_total + two.segment_total} 段", line)

    def test_the_search_result_is_a_match_with_the_same_shape(self):
        chunk = make_chunk(
            'def zebra_sighting():\n    """记一笔斑马目击。"""\n    return 1\n',
            symbol="zebra_sighting",
        )
        embedder = StubEmbedder(
            window=40, markers=(("zebra", (1.0, 0.0, 0.0)),)
        )
        retriever = Retriever(
            index=self.fake_index(chunk), scorers=hybrid_scorers(embedder, chunks=(chunk,))
        )

        found = retriever.search("zebra")

        self.assertEqual(found[0].chunk.id, chunk.id)
        self.assertIn("embedding", {match.kind for match in found[0].matches})

    @staticmethod
    def fake_index(chunk: CodeChunk) -> object:
        """一个假的 ChunkIndex：retriever 只用到 built / chunks()。"""

        class FakeIndex:
            built = True

            def chunks(self):
                return (chunk,)

        return FakeIndex()


class DegradingTest(unittest.TestCase):
    def test_a_broken_embedder_falls_back_to_the_literal_scorer(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n')

        class Broken(StubEmbedder):
            def embed(self, texts):
                raise ToolError("模型加载失败")

        guard = DegradingScorer(EmbeddingScorer(Broken(window=40)))

        matches = guard(query("do_work"), chunk)

        self.assertEqual(matches, [])
        self.assertTrue(guard.disabled)
        self.assertIn("模型加载失败", guard.error)

    def test_guarded_retrieval_still_returns_literal_hits(self):
        chunk = make_chunk('def do_work():\n    """干活。"""\n')

        class Broken(StubEmbedder):
            def embed(self, texts):
                raise ToolError("模型加载失败")

        retriever = Retriever(
            index=CompatibilityTest.fake_index(chunk),
            scorers=hybrid_scorers(Broken(window=40), chunks=(chunk,), guard=True),
        )

        found = retriever.search("do_work")

        self.assertEqual([item.chunk.id for item in found], [chunk.id])
        self.assertNotIn("embedding", {match.kind for match in found[0].matches})


class FixtureCorpusTest(unittest.TestCase):
    """C++ 评估语料：真实存在、能检索，且不在生产索引里。"""

    ROOT = Path(__file__).resolve().parent.parent

    def test_the_ignore_rule_keeps_fixtures_out_of_the_project_corpus(self):
        from tools.tree import iter_files

        files = list(iter_files(self.ROOT))

        self.assertFalse(
            [path for path in files if "fixtures" in path.parts],
            "fixtures/ 不该进生产索引",
        )

    def test_the_fixture_index_has_the_ring_buffer(self):
        index = ChunkIndex(root=self.ROOT / "tests" / "fixtures" / "cpp")
        index.build()

        symbols = {chunk.symbol for chunk in index.chunks()}
        self.assertIn("RingBuffer.push", symbols)
        self.assertIn("RingBuffer.pop", symbols)
        self.assertTrue(any(chunk.path == "ring_buffer.h" for chunk in index.chunks()))


if __name__ == "__main__":
    unittest.main()
