"""Embedding 优化测试（9.3）：紧凑文本、批量调用、磁盘缓存。

三件事分开钉：

- **compact text**（index/compact.py）：位置行（path:symbol [kind]）在、
  描述段在、代码开头在；同输入同输出（缓存 key 依赖它稳定）；长文本截
  到常量上限；chunk 的 `code` 一个字不动。
- **批量**：给了 `chunks=` 来源时，N 个 chunk 只产生**一次** embed 调用
  （而不是 N 次）；批次里文本顺序和 chunk 顺序一一对应；查询照旧逐条、
  只算一次；不给来源就退回逐条的老路——行为一样，只是慢。
- **缓存**：第二次运行一条都不重嵌（含查询）；digest 变、模型 / provider
  变自动失效；文件坏了 / 写不进去只是「没有缓存」，检索结果一个字不改；
  baseline（不挂 embedding）完全不受影响。9.6 起 prefetch 还会顺手打扫：
  这份语料里已经查不到的旧 digest / 旧段号从文件里清掉（别的语料和查询
  向量一条不碰）。

全部离线：LocalEmbedder 用注入口塞假模型（假模型就 encode +
get_sentence_embedding_dimension 两个方法），不装库、不联网、不下模型。
缓存全部落在临时目录，不碰用户目录。

注意本文件也在语料里（项目把自己也索引了）：**不许把评估 case 的查询
原样写进来**——写进来那条 case 就会命中这个测试文件自己（8.4 / 8.5 都
踩过）。这里用的查询都是另编的。
"""

import json
import tempfile
import unittest
from pathlib import Path

from config import PROJECT_ROOT
from index import (
    DEFAULT_CORPUS,
    DEFAULT_SCORERS,
    MAX_CODE_CHARS,
    MAX_DOC_CHARS,
    TEXT_STRATEGY,
    ChunkIndex,
    CodeChunk,
    EmbeddingScorer,
    EvaluationCase,
    FakeEmbedder,
    FUNCTION,
    LocalEmbedder,
    Query,
    Retriever,
    VectorCache,
    cache_file,
    chunk_key,
    compact_text,
    default_cache_dir,
    hybrid_scorers,
    namespace_of,
    query_key,
)
from index.__main__ import run_reports, variants

STORE = '''\
"""存东西的小库。"""


def save_row():
    """保存一条记录。"""
    return 1


def verify_row():
    """核对记录有没有被改过。"""
    return 2


def unrelated():
    """和这条线没关系。"""
    return 3
'''

ROW_ALIASES = (
    ("核对", "校验", "verify"),
    ("改过", "被改", "changed"),
    ("记录", "row"),
)

# 拼出来的：别让评估 case 的原文出现在这个文件里（见模块头）。
ROW_QUERY = "记录" + "有没有被动过"
CASES = (EvaluationCase(query=ROW_QUERY, expected_symbol="verify_row"),)


def make_chunk(
    symbol: str,
    code: str,
    *,
    path: str = "pkg/store.py",
    kind: str = FUNCTION,
    language: str = "python",
    digest: str = "aaaa1111",
) -> CodeChunk:
    return CodeChunk(
        path=path,
        symbol=symbol,
        kind=kind,
        language=language,
        start_line=1,
        end_line=2,
        digest=digest,
        code=code,
    )


class RecordingEmbedder:
    """包一层，记下每一批收到的文本（数调用次数、看批次顺序用）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.batches: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed(self, texts):
        batch = tuple(texts)
        self.batches.append(batch)
        return self.inner.embed(batch)


class FakeModel:
    """装成 sentence-transformers 模型的样子：encode + 维度，够 LocalEmbedder 用。"""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.batches: list[tuple[str, ...]] = []

    def encode(self, texts, batch_size: int = 32):
        self.batches.append(tuple(texts))
        # 只认「记录」一个词：向量可预期，语义分也稳
        return [[1.0 if "记录" in text else 0.0, 1.0, 0.0] for text in texts]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class WarehouseTest(unittest.TestCase):
    """一个临时小项目 + 一个独立的临时缓存目录。"""

    def setUp(self):
        project = tempfile.TemporaryDirectory()
        self.addCleanup(project.cleanup)
        self.root = Path(project.name).resolve()

        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache_dir = Path(cache.name).resolve()

        self.write(STORE)

    def write(self, content: str) -> None:
        path = self.root / "pkg" / "store.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")

    def index(self) -> ChunkIndex:
        index = ChunkIndex(root=self.root)
        index.build()
        return index

    def scorer(self, embedder, cache: VectorCache | None = None, index=None):
        if index is None:
            index = self.index()
        return EmbeddingScorer(embedder, cache=cache, chunks=index.chunks)

    def cache_for(self, embedder, directory: Path | None = None) -> VectorCache:
        directory = self.cache_dir if directory is None else directory
        namespace = namespace_of(embedder)
        return VectorCache(cache_file(directory, namespace), namespace)

    def query(self, text: str = ROW_QUERY) -> Query:
        return Query(text=text, stripped=text.lower(), tokens=(text.lower(),))


# ---------------------------------------------------------------------------
# 1. compact text
# ---------------------------------------------------------------------------


class CompactTextTest(unittest.TestCase):
    def test_carries_the_location_line(self):
        chunk = make_chunk("verify_row", 'def verify_row():\n    """核对。"""\n')

        text = compact_text(chunk)

        self.assertIn("pkg/store.py:verify_row [function]", text)

    def test_carries_the_docstring_and_the_code_head(self):
        code = 'def verify_row():\n    """核对记录有没有被改过。"""\n    return 2\n'
        chunk = make_chunk("verify_row", code)

        text = compact_text(chunk)

        self.assertIn("核对记录有没有被改过。", text)
        self.assertIn("def verify_row():", text)
        self.assertIn("return 2", text)

    def test_is_deterministic(self):
        chunk = make_chunk("verify_row", 'def verify_row():\n    """核对。"""\n')
        twin = make_chunk("verify_row", 'def verify_row():\n    """核对。"""\n')

        self.assertEqual(compact_text(chunk), compact_text(chunk))
        self.assertEqual(compact_text(chunk), compact_text(twin))

    def test_does_not_touch_the_chunk_code(self):
        code = 'def verify_row():\n    """核对。"""\n    return 2\n'
        chunk = make_chunk("verify_row", code)

        compact_text(chunk)

        self.assertEqual(chunk.code, code)

    def test_clips_a_long_docstring(self):
        long_doc = "说明" * (MAX_DOC_CHARS)
        chunk = make_chunk("f", f'def f():\n    """{long_doc}"""\n    return 1\n')

        text = compact_text(chunk)

        self.assertLessEqual(len(text.splitlines()[1]), MAX_DOC_CHARS)

    def test_clips_a_long_code_head(self):
        code = "x = 1\n" + "y = 2\n" * 500
        chunk = make_chunk("f", code)

        text = compact_text(chunk)

        body = text.split("\n", 1)[1]
        self.assertLessEqual(len(body), MAX_CODE_CHARS)

    def test_flattens_the_docstring_whitespace(self):
        chunk = make_chunk("f", 'def f():\n    """第一行\n\n    第二行。"""\n')

        text = compact_text(chunk)

        self.assertIn("第一行 第二行。", text)

    def test_the_docstring_is_not_repeated_in_the_code_head(self):
        code = 'def verify_row():\n    """核对记录有没有被改过。"""\n    return 2\n'
        chunk = make_chunk("verify_row", code)

        text = compact_text(chunk)

        # 描述段一次、代码开头里掏空，不再抄第二遍
        self.assertEqual(text.count("核对记录有没有被改过。"), 1)
        self.assertIn("def verify_row():", text)

    def test_the_removed_docstring_leaves_no_indentation_shell(self):
        code = 'def f():\n    """说明。"""\n    return 1\n'
        chunk = make_chunk("f", code)

        text = compact_text(chunk)

        for line in text.splitlines():
            self.assertEqual(line, line.rstrip())

    def test_non_python_chunks_have_no_docstring_section(self):
        code = '/* 核对记录 */\nint verify_row(void) { return 2; }\n'
        chunk = make_chunk("verify_row", code, path="src/store.c", language="c")

        text = compact_text(chunk)

        lines = text.splitlines()
        self.assertEqual(lines[0], "src/store.c:verify_row [function]")
        self.assertIn("int verify_row(void)", text)
        # 注释只来自代码开头这一段，没有被当成描述另抄一份
        self.assertEqual(text.count("核对记录"), 1)


# ---------------------------------------------------------------------------
# 2. 批量 embedding
# ---------------------------------------------------------------------------


class BatchEmbedTest(WarehouseTest):
    def test_prefetch_embeds_all_chunks_in_one_call(self):
        index = self.index()
        recorder = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        scorer = self.scorer(recorder, index=index)

        scorer.prefetch()

        chunks = index.chunks()
        self.assertEqual(len(recorder.batches), 1)  # N 个 chunk，一次调用
        self.assertEqual(
            recorder.batches[0], tuple(compact_text(c) for c in chunks)
        )
        self.assertEqual(scorer.embed_calls, 1)
        self.assertEqual(scorer.texts_embedded, len(chunks))

    def test_the_batch_keeps_the_chunk_order(self):
        index = self.index()
        recorder = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        scorer = self.scorer(recorder, index=index)

        scorer.prefetch()

        expected = [compact_text(c) for c in index.chunks()]
        for chunk, text in zip(index.chunks(), expected):
            self.assertIn(f"pkg/store.py:{chunk.symbol}", text)
        self.assertEqual(list(recorder.batches[0]), expected)

    def test_a_search_does_not_embed_chunks_one_by_one(self):
        index = self.index()
        recorder = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        retriever = Retriever(
            index=index,
            scorers=hybrid_scorers(recorder, chunks=index.chunks),
        )

        retriever.search("记录")

        big = [batch for batch in recorder.batches if len(batch) > 1]
        self.assertEqual(len(big), 1)  # 只有预热那一次是批量的
        self.assertEqual(len(big[0]), len(index.chunks()))

    def test_a_query_is_embedded_once_and_alone(self):
        index = self.index()
        recorder = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        retriever = Retriever(
            index=index,
            scorers=hybrid_scorers(recorder, chunks=index.chunks),
        )

        retriever.search("记录")
        retriever.search("记录")

        singles = [batch for batch in recorder.batches if len(batch) == 1]
        self.assertEqual(singles, [("记录",)])  # 同一条查询只嵌一次

    def test_without_a_chunk_source_it_falls_back_to_single_calls(self):
        index = self.index()
        recorder = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        scorer = EmbeddingScorer(recorder)  # 不给 chunks：老路

        result = scorer(self.query(), index.chunks()[0])

        self.assertTrue(result or result == [])
        # 查询一条、chunk 一条：两批各一条，没有批量调用
        self.assertEqual([len(batch) for batch in recorder.batches], [1, 1])

    def test_prefetching_twice_does_not_embed_twice(self):
        index = self.index()
        recorder = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        scorer = self.scorer(recorder, index=index)

        scorer.prefetch()
        scorer.prefetch()

        self.assertEqual(len(recorder.batches), 1)

    def test_a_local_embedder_prefetches_through_one_encode(self):
        index = self.index()
        model = FakeModel()
        embedder = LocalEmbedder(model_loader=lambda name: model)
        scorer = self.scorer(embedder, index=index)

        scorer.prefetch()

        self.assertEqual(len(model.batches), 1)
        self.assertEqual(len(model.batches[0]), len(index.chunks()))


# ---------------------------------------------------------------------------
# 3. 缓存：单件行为
# ---------------------------------------------------------------------------


class VectorCacheUnitTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name).resolve()

    def cache(self, namespace: str = "Fake:v1") -> VectorCache:
        return VectorCache(cache_file(self.dir, namespace), namespace)

    def test_chunk_key_carries_id_and_digest(self):
        chunk = make_chunk("f", "def f(): pass", digest="d1")

        key = chunk_key(chunk)

        self.assertIn(chunk.id, key)
        self.assertIn("d1", key)

    def test_the_query_key_is_prefixed(self):
        self.assertEqual(query_key("记录"), "query:记录")

    def test_a_missing_file_loads_as_empty(self):
        cache = self.cache()

        self.assertEqual(cache.get("nope"), None)
        self.assertFalse(cache.degraded)

    def test_put_then_get_round_trips(self):
        cache = self.cache()
        cache.put_many({"k": (0.5, 0.25)})

        self.assertEqual(self.cache().get("k"), (0.5, 0.25))

    def test_a_corrupt_file_degrades_to_empty(self):
        path = cache_file(self.dir, "Fake:v1")
        path.write_bytes(b"{ not json at all")

        cache = self.cache()

        self.assertIsNone(cache.get("k"))
        self.assertTrue(cache.degraded)
        self.assertIsNotNone(cache.load_error)

    def test_a_foreign_namespace_is_not_read_and_not_overwritten(self):
        # 挂在眼前的文件写着别人的命名空间：当空缓存，也不许覆盖（真发生
        # 这种事的场景是文件被手改过 / 哈希撞车 / 格式改了名没改）
        path = cache_file(self.dir, "Fake:v2")
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "namespace": "Fake:v1",
                    "dimension": 2,
                    "vectors": {"k": [1.0, 0.0]},
                }
            ),
            encoding="utf-8",
        )

        cache = self.cache("Fake:v2")
        cache.put_many({"k": (0.0, 1.0)})

        self.assertIsNone(cache.get("k"))
        self.assertIsNotNone(cache.write_error)
        kept = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(kept["namespace"], "Fake:v1")  # 没被覆盖
        self.assertEqual(kept["vectors"]["k"], [1.0, 0.0])

    def test_one_bad_entry_does_not_sink_the_rest(self):
        path = cache_file(self.dir, "Fake:v1")
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "namespace": "Fake:v1",
                    "dimension": 2,
                    "vectors": {"good": [1.0, 0.0], "bad": ["x", "y"], "short": [1.0]},
                }
            ),
            encoding="utf-8",
        )

        cache = self.cache()

        self.assertEqual(cache.get("good"), (1.0, 0.0))
        self.assertIsNone(cache.get("bad"))
        self.assertIsNone(cache.get("short"))  # 长度对不上也丢

    def test_a_failed_write_does_not_raise(self):
        blocker = self.dir / "not-a-dir"
        blocker.write_text("x", encoding="utf-8")
        cache = VectorCache(blocker / "vectors.json", "Fake:v1")

        cache.put_many({"k": (1.0, 0.0)})  # 不炸

        self.assertIsNotNone(cache.write_error)

    def test_deferred_flush_writes_nothing_until_asked(self):
        cache = self.cache()
        cache.put_many({"k": (1.0, 0.0)}, flush=False)

        path = cache_file(self.dir, "Fake:v1")
        self.assertFalse(path.exists())

        cache.flush()

        self.assertTrue(path.exists())
        self.assertEqual(self.cache().get("k"), (1.0, 0.0))

    def test_no_scratch_file_is_left_behind(self):
        cache = self.cache()
        cache.put_many({"k": (1.0, 0.0)})

        leftovers = list(self.dir.glob("*.tmp"))
        self.assertEqual(leftovers, [])


class CachePruneTest(unittest.TestCase):
    """缓存清理（9.6）：旧 digest / 旧段号删掉，查询和别的语料不碰。"""

    OLD = "pkg/a.py:f:1-2|d1"
    NEW = "pkg/a.py:f:1-2|d2#s0"
    EXTRA = "pkg/a.py:f:1-2|d2#s1"
    OTHER = "other/b.py:g:3-4|d9"
    QUERY = "query:怎么保存"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name).resolve()

    def cache(self, namespace: str = "Fake:v1") -> VectorCache:
        return VectorCache(cache_file(self.dir, namespace), namespace)

    def stored(self) -> dict:
        """磁盘上现在到底存了哪些 key（新开一份读，不看缓存对象的内存）。"""
        path = cache_file(self.dir, "Fake:v1")
        return json.loads(path.read_text(encoding="utf-8"))["vectors"]

    def fill(self, cache: VectorCache) -> None:
        cache.put_many(
            {
                self.OLD: (1.0, 0.0),
                self.NEW: (0.0, 1.0),
                self.EXTRA: (1.0, 1.0),
                self.OTHER: (2.0, 0.0),
                self.QUERY: (0.5, 0.5),
            }
        )

    def test_prune_drops_stale_versions_only_inside_the_scope(self):
        cache = self.cache()
        self.fill(cache)

        dropped = cache.prune({self.NEW, self.EXTRA}, scope={"pkg/a.py:f:1-2"})

        self.assertEqual(dropped, 1)
        kept = self.stored()
        self.assertNotIn(self.OLD, kept)  # 旧 digest：清掉
        self.assertIn(self.NEW, kept)  # 现役段：都在
        self.assertIn(self.EXTRA, kept)
        self.assertIn(self.OTHER, kept)  # 别的语料：没碰
        self.assertIn(self.QUERY, kept)  # 查询向量：永远不动

    def test_without_a_scope_every_orphan_chunk_key_is_fair_game(self):
        cache = self.cache()
        self.fill(cache)

        dropped = cache.prune({self.NEW, self.EXTRA})

        self.assertEqual(dropped, 2)  # 旧 digest + 别的语料的孤儿
        self.assertEqual(
            set(self.stored()), {self.NEW, self.EXTRA, self.QUERY}
        )

    def test_an_empty_keep_with_no_scope_is_a_full_sweep(self):
        cache = self.cache()
        self.fill(cache)

        dropped = cache.prune(set())

        self.assertEqual(dropped, 4)
        self.assertEqual(set(self.stored()), {self.QUERY})  # 查询照旧留下

    def test_nothing_stale_means_the_file_is_left_alone(self):
        cache = self.cache()
        cache.put_many({self.NEW: (0.0, 1.0)})
        path = cache_file(self.dir, "Fake:v1")
        before = path.read_bytes()

        dropped = cache.prune({self.NEW}, scope={"pkg/a.py:f:1-2"})

        self.assertEqual(dropped, 0)
        self.assertEqual(path.read_bytes(), before)  # 没删就不写盘

    def test_a_foreign_file_is_never_pruned(self):
        path = cache_file(self.dir, "Fake:v2")
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "namespace": "Fake:v1",
                    "dimension": 2,
                    "vectors": {"k": [1.0, 0.0]},
                }
            ),
            encoding="utf-8",
        )
        cache = self.cache("Fake:v2")
        before = path.read_bytes()

        dropped = cache.prune(set())

        self.assertEqual(dropped, 0)
        self.assertEqual(path.read_bytes(), before)  # 没动人家的缓存

    def test_a_broken_file_degrades_instead_of_raising(self):
        path = cache_file(self.dir, "Fake:v1")
        path.write_bytes(b"{ not json")
        cache = self.cache()
        before = path.read_bytes()

        dropped = cache.prune(set())

        self.assertEqual(dropped, 0)
        self.assertTrue(cache.degraded)
        self.assertEqual(path.read_bytes(), before)


# ---------------------------------------------------------------------------
# 4. 缓存：scorer 的第二次运行
# ---------------------------------------------------------------------------


class CacheIntegrationTest(WarehouseTest):
    def test_a_repeat_run_does_not_call_the_model_for_chunks(self):
        index = self.index()
        first = FakeEmbedder(ROW_ALIASES)
        scorer = self.scorer(
            RecordingEmbedder(first), cache=self.cache_for(first), index=index
        )
        scorer.prefetch()
        scorer.flush_cache()

        second = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        repeat = self.scorer(
            second, cache=self.cache_for(FakeEmbedder(ROW_ALIASES)), index=index
        )
        repeat.prefetch()

        self.assertEqual(second.batches, [])  # 一条都没重嵌
        self.assertEqual(repeat.cache_hits, len(index.chunks()))
        self.assertEqual(repeat.embed_calls, 0)

    def test_queries_hit_the_cache_too(self):
        index = self.index()
        first = FakeEmbedder(ROW_ALIASES)
        scorer = self.scorer(
            RecordingEmbedder(first), cache=self.cache_for(first), index=index
        )
        scorer.prefetch()
        scorer(self.query(), index.chunks()[0])  # 嵌一次查询
        scorer.flush_cache()

        second = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        repeat = self.scorer(
            second, cache=self.cache_for(FakeEmbedder(ROW_ALIASES)), index=index
        )
        repeat.prefetch()
        repeat(self.query(), index.chunks()[0])

        self.assertEqual(second.batches, [])
        self.assertEqual(repeat.embed_calls, 0)
        self.assertEqual(repeat.cache_hits, len(index.chunks()) + 1)

    def test_a_digest_change_invalidates_the_old_vectors(self):
        index = self.index()
        first = FakeEmbedder(ROW_ALIASES)
        self.scorer(
            RecordingEmbedder(first), cache=self.cache_for(first), index=index
        ).prefetch()

        self.write(STORE.replace("return 1", "return 100"))  # 文件改了
        rebuilt = self.index()
        second = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        scorer = self.scorer(
            second, cache=self.cache_for(FakeEmbedder(ROW_ALIASES)), index=rebuilt
        )

        scorer.prefetch()

        self.assertEqual(len(second.batches), 1)  # 全部重嵌
        self.assertEqual(scorer.cache_hits, 0)
        self.assertEqual(scorer.texts_embedded, len(rebuilt.chunks()))

    def test_a_digest_change_prunes_the_stale_vectors_away(self):
        index = self.index()
        first = FakeEmbedder(ROW_ALIASES)
        self.scorer(
            RecordingEmbedder(first), cache=self.cache_for(first), index=index
        ).prefetch()

        self.write(STORE.replace("return 1", "return 100"))  # 文件改了
        rebuilt = self.index()
        scorer = self.scorer(
            RecordingEmbedder(FakeEmbedder(ROW_ALIASES)),
            cache=self.cache_for(FakeEmbedder(ROW_ALIASES)),
            index=rebuilt,
        )
        scorer.prefetch()

        path = cache_file(self.cache_dir, namespace_of(FakeEmbedder(ROW_ALIASES)))
        stored = set(json.loads(path.read_text(encoding="utf-8"))["vectors"])
        # 文件里只剩现役 key：旧 digest 的向量被顺手清掉了
        self.assertEqual(stored, {chunk_key(chunk) for chunk in rebuilt.chunks()})

    def test_a_third_run_still_hits_everything_after_a_prune(self):
        index = self.index()
        first = FakeEmbedder(ROW_ALIASES)
        self.scorer(
            RecordingEmbedder(first), cache=self.cache_for(first), index=index
        ).prefetch()

        self.write(STORE.replace("return 1", "return 100"))
        rebuilt = self.index()
        second = FakeEmbedder(ROW_ALIASES)
        self.scorer(
            RecordingEmbedder(second), cache=self.cache_for(second), index=rebuilt
        ).prefetch()

        third = RecordingEmbedder(FakeEmbedder(ROW_ALIASES))
        repeat = self.scorer(
            third, cache=self.cache_for(FakeEmbedder(ROW_ALIASES)), index=rebuilt
        )
        repeat.prefetch()

        self.assertEqual(third.batches, [])  # 清理没有伤到现役向量
        self.assertEqual(repeat.cache_hits, len(rebuilt.chunks()))
        self.assertEqual(repeat.embed_calls, 0)

    def test_a_model_change_starts_from_scratch(self):
        index = self.index()
        first = FakeEmbedder(ROW_ALIASES)
        self.scorer(
            RecordingEmbedder(first), cache=self.cache_for(first), index=index
        ).prefetch()

        other = RecordingEmbedder(FakeEmbedder((("保存",),)))
        scorer = self.scorer(
            other, cache=self.cache_for(FakeEmbedder((("保存",),))), index=index
        )

        scorer.prefetch()

        self.assertEqual(len(other.batches), 1)
        self.assertEqual(scorer.cache_hits, 0)
        # 两套向量落在两个文件里，谁也没动谁
        self.assertEqual(len(list(self.cache_dir.glob("vectors-*.json"))), 2)

    def test_a_model_change_inside_the_same_class_also_separates(self):
        first = LocalEmbedder("model-a")
        second = LocalEmbedder("model-b")

        self.assertNotEqual(namespace_of(first), namespace_of(second))
        self.assertNotEqual(
            cache_file(self.cache_dir, namespace_of(first)),
            cache_file(self.cache_dir, namespace_of(second)),
        )

    def test_a_corrupt_cache_changes_nothing_about_the_results(self):
        index = self.index()
        plain = EmbeddingScorer(FakeEmbedder(ROW_ALIASES), chunks=index.chunks)
        expected = plain(self.query(), index.chunks()[1])

        cache = self.cache_for(FakeEmbedder(ROW_ALIASES))
        cache.path.write_bytes(b"{ broken")
        scorer = EmbeddingScorer(
            FakeEmbedder(ROW_ALIASES), cache=cache, chunks=index.chunks
        )
        scorer.prefetch()
        got = scorer(self.query(), index.chunks()[1])

        self.assertTrue(cache.degraded)
        self.assertEqual(
            [(m.kind, m.score) for m in got], [(m.kind, m.score) for m in expected]
        )

    def test_a_local_embedder_round_trips_through_the_cache(self):
        index = self.index()
        first_model = FakeModel()
        first = LocalEmbedder(model_loader=lambda name: first_model)
        scorer = self.scorer(first, cache=self.cache_for(first), index=index)
        scorer.prefetch()
        scorer(self.query(), index.chunks()[0])
        scorer.flush_cache()

        second_model = FakeModel()
        second = LocalEmbedder(model_loader=lambda name: second_model)
        repeat = self.scorer(second, cache=self.cache_for(second), index=index)
        repeat.prefetch()
        result = repeat(self.query(), index.chunks()[0])
        repeat.flush_cache()

        self.assertEqual(second_model.batches, [])
        self.assertEqual(repeat.embed_calls, 0)
        self.assertEqual(result, scorer(self.query(), index.chunks()[0]))

    def test_a_local_embedder_without_cache_is_still_correct(self):
        index = self.index()
        model = FakeModel()
        scorer = EmbeddingScorer(
            LocalEmbedder(model_loader=lambda name: model), chunks=index.chunks
        )

        scorer.prefetch()
        result = scorer(self.query(), index.chunks()[0])

        self.assertTrue(result)  # 「记录」在词里，语义分该给
        self.assertEqual(scorer.embed_calls, 2)  # 预热一次（批量）+ 查询一次


# ---------------------------------------------------------------------------
# 5. 兼容性：baseline 一个字不动
# ---------------------------------------------------------------------------


class CompatibilityTest(WarehouseTest):
    def test_the_baseline_retriever_is_untouched_by_hybrid_runs(self):
        index = self.index()
        plain = Retriever(index=index)

        before = plain.search("verify_row")

        embedder = FakeEmbedder(ROW_ALIASES)
        hybrid = Retriever(
            index=index,
            scorers=hybrid_scorers(
                RecordingEmbedder(embedder),
                chunks=index.chunks,
                cache=self.cache_for(embedder),
            ),
        )
        hybrid.search("verify_row")
        plain.search("verify_row")  # 同一条 baseline 再跑一遍

        after = plain.search("verify_row")
        self.assertEqual(
            [(r.chunk.id, r.score) for r in before], [(r.chunk.id, r.score) for r in after]
        )
        self.assertEqual(plain.scorers, DEFAULT_SCORERS)

    def test_hybrid_scorers_without_the_new_options_behaves_like_before(self):
        embedder = FakeEmbedder(ROW_ALIASES)

        scorers = hybrid_scorers(embedder)

        self.assertEqual(scorers[: len(DEFAULT_SCORERS)], DEFAULT_SCORERS)
        scorer = scorers[-1]
        self.assertIsInstance(scorer, EmbeddingScorer)
        self.assertIs(scorer.embedder, embedder)
        self.assertIsNone(scorer.cache)

    def test_a_warm_cache_run_matches_a_plain_run(self):
        index = self.index()
        plain = self.scorer(FakeEmbedder(ROW_ALIASES), index=index)
        expected = plain(self.query(), index.chunks()[1])

        warm = self.scorer(FakeEmbedder(ROW_ALIASES), cache=self.cache_for(plain.embedder), index=index)
        warm.prefetch()
        warm.flush_cache()
        second = self.scorer(
            FakeEmbedder(ROW_ALIASES), cache=self.cache_for(plain.embedder), index=index
        )
        second.prefetch()
        got = second(self.query(), index.chunks()[1])

        self.assertEqual([m.score for m in got], [m.score for m in expected])

    def test_the_default_cache_dir_is_outside_the_project(self):
        inside = default_cache_dir().resolve()
        project = Path(PROJECT_ROOT).resolve()

        self.assertNotEqual(inside, project)
        self.assertNotIn(project, inside.parents)


# ---------------------------------------------------------------------------
# 6. 评估入口的接线（--hybrid 组走加速管线）
# ---------------------------------------------------------------------------


class EntryWiringTest(WarehouseTest):
    def test_groups_get_the_batch_and_cache_wiring(self):
        index = self.index()

        # 9.5 起一组是「语料名 → retriever」；这里只有默认语料那一份
        plain = variants(index, ["--hybrid"])[1][1][DEFAULT_CORPUS]
        cached = variants(index, ["--hybrid"], cache_dir=self.cache_dir)[1][1][
            DEFAULT_CORPUS
        ]

        self.assertIsNone(plain.scorers[-1].cache)
        self.assertIsNotNone(cached.scorers[-1].cache)
        self.assertTrue(callable(cached.scorers[-1]._chunk_source))

    def test_the_cache_namespace_carries_the_text_strategy(self):
        index = self.index()

        cached = variants(index, ["--hybrid"], cache_dir=self.cache_dir)[1][1][
            DEFAULT_CORPUS
        ]

        # 拼法改了（TEXT_STRATEGY 加一版）就换文件：旧向量不会冒充新结果
        self.assertIn(TEXT_STRATEGY, cached.scorers[-1].cache.namespace)

    def test_the_stats_line_reports_the_counters(self):
        lines: list[str] = []
        code = run_reports(
            CASES, self.index(), ["--hybrid"], lines.append, cache_dir=self.cache_dir
        )
        text = "\n".join(lines)

        self.assertEqual(code, 0)
        self.assertEqual(text.count("[embedding]"), 1)
        self.assertIn("embed() 调用", text)
        self.assertIn("缓存命中", text)
        # 别撞上报告里的计数器（test_evaluation_entry 数这些）
        self.assertNotIn("Hit@1", text.split("[embedding]")[1])
        self.assertEqual(text.count("==="), 4)

    def test_a_repeat_entry_run_embeds_nothing(self):
        first: list[str] = []
        run_reports(
            CASES, self.index(), ["--hybrid"], first.append, cache_dir=self.cache_dir
        )
        second: list[str] = []
        run_reports(
            CASES, self.index(), ["--hybrid"], second.append, cache_dir=self.cache_dir
        )

        stats = next(line for line in second if line.startswith("[embedding]"))
        self.assertIn("embed() 调用 0 次", stats)
        self.assertNotIn("缓存命中 0 条", stats)

    def test_without_a_cache_dir_there_is_no_cache_line(self):
        lines: list[str] = []
        run_reports(CASES, self.index(), ["--hybrid"], lines.append)

        self.assertFalse(any(line.startswith("[cache]") for line in lines))

    def test_the_cache_line_names_the_directory(self):
        lines: list[str] = []
        run_reports(
            CASES, self.index(), ["--hybrid"], lines.append, cache_dir=self.cache_dir
        )

        self.assertIn(f"[cache] 向量缓存目录：{self.cache_dir}", lines[0])


if __name__ == "__main__":
    unittest.main()
