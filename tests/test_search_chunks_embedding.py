"""SearchChunksTool 的 embedding 接入测试（9.4）：模式、懒加载、降级、缓存。

六块分开钉：

- **模式**：默认 off——scorers 就是默认那三个字面打分器，结果和 8.8 一字
  不差；fake / local / multilingual 可选；坏名字报 ToolError 并列出可选项。
- **懒加载**：构造工具、注册工具都不碰模型；第一次 `execute()` 才加载，
  而且只加载一次。
- **语义生效**：字面一个词都对不上的查询，开了 embedding 能捞到目标块；
  输出格式还是那五段，一个字没改。
- **降级**：模型装不上 / 搜索那一步炸 / 缓存文件坏了，都不许让工具失败
  ——自动退回字面检索，错因记在 `embedding_error` 里。
- **缓存**：换个工具（新进程）用同一个缓存目录，一条都不重嵌；文件改了
  只重嵌那个文件的块（digest 是文件级的）。
- **配置**：环境变量选模式，默认 off；坏值启动即报错。

全部离线：假模型只有 `encode` + `get_sentence_embedding_dimension` 两个
方法；缓存全落在临时目录，不碰用户目录。查询都是另编的——本文件也在
语料里（项目把自己也索引了），不许把评估 case 的查询原样写进来。
"""

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from config import (
    SEARCH_EMBEDDING_ENV,
    SEARCH_EMBEDDING_MODES,
    ConfigError,
    load_search_embedding,
)
from index import (
    DEFAULT_SCORERS,
    FUNCTION,
    TEXT_STRATEGY,
    CodeChunk,
    DegradingScorer,
    EmbeddingScorer,
    EmbeddingUnavailable,
    FakeEmbedder,
    LocalEmbedder,
    Query,
    Retriever,
    cache_file,
    hybrid_scorers,
    namespace_of,
)
from index.providers import PROVIDERS
from tools import SearchChunksTool, ToolError

STORE = '''\
"""存东西的小库。"""


def save_row():
    """保存一条记录。"""
    return 1


def verify_row():
    """复核明细。"""
    return 2
'''

SPARE = '''\
"""备用的小库。"""


def spare_one():
    """和这条线没关系。"""
    return 3
'''

# 假 embedding 的小抄：只认「点验 / 复核」一组。查询用的词（点验）刻意
# 不出现在代码里，代码里的词（复核）刻意不出现在查询里——两头都对不上
# 字面，能命中就只可能来自语义分。
SEMANTIC_ALIASES = (("点验", "复核"),)
SEMANTIC_QUERY = "点验"


def a_chunk(symbol: str = "f") -> CodeChunk:
    """一个最小可用的 chunk，给 scorer 层的单元测试用。"""
    return CodeChunk(
        path="p.py",
        symbol=symbol,
        kind=FUNCTION,
        language="python",
        start_line=1,
        end_line=2,
        digest="d",
        code="def f():\n    return 1",
    )


class StubModel:
    """装成 sentence-transformers 的假模型：记批次、可控地报错。

    `explode_on` 给了就表示「文本里出现这个词就抛异常」——用来制造
    「预热没事、查询那一步炸」的场景。
    """

    dimension = 3

    def __init__(self, explode_on: str | None = None) -> None:
        self.explode_on = explode_on
        self.batches: list[tuple[str, ...]] = []

    def encode(self, texts, batch_size: int = 32):
        batch = tuple(texts)
        self.batches.append(batch)
        if self.explode_on and any(self.explode_on in text for text in batch):
            raise RuntimeError("模型炸了")
        return [[1.0 if "记录" in text else 0.0, 1.0, 0.0] for text in batch]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class StubLoader:
    """数一数模型被加载了几次；给了 error 就每次都失败。"""

    def __init__(self, model=None, error: Exception | None = None) -> None:
        self.model = model
        self.error = error
        self.calls = 0

    def __call__(self, name):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.model


class ProjectCase(unittest.TestCase):
    """一个临时小项目 + 一个独立的临时缓存目录。"""

    def setUp(self):
        project = tempfile.TemporaryDirectory()
        self.addCleanup(project.cleanup)
        self.root = Path(project.name).resolve()

        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache_dir = Path(cache.name).resolve()

        self.write("pkg/store.py", STORE)
        self.write("pkg/spare.py", SPARE)

    def write(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def make_tool(self, mode: str = "local", *, loader=None, model=None, **kwargs):
        """造一个开了 embedding 的工具，模型走注入口（离线、可数批次）。"""
        loader = loader or StubLoader(model or StubModel())
        embedder = LocalEmbedder("stub-model", model_loader=loader)
        tool = SearchChunksTool(
            root=self.root,
            embedding=mode,
            embedder=embedder,
            cache_dir=self.cache_dir,
            **kwargs,
        )
        return tool, loader


class OffModeTest(ProjectCase):
    """默认模式：8.8 起的行为，一个字节不差。"""

    def test_default_is_off_and_uses_the_literal_scorers(self):
        tool = SearchChunksTool(root=self.root)

        self.assertEqual(tool.mode, "off")
        self.assertEqual(tool.retriever.scorers, DEFAULT_SCORERS)
        self.assertIsNone(tool.embedder)
        self.assertIsNone(tool.scorer)
        self.assertIsNone(tool.embedding_error)

    def test_off_mode_gives_the_same_results_as_a_plain_retriever(self):
        tool = SearchChunksTool(root=self.root)

        self.assertEqual(
            tool.retriever.search("save_row"),
            Retriever(root=self.root).search("save_row"),
        )

    def test_off_mode_never_touches_the_cache_directory(self):
        never = self.cache_dir / "never"
        tool = SearchChunksTool(root=self.root, cache_dir=never)

        tool.execute("save_row")

        self.assertFalse(never.exists())

    def test_off_mode_rejects_an_injected_embedder(self):
        with self.assertRaises(ToolError):
            SearchChunksTool(root=self.root, embedder=FakeEmbedder())


class ModeTest(ProjectCase):
    """fake / local / multilingual 三种开着的模式。"""

    def test_an_unknown_mode_is_refused_with_the_options(self):
        with self.assertRaises(ToolError) as caught:
            SearchChunksTool(root=self.root, embedding="word2vec")

        message = str(caught.exception)
        self.assertIn("off", message)
        self.assertIn("multilingual", message)

    def test_the_mode_is_case_and_space_insensitive(self):
        tool = SearchChunksTool(root=self.root, embedding="  LOCAL ")

        self.assertEqual(tool.mode, "local")
        self.assertIsInstance(tool.embedder, LocalEmbedder)
        # 构造不加载模型：真模型 / 假 loader 在这条路上都一样
        self.assertFalse(tool.embedder.loaded)

    def test_fake_mode_wires_a_guarded_semantic_scorer(self):
        tool = SearchChunksTool(root=self.root, embedding="fake", cache=False)

        self.assertEqual(tool.mode, "fake")
        self.assertIsInstance(tool.embedder, FakeEmbedder)
        self.assertEqual(len(tool.retriever.scorers), len(DEFAULT_SCORERS) + 1)
        self.assertIsInstance(tool.guard, DegradingScorer)
        self.assertIsInstance(tool.scorer, EmbeddingScorer)
        self.assertIs(tool.guard.scorer, tool.scorer)
        self.assertFalse(tool.guard.disabled)

    def test_cache_can_be_turned_off(self):
        tool = SearchChunksTool(root=self.root, embedding="fake", cache=False)

        self.assertIsNone(tool.vector_cache)
        # 没有缓存也照常出结果
        self.assertIn("save_row", tool.execute("save_row"))

    def test_the_injected_embedder_is_used_as_is(self):
        embedder = FakeEmbedder(SEMANTIC_ALIASES)

        tool = SearchChunksTool(
            root=self.root, embedding="fake", embedder=embedder, cache=False
        )

        self.assertIs(tool.embedder, embedder)

    def test_every_provider_name_is_accepted(self):
        for name in PROVIDERS:
            with self.subTest(name=name):
                tool = SearchChunksTool(root=self.root, embedding=name, cache=False)
                self.assertEqual(tool.mode, name)
                self.assertIsNotNone(tool.guard)


class LazyLoadingTest(ProjectCase):
    """模型只在第一次真正检索时加载，而且只加载一次。"""

    def test_the_model_is_not_loaded_before_the_first_search(self):
        tool, loader = self.make_tool("local")

        self.assertEqual(loader.calls, 0)

        tool.execute("save_row")
        self.assertEqual(loader.calls, 1)

        tool.execute("spare_one")
        self.assertEqual(loader.calls, 1)

    def test_the_first_search_embeds_all_chunks_in_one_batch(self):
        tool, loader = self.make_tool("local")

        tool.execute("save_row")

        chunks = tool.index.chunks()
        self.assertEqual(len(chunks), 3)
        self.assertEqual(tool.scorer.embed_calls, 2)  # 1 次全量 + 1 条查询
        self.assertEqual(tool.scorer.texts_embedded, len(chunks) + 1)
        # 模型那边看：第一批就是全部 chunk（不是 N 个 chunk N 次调用）
        self.assertEqual(len(loader.model.batches), 2)
        self.assertEqual(len(loader.model.batches[0]), len(chunks))
        self.assertEqual(len(loader.model.batches[1]), 1)

    def test_a_second_search_embeds_only_the_new_query(self):
        tool, _ = self.make_tool("local")

        tool.execute("save_row")
        after_first = tool.scorer.texts_embedded

        tool.execute("spare_one")

        self.assertEqual(tool.scorer.embed_calls, 3)
        self.assertEqual(tool.scorer.texts_embedded, after_first + 1)


class SemanticEffectTest(ProjectCase):
    """开着 embedding 时，语义分真的把字面查不到的块捞出来。"""

    def make_semantic_tool(self):
        return SearchChunksTool(
            root=self.root,
            embedding="fake",
            embedder=FakeEmbedder(SEMANTIC_ALIASES),
            cache=False,
        )

    def test_a_query_with_no_literal_overlap_finds_the_chunk_by_meaning(self):
        output = self.make_semantic_tool().execute(SEMANTIC_QUERY)

        self.assertIn("verify_row", output)
        self.assertIn("语义相近", output)

    def test_the_literal_only_tool_finds_nothing_for_that_query(self):
        output = SearchChunksTool(root=self.root).execute(SEMANTIC_QUERY)

        self.assertTrue(output.startswith("（没有找到"))

    def test_the_output_format_is_unchanged(self):
        output = self.make_semantic_tool().execute(SEMANTIC_QUERY)
        line = output.splitlines()[0]

        # 五段：序号 + 位置、符号、[类型]、分数、命中原因——和字面结果形状
        # 一字不差，唯一区别是原因写的是「语义相近」（括号里照旧是分数）
        self.assertRegex(
            line,
            re.compile(
                r"^\d+\. \S+:\d+-\d+  \S+  \[[a-z_]+\]  [\d.]+  "
                r"语义相近 [\d.]+（\+[\d.]+）$"
            ),
        )
        self.assertEqual(len(line.split("  ")), 5)


class DegradeTest(ProjectCase):
    """embedding 出任何问题，工具都不许失败——退回字面。"""

    def literal_output(self, query: str) -> str:
        return SearchChunksTool(root=self.root).execute(query)

    def test_a_loader_failure_degrades_to_literal_results(self):
        loader = StubLoader(error=EmbeddingUnavailable("没装 sentence-transformers"))
        tool, _ = self.make_tool("local", loader=loader)

        output = tool.execute("save_row")

        self.assertEqual(output, self.literal_output("save_row"))
        self.assertIn("没装", tool.embedding_error)
        self.assertTrue(tool.guard.disabled)

    def test_a_degraded_tool_does_not_retry_the_model(self):
        loader = StubLoader(error=EmbeddingUnavailable("下不动"))
        tool, _ = self.make_tool("local", loader=loader)

        tool.execute("save_row")
        tool.execute("spare_one")

        self.assertEqual(loader.calls, 1)

    def test_a_failure_during_the_search_degrades_too(self):
        model = StubModel(explode_on="爆炸")
        tool, _ = self.make_tool("local", model=model)

        # 预热没事（块文本里没有那个词），第一条查询也正常
        self.assertIn("save_row", tool.execute("save_row"))
        self.assertIsNone(tool.embedding_error)

        # 查询里带那个词：查询向量那一步炸 → 降级，结果退回字面
        output = tool.execute("爆炸")
        self.assertEqual(output, self.literal_output("爆炸"))
        self.assertIn("模型炸了", tool.embedding_error)

        # 降级之后不再惊动模型
        before = len(model.batches)
        tool.execute("save_row")
        self.assertEqual(len(model.batches), before)

    def test_a_broken_cache_file_does_not_break_the_search(self):
        tool, _ = self.make_tool("local")
        path = cache_file(
            self.cache_dir, f"{namespace_of(tool.embedder)}#{TEXT_STRATEGY}"
        )
        path.write_text("{ 这不是 JSON", encoding="utf-8")

        output = tool.execute("save_row")

        self.assertIn("save_row", output)
        self.assertTrue(tool.vector_cache.degraded)
        # 缓存读不了 = 没有缓存：重新嵌一遍，结果照旧
        self.assertGreater(tool.scorer.texts_embedded, 0)

    def test_a_scorer_that_breaks_is_swallowed_by_the_guard(self):
        calls = []

        def broken(query, chunk):
            calls.append(chunk.symbol)
            raise RuntimeError("打分器炸了")

        guard = DegradingScorer(broken)
        query = Query(text="x", stripped="x", tokens=("x",))
        chunk = a_chunk()

        self.assertEqual(guard(query, chunk), [])
        self.assertEqual(guard(query, chunk), [])
        self.assertEqual(calls, ["f"])  # 第二次不再问里面
        self.assertTrue(guard.disabled)
        self.assertEqual(guard.error, "RuntimeError: 打分器炸了")

    def test_disable_keeps_the_first_reason(self):
        guard = DegradingScorer(lambda query, chunk: [])

        guard.disable("第一个原因")
        guard.disable("第二个原因")

        self.assertEqual(guard.error, "第一个原因")


class CacheReuseTest(ProjectCase):
    """缓存跨工具、跨进程复用；文件改了什么失效。"""

    def test_a_second_tool_with_the_same_cache_embeds_nothing(self):
        first, _ = self.make_tool("local")
        first.execute("save_row")

        second, _ = self.make_tool("local")
        output = second.execute("save_row")

        self.assertEqual(second.scorer.embed_calls, 0)
        self.assertEqual(second.scorer.texts_embedded, 0)
        self.assertEqual(output, first.execute("save_row"))

    def test_a_changed_file_only_invalidates_its_own_chunks(self):
        first, _ = self.make_tool("local")
        first.execute("save_row")

        # 只改 spare.py（一个函数一块）：只有它的 digest 变了
        self.write("pkg/spare.py", SPARE.replace("和这条线没关系。", "换了一个说法。"))

        second, _ = self.make_tool("local")
        second.execute("save_row")

        self.assertEqual(second.scorer.texts_embedded, 1)

    def test_the_cache_file_lands_in_the_given_directory_with_the_right_namespace(self):
        tool, _ = self.make_tool("local")

        tool.execute("save_row")

        files = list(self.cache_dir.glob("vectors-*.json"))
        self.assertEqual(len(files), 1)
        expected = f"{namespace_of(tool.embedder)}#{TEXT_STRATEGY}"
        self.assertIn(expected, files[0].read_text(encoding="utf-8"))


class ConfigTest(unittest.TestCase):
    """环境变量 → 模式；默认 off；坏值启动即报错。"""

    def test_the_default_is_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SEARCH_EMBEDDING_ENV, None)
            self.assertEqual(load_search_embedding(), "off")

    def test_an_empty_value_is_off(self):
        with patch.dict(os.environ, {SEARCH_EMBEDDING_ENV: "   "}):
            self.assertEqual(load_search_embedding(), "off")

    def test_the_value_is_normalized(self):
        with patch.dict(os.environ, {SEARCH_EMBEDDING_ENV: " Multilingual "}):
            self.assertEqual(load_search_embedding(), "multilingual")

    def test_a_bad_value_is_a_config_error(self):
        with patch.dict(os.environ, {SEARCH_EMBEDDING_ENV: "word2vec"}):
            with self.assertRaises(ConfigError):
                load_search_embedding()

    def test_config_modes_match_the_providers(self):
        # 两处名单必须一致：往 PROVIDERS 加 provider 时，配置这边别漏
        self.assertEqual(set(SEARCH_EMBEDDING_MODES), {"off"} | set(PROVIDERS))


class RegistryWiringTest(unittest.TestCase):
    """main.build_registry：把环境变量里的模式装进工具，默认不加载模型。"""

    def find_tool(self, registry) -> SearchChunksTool:
        return next(t for t in registry.list() if isinstance(t, SearchChunksTool))

    def test_the_registry_defaults_to_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(SEARCH_EMBEDDING_ENV, None)
            tool = self.find_tool(main.build_registry())

        self.assertEqual(tool.mode, "off")

    def test_the_registry_follows_the_env_var(self):
        with patch.dict(os.environ, {SEARCH_EMBEDDING_ENV: "fake"}):
            tool = self.find_tool(main.build_registry())

        self.assertEqual(tool.mode, "fake")
        self.assertIsInstance(tool.embedder, FakeEmbedder)

    def test_a_bad_env_value_fails_at_startup(self):
        with patch.dict(os.environ, {SEARCH_EMBEDDING_ENV: "word2vec"}):
            with self.assertRaises(ConfigError):
                main.build_registry()


class ScorerPairingTest(ProjectCase):
    """hybrid_scorers(guard=...) 的形状：守卫在、顺序对、没开就是老样子。"""

    def test_guard_true_wraps_the_embedding_scorer(self):
        scorers = hybrid_scorers(FakeEmbedder(), guard=True)

        self.assertEqual(scorers[:-1], DEFAULT_SCORERS)
        self.assertIsInstance(scorers[-1], DegradingScorer)
        self.assertIsInstance(scorers[-1].scorer, EmbeddingScorer)

    def test_guard_false_is_the_pre_9_4_shape(self):
        scorers = hybrid_scorers(FakeEmbedder())

        self.assertEqual(scorers[:-1], DEFAULT_SCORERS)
        self.assertIsInstance(scorers[-1], EmbeddingScorer)


if __name__ == "__main__":
    unittest.main()
