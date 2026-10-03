"""Local Embedder 与 provider 工厂（9.0）。

全部离线：不装 sentence-transformers、不联网、不下载模型。

- **工厂**（`create_embedder`）：名字 → 对象；不认识的名字报错并列出
  可选项；`"local"` 只是构造、不加载模型（没装库的机器上也拿得到对象）。
- **LocalEmbedder 对模型的封装**：一个假模型（encode +
  get_sentence_embedding_dimension，就这两个方法）就能把行为全测了——
  批量进批量出、顺序对应、维度对得上、一批只调模型一次、模型只加载
  一次、空输入不惊动模型。9.6 起量维度优先用新 API
  `get_embedding_dimension()`（旧名字在新版 sentence-transformers 会发
  FutureWarning），假模型只有旧名字，正好把兜底那条路钉住。
- **加载失败**：没装库、模型加载炸了，都包成 `EmbeddingUnavailable`
  且信息可行动；失败不粘住，下一次还会再试。
- **接回现有管线**：挂着假模型的 LocalEmbedder 塞进 `hybrid_scorers`、
  在真 `Retriever` 上跑一句查询——Retriever 一行没为 9.0 改。
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from index import (
    DEFAULT_LOCAL_MODEL,
    DEMO_ALIASES,
    Embedder,
    EmbeddingUnavailable,
    FakeEmbedder,
    LocalEmbedder,
    Retriever,
    create_embedder,
    hybrid_scorers,
)
from index.local_embedding import load_sentence_transformer
from tools.base import ToolError


class FakeModel:
    """按模型协议做的假模型：就 encode + get_sentence_embedding_dimension。

    向量是「文本长度的重复」：同样的输入同样的向量，顺序对不对一测便知。
    每次 encode 把「收到哪几条、batch_size 是多少」记下来，调用方式由
    测试盯着。
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.encodes: list[tuple[list[str], int | None]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, batch_size=None):
        self.encodes.append((list(texts), batch_size))
        return [[float(len(text))] * self._dimension for text in texts]


class LocalEmbedderTest(unittest.TestCase):
    """封装逻辑：全走假模型，一个字节的 sentence-transformers 都不用碰。"""

    def setUp(self):
        self.model = FakeModel(dimension=4)
        self.loads: list[str] = []
        self.embedder = LocalEmbedder(model_loader=self.load)

    def load(self, name: str) -> FakeModel:
        self.loads.append(name)
        return self.model

    def test_it_is_an_embedder(self):
        self.assertIsInstance(self.embedder, Embedder)

    def test_batch_in_batch_out_in_order(self):
        vectors = self.embedder.embed(("aa", "bbbb", "c"))

        self.assertEqual(len(vectors), 3)
        self.assertEqual(vectors[0][0], 2.0)
        self.assertEqual(vectors[1][0], 4.0)
        self.assertEqual(vectors[2][0], 1.0)

    def test_vectors_are_plain_tuples_of_floats(self):
        vectors = self.embedder.embed(("ab",))

        self.assertIsInstance(vectors, tuple)
        self.assertIsInstance(vectors[0], tuple)
        self.assertIsInstance(vectors[0][0], float)

    def test_the_whole_batch_goes_to_the_model_in_one_call(self):
        self.embedder.embed(("a", "b", "c"))

        # 一批一次，不是逐条调——逐条是真模型时最贵的用法
        self.assertEqual(len(self.model.encodes), 1)
        self.assertEqual(self.model.encodes[0][0], ["a", "b", "c"])

    def test_dimension_comes_from_the_model(self):
        self.assertEqual(self.embedder.dimension, 4)
        self.assertEqual(len(self.embedder.embed(("abc",))[0]), 4)

    def test_a_custom_batch_size_reaches_the_model(self):
        LocalEmbedder(model_loader=self.load, batch_size=8).embed(("a",))

        self.assertEqual(self.model.encodes[0][1], 8)

    def test_the_model_is_loaded_once(self):
        self.embedder.embed(("a",))
        self.embedder.embed(("b",))

        self.assertEqual(self.loads, [DEFAULT_LOCAL_MODEL])

    def test_a_custom_model_name_is_passed_to_the_loader(self):
        LocalEmbedder("my-model", model_loader=self.load).embed(("a",))

        self.assertEqual(self.loads, ["my-model"])

    def test_empty_batch_returns_empty_without_touching_the_model(self):
        self.assertEqual(self.embedder.embed(()), ())
        self.assertEqual(self.model.encodes, [])
        self.assertFalse(self.embedder.loaded)

    def test_loaded_flips_after_the_first_embed(self):
        self.assertFalse(self.embedder.loaded)
        self.embedder.embed(("a",))
        self.assertTrue(self.embedder.loaded)

    def test_a_count_mismatch_is_caught(self):
        class ConfusedModel(FakeModel):
            def encode(self, texts, batch_size=None):
                return [[1.0]]  # 不管给几条，都只回一条

        embedder = LocalEmbedder(model_loader=lambda name: ConfusedModel())

        with self.assertRaises(ToolError) as caught:
            embedder.embed(("a", "b"))

        self.assertIn("2", str(caught.exception))
        self.assertIn("1", str(caught.exception))

    def test_a_bad_batch_size_is_refused(self):
        with self.assertRaises(ToolError) as caught:
            LocalEmbedder(model_loader=self.load, batch_size=0)

        self.assertIn("batch_size", str(caught.exception))


class DimensionApiTest(unittest.TestCase):
    """9.6：量维度优先新 API，旧名字兜底——旧名字在新版会发 FutureWarning。"""

    def test_the_new_method_wins_when_both_exist(self):
        calls: list[str] = []

        class Both:
            def get_embedding_dimension(self) -> int:
                calls.append("new")
                return 7

            def get_sentence_embedding_dimension(self) -> int:
                calls.append("old")
                return 3

        embedder = LocalEmbedder(model_loader=lambda name: Both())

        self.assertEqual(embedder.dimension, 7)
        self.assertEqual(calls, ["new"])  # 旧方法一次都没调，不会触警告

    def test_the_old_method_still_works_as_the_fallback(self):
        embedder = LocalEmbedder(model_loader=lambda name: FakeModel(dimension=4))

        self.assertEqual(embedder.dimension, 4)

    def test_a_new_only_model_works_end_to_end(self):
        class NewOnly:
            def get_embedding_dimension(self) -> int:
                return 5

            def encode(self, texts, batch_size=None):
                return [[0.5] * 5 for _ in texts]

        embedder = LocalEmbedder(model_loader=lambda name: NewOnly())

        self.assertEqual(embedder.dimension, 5)
        self.assertEqual(len(embedder.embed(("abc",))[0]), 5)

    def test_a_model_with_neither_api_is_refused(self):
        class Neither:
            def encode(self, texts, batch_size=None):
                return [[0.0] for _ in texts]

        embedder = LocalEmbedder(model_loader=lambda name: Neither())

        with self.assertRaises(ToolError) as caught:
            embedder.embed(("a",))

        message = str(caught.exception)
        self.assertIn("get_embedding_dimension", message)
        self.assertIn("get_sentence_embedding_dimension", message)


class LoadFailureTest(unittest.TestCase):
    """加载失败这条路：报错要能行动，而且失败不许粘住。"""

    def test_missing_library_says_how_to_install(self):
        # sys.modules 里放个 None：再 import 它就会 ImportError，
        # 和真没装的机器上走的是同一条路
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(EmbeddingUnavailable) as caught:
                load_sentence_transformer("any-model")

        self.assertIn("pip install sentence-transformers", str(caught.exception))

    def test_a_loader_failure_becomes_unavailable(self):
        def boom(name):
            raise RuntimeError("下载中断了")

        embedder = LocalEmbedder("m", model_loader=boom)

        with self.assertRaises(EmbeddingUnavailable) as caught:
            embedder.embed(("a",))

        message = str(caught.exception)
        self.assertIn("m", message)
        self.assertIn("下载中断了", message)

    def test_an_unavailable_from_a_custom_loader_passes_through(self):
        def unavailable(name):
            raise EmbeddingUnavailable("自定义 loader 自己说清楚了")

        embedder = LocalEmbedder(model_loader=unavailable)

        with self.assertRaises(EmbeddingUnavailable) as caught:
            embedder.embed(("a",))

        self.assertEqual(str(caught.exception), "自定义 loader 自己说清楚了")

    def test_a_failure_does_not_stick(self):
        tries: list[str] = []

        def flaky(name):
            tries.append(name)
            if len(tries) == 1:
                raise RuntimeError("这一次断了")
            return FakeModel()

        embedder = LocalEmbedder(model_loader=flaky)

        with self.assertRaises(EmbeddingUnavailable):
            embedder.embed(("a",))
        # 第二次再试：连上了就用它——失败不该粘住
        self.assertEqual(len(embedder.embed(("a",))), 1)
        self.assertEqual(len(tries), 2)


class FactoryTest(unittest.TestCase):
    """create_embedder：名字 → Embedder，统一调用方式。"""

    def test_the_default_provider_is_fake(self):
        embedder = create_embedder()

        self.assertIsInstance(embedder, FakeEmbedder)
        self.assertEqual(embedder.aliases, DEMO_ALIASES)
        self.assertEqual(embedder.dimension, len(DEMO_ALIASES))

    def test_fake_accepts_custom_aliases(self):
        embedder = create_embedder("fake", aliases=(("a", "b"),))

        self.assertEqual(embedder.dimension, 1)

    def test_local_is_built_without_loading_a_model(self):
        embedder = create_embedder("local")

        self.assertIsInstance(embedder, LocalEmbedder)
        self.assertEqual(embedder.model_name, DEFAULT_LOCAL_MODEL)
        # 构造完没加载：没装库的机器上这一步也不该报错
        self.assertFalse(embedder.loaded)

    def test_local_parameters_pass_through(self):
        model = FakeModel()

        embedder = create_embedder(
            "local", model_name="my-model", model_loader=lambda name: model, batch_size=8
        )
        embedder.embed(("a",))

        self.assertEqual(model.encodes[0][1], 8)

    def test_an_unknown_provider_is_refused_with_the_options(self):
        with self.assertRaises(ToolError) as caught:
            create_embedder("openai")

        message = str(caught.exception)
        self.assertIn("openai", message)
        self.assertIn("fake", message)
        self.assertIn("local", message)


class TagModel:
    """极简本地模型：向量只有「文本里有没有这个词」两个维度。

    用来证明 LocalEmbedder 包出来的东西 Retriever 直接能吃——维度和
    语义都不重要，接口对上就行。
    """

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, batch_size=None):
        return [[1.0, 0.0] if "zebra" in text else [0.0, 1.0] for text in texts]


class PipelineCompatTest(unittest.TestCase):
    """接回现有管线：hybrid_scorers / Retriever 一行没为 9.0 改过。"""

    ANIMALS = '''\
"""动物记录。"""


def zebra_sighting():
    """记一笔斑马目击。"""
    return 1


def fence_check():
    """看看围栏。"""
    return 2
'''

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        path = self.root / "pkg" / "animals.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.ANIMALS, encoding="utf-8", newline="")

    def test_a_local_embedder_plugs_into_hybrid_scorers(self):
        embedder = create_embedder("local", model_loader=lambda name: TagModel())
        retriever = Retriever(root=self.root, scorers=hybrid_scorers(embedder))
        retriever.index.build()

        results = retriever.search("zebra 目击", top_k=3)

        self.assertEqual(results[0].chunk.symbol, "zebra_sighting")
        self.assertIn("语义相近", results[0].reason)


if __name__ == "__main__":
    unittest.main()
