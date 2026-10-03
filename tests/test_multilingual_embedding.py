"""多语言 embedding provider 与第四组评估入口（9.2）。

三块：

- **模型配置选择**：`"multilingual"` 在工厂注册表里，默认模型是
  `DEFAULT_MULTILINGUAL_MODEL`，`model_name=` 能覆盖——模型名是配置，
  不是写死的业务逻辑。
- **参数传递**：`model_loader` / `batch_size` / `model_name` 一路传到模型，
  批量进批量出、顺序对应、维度来自模型。
- **不可用模型**：下载失败 / 模型名不存在都包成 `EmbeddingUnavailable`；
  多语言组跑不动时，评估入口里前三组的报告原样保留。
- **评估入口**：`--multilingual` 出四组，组是递进的（baseline / 假 embedding /
  英文本地模型 / 多语言模型）；`--local` 仍是三组；不给参数仍是纯
  baseline，Hit 数字一个没变。

全部离线：假模型只有 `encode` + `get_sentence_embedding_dimension` 两个
方法，不装库、不联网、不下载模型。
"""

import tempfile
import unittest
from pathlib import Path

from index import (
    DEFAULT_CORPUS,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MULTILINGUAL_MODEL,
    DEFAULT_SCORERS,
    DEMO_ALIASES,
    PROVIDERS,
    ChunkIndex,
    Embedder,
    EmbeddingScorer,
    EmbeddingUnavailable,
    EvaluationCase,
    FakeEmbedder,
    LocalEmbedder,
    create_embedder,
)
from index.__main__ import run_reports, variants
from tools.base import ToolError

STORE = '''\
"""存东西。"""


def zebra_sighting():
    """记一笔斑马目击。"""
    return 1


def fence_check():
    """看看围栏。"""
    return 2
'''

CASES = (
    EvaluationCase(query="zebra", expected_symbol="zebra_sighting"),
    EvaluationCase(query="fence", expected_symbol="fence_check"),
)


class FakeModel:
    """按模型协议做的假模型：就 encode + get_sentence_embedding_dimension。

    向量是「文本长度的重复」：同样的输入同样的向量，顺序对不对一测便知。
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.encodes: list[tuple[list[str], int | None]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts, batch_size=None):
        self.encodes.append((list(texts), batch_size))
        return [[float(len(text))] * self._dimension for text in texts]


class UnavailableEmbedder(Embedder):
    """一开口就说「跑不动」：模拟没装库 / 模型下不下来。"""

    @property
    def dimension(self) -> int:
        raise EmbeddingUnavailable("这台机器上没装 sentence-transformers")

    def embed(self, texts):
        raise EmbeddingUnavailable("这台机器上没装 sentence-transformers")


def retriever_of(group, name: str = DEFAULT_CORPUS):
    """一组里某个语料的 retriever（9.5 起一组是「语料名 → retriever」）。"""
    return group[1][name]


def embedder_of(group) -> Embedder:
    """拎出一组上挂着的 embedding（hybrid 组的最后一个 scorer）。"""
    return next(
        scorer.embedder
        for scorer in retriever_of(group).scorers
        if isinstance(scorer, EmbeddingScorer)
    )


class ProviderConfigTest(unittest.TestCase):
    """模型配置选择：名字 → 模型，默认值可覆盖。"""

    def test_multilingual_is_registered(self):
        self.assertIn("multilingual", PROVIDERS)

    def test_multilingual_defaults_to_the_multilingual_model(self):
        embedder = create_embedder("multilingual")

        self.assertIsInstance(embedder, LocalEmbedder)
        self.assertEqual(embedder.model_name, DEFAULT_MULTILINGUAL_MODEL)
        # 构造不加载：没装库的机器上这一步也不该报错
        self.assertFalse(embedder.loaded)

    def test_the_default_model_can_be_overridden(self):
        embedder = create_embedder("multilingual", model_name="my-model")

        self.assertEqual(embedder.model_name, "my-model")

    def test_local_and_multilingual_default_to_different_models(self):
        self.assertNotEqual(DEFAULT_LOCAL_MODEL, DEFAULT_MULTILINGUAL_MODEL)
        self.assertEqual(
            create_embedder("local").model_name, DEFAULT_LOCAL_MODEL
        )

    def test_an_unknown_provider_still_lists_all_three(self):
        with self.assertRaises(ToolError) as caught:
            create_embedder("openai")

        message = str(caught.exception)
        for name in ("fake", "local", "multilingual"):
            self.assertIn(name, message)


class ParameterPassingTest(unittest.TestCase):
    """LocalEmbedder 参数传递：配置怎么给，模型就怎么收到。"""

    def test_loader_gets_the_configured_model_name(self):
        loads: list[str] = []

        def load(name):
            loads.append(name)
            return FakeModel()

        embedder = create_embedder("multilingual", model_loader=load)
        embedder.embed(("a",))

        self.assertEqual(loads, [DEFAULT_MULTILINGUAL_MODEL])

    def test_batch_size_and_batch_shape_reach_the_model(self):
        model = FakeModel()
        embedder = create_embedder(
            "multilingual", model_loader=lambda name: model, batch_size=8
        )

        vectors = embedder.embed(("aa", "bbbb", "c"))

        self.assertEqual(len(model.encodes), 1)  # 一批一次
        self.assertEqual(model.encodes[0][0], ["aa", "bbbb", "c"])  # 顺序原样
        self.assertEqual(model.encodes[0][1], 8)
        self.assertEqual([v[0] for v in vectors], [2.0, 4.0, 1.0])

    def test_dimension_comes_from_the_model(self):
        embedder = create_embedder(
            "multilingual", model_loader=lambda name: FakeModel(dimension=6)
        )

        self.assertEqual(embedder.dimension, 6)
        self.assertEqual(len(embedder.embed(("abc",))[0]), 6)


class UnavailableModelTest(unittest.TestCase):
    """不可用模型：报错要可行动，前面的结果不许被拖下水。"""

    def test_a_failed_download_becomes_unavailable(self):
        def broken(name):
            raise RuntimeError("下载中断了")

        embedder = create_embedder("multilingual", model_loader=broken)

        with self.assertRaises(EmbeddingUnavailable) as caught:
            embedder.embed(("a",))

        message = str(caught.exception)
        self.assertIn(DEFAULT_MULTILINGUAL_MODEL, message)
        self.assertIn("下载中断了", message)

    def test_a_missing_model_name_becomes_unavailable(self):
        def not_found(name):
            raise OSError(f"no such model: {name}")

        embedder = create_embedder(
            "multilingual", model_name="no-such-model", model_loader=not_found
        )

        with self.assertRaises(EmbeddingUnavailable) as caught:
            embedder.embed(("a",))

        self.assertIn("no-such-model", str(caught.exception))


class EntryTestBase(unittest.TestCase):
    """一个小工程 + 两条能查到的 case，评估跑得又快又确定。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        path = self.root / "pkg" / "store.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(STORE, encoding="utf-8", newline="")
        self.index = ChunkIndex(root=self.root)
        self.index.build()

    def lines(self, argv, **kwargs) -> tuple[list[str], int]:
        lines: list[str] = []
        code = run_reports(CASES, self.index, argv, out=lines.append, **kwargs)
        return lines, code


class MultilingualEntryTest(EntryTestBase):
    """多语言评估入口：四组递进，注入口原样生效。"""

    def test_multilingual_runs_four_groups(self):
        groups = variants(self.index, ["--multilingual"])

        self.assertEqual(len(groups), 4)
        self.assertIn("baseline", groups[0][0])
        self.assertIn("假 embedding", groups[1][0])
        self.assertIn("英文本地模型", groups[2][0])
        self.assertIn("多语言模型", groups[3][0])
        self.assertEqual(
            embedder_of(groups[2]).model_name, DEFAULT_LOCAL_MODEL
        )
        self.assertEqual(
            embedder_of(groups[3]).model_name, DEFAULT_MULTILINGUAL_MODEL
        )

    def test_the_multilingual_group_uses_the_injected_embedder(self):
        fake = FakeEmbedder(DEMO_ALIASES)

        groups = variants(self.index, ["--multilingual"], multilingual_embedder=fake)

        self.assertIs(embedder_of(groups[3]), fake)

    def test_four_reports_with_an_injected_multilingual_model(self):
        lines, code = self.lines(
            ["--multilingual"],
            local_embedder=FakeEmbedder(DEMO_ALIASES),
            multilingual_embedder=create_embedder(
                "multilingual", model_loader=lambda name: FakeModel()
            ),
        )
        text = "\n".join(lines)

        self.assertEqual(code, 0)
        self.assertEqual(text.count("Hit@1"), 4)
        self.assertIn("multilingual hybrid", text)

    def test_an_unavailable_multilingual_model_keeps_the_first_three(self):
        lines, code = self.lines(
            ["--multilingual"],
            local_embedder=FakeEmbedder(DEMO_ALIASES),
            multilingual_embedder=UnavailableEmbedder(),
        )
        text = "\n".join(lines)

        self.assertEqual(code, 1)
        self.assertIn("本地模型不可用", text)
        self.assertEqual(text.count("Hit@1"), 3)  # 前三组照常打出来了


class ExistingGroupsTest(EntryTestBase):
    """原有分组与 baseline 数字：一个都没变。"""

    def test_local_still_runs_three_groups(self):
        groups = variants(self.index, ["--local"])

        self.assertEqual(len(groups), 3)
        self.assertNotIn("多语言", groups[2][0])

    def test_no_flags_still_runs_only_the_baseline(self):
        groups = variants(self.index, [])

        self.assertEqual(len(groups), 1)
        self.assertEqual(retriever_of(groups[0]).scorers, DEFAULT_SCORERS)

    def test_the_baseline_numbers_are_unchanged(self):
        lines, code = self.lines([])
        text = "\n".join(lines)

        self.assertEqual(code, 0)
        self.assertEqual(text.count("Hit@1"), 1)
        self.assertIn("Hit@1 2/2", text)  # 两条 case 都还在第 1 名命中


if __name__ == "__main__":
    unittest.main()
