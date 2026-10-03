"""三组评估入口的测试（9.1）。

`python -m index` 本身是薄壳（真项目索引 + 真打印）；这里钉的是它选组、
每组挂什么、报告怎么打的部分：

- 不给参数：只有 baseline 一组（行为和 8.5 一样）。
- `--hybrid`：两组，第二组挂着假 embedding。
- `--local`：三组，第三组挂着 LocalEmbedder——**构造不加载模型**，
  没装 sentence-transformers 的机器上这一步也不该报错。
- 每组一份报告，Hit@1/3/5 和耗时都在；本地模型跑不动时前面几组照常
  留在输出里，退出码说明有组没跑成。

全部离线：`--local` 那组用注入口塞假模型的 embedder，真 LocalEmbedder
只验证到「构造出来、没加载」为止；不联网、不装库、不下载模型。
"""

import tempfile
import unittest
from pathlib import Path

from index import (
    DEFAULT_CORPUS,
    DEFAULT_SCORERS,
    DEMO_ALIASES,
    ChunkIndex,
    Embedder,
    EmbeddingScorer,
    EmbeddingUnavailable,
    EvaluationCase,
    FakeEmbedder,
    LocalEmbedder,
)
from index.__main__ import run_reports, variants

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


class UnavailableEmbedder(Embedder):
    """一开口就说「跑不动」：模拟没装库 / 模型下不下来。"""

    @property
    def dimension(self) -> int:
        raise EmbeddingUnavailable("这台机器上没装 sentence-transformers")

    def embed(self, texts):
        raise EmbeddingUnavailable("这台机器上没装 sentence-transformers")


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


class VariantsTest(EntryTestBase):
    def test_no_flags_runs_only_the_baseline(self):
        groups = variants(self.index, [])

        self.assertEqual(len(groups), 1)
        self.assertIn("baseline", groups[0][0])
        self.assertEqual(retriever_of(groups[0]).scorers, DEFAULT_SCORERS)

    def test_hybrid_adds_the_fake_group(self):
        groups = variants(self.index, ["--hybrid"])

        self.assertEqual(len(groups), 2)
        embedder = embedder_of(groups[1])
        self.assertIsInstance(embedder, FakeEmbedder)
        self.assertEqual(embedder.aliases, DEMO_ALIASES)

    def test_local_adds_the_third_group_without_loading_a_model(self):
        groups = variants(self.index, ["--local"])

        self.assertEqual(len(groups), 3)
        self.assertIn("baseline", groups[0][0])
        self.assertIn("假 embedding", groups[1][0])
        self.assertIn("本地模型", groups[2][0])
        embedder = embedder_of(groups[2])
        self.assertIsInstance(embedder, LocalEmbedder)
        # 只造对象：没装 sentence-transformers 的机器上这一步也不该报错
        self.assertFalse(embedder.loaded)

    def test_the_local_group_uses_the_injected_embedder(self):
        fake = FakeEmbedder((("zebra",),))

        groups = variants(self.index, ["--local"], local_embedder=fake)

        self.assertIs(embedder_of(groups[2]), fake)


class ReportsTest(EntryTestBase):
    def test_one_report_per_group_with_the_metrics(self):
        lines, code = self.lines(["--hybrid"])
        text = "\n".join(lines)

        self.assertEqual(code, 0)
        self.assertEqual(text.count("Hit@1"), 2)  # 两组各一份摘要
        self.assertEqual(text.count("平均检索耗时"), 2)
        self.assertEqual(text.count("==="), 4)  # 两个标题，各带一对 ===

    def test_three_groups_all_report(self):
        lines, code = self.lines(
            ["--local"], local_embedder=FakeEmbedder(DEMO_ALIASES)
        )
        text = "\n".join(lines)

        self.assertEqual(code, 0)
        self.assertEqual(text.count("Hit@1"), 3)
        self.assertIn("local hybrid", text)

    def test_an_unavailable_local_model_leaves_the_earlier_reports(self):
        lines, code = self.lines(["--local"], local_embedder=UnavailableEmbedder())
        text = "\n".join(lines)

        self.assertEqual(code, 1)
        self.assertIn("本地模型不可用", text)
        self.assertEqual(text.count("Hit@1"), 2)  # 前两组照常打出来了


if __name__ == "__main__":
    unittest.main()
