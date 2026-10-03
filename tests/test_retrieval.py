"""Code Retrieval 测试：查得准不准、排得稳不稳、找不到时干不干净。

在临时目录里摆三个固定文件当检索对象（类 / 方法 / 函数 / 中文注释都有），
查询和期望结果都是手算好的——「查 run 应该先给名字就叫 run 的那个，而不是
正文里碰巧有 run 的」这类判断，是这一层唯一值钱的东西。

三条钉得最死的规矩：

- **排序确定**：同一句查询跑两遍，结果逐条一样；整名命中永远最前（排队时
  单独一档），其余同分的按 路径 → 行号 排。
- **不重扫**：索引建好之后往磁盘上加文件，检索看不见它——要看见得自己
  调 index.build()。
- **打分可换**：自定义 scorer 能加进来（embedding 以后就走这个口），
  也能把默认的几个整个换掉。
"""

import tempfile
import time
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from config import PROJECT_ROOT
from index import (
    DEFAULT_SCORERS,
    MAX_CODE_HITS,
    SCORE_CODE_HIT,
    SCORE_SPECIFIC,
    SCORE_SYMBOL_WORD,
    ChunkIndex,
    Evaluator,
    Match,
    Retriever,
    SearchResult,
    load_cases,
)
from index.__main__ import baseline_retrievers, build_corpora
from tools.base import ToolError


# 检索对象：一个类（带方法）、一个同名函数、另一个类。行号按注释里标的算。
AGENT = "\n".join(
    [
        "class Agent:",  # 1
        "    def run(self, question):",  # 2
        "        answer = self._ask(question)",  # 3
        "        return answer",  # 4
        "",  # 5
        "    def _ask(self, question):",  # 6
        "        return question",  # 7
        "",  # 8
        "    def quit(self):",  # 9
        "        pass",  # 10
    ]
) + "\n"

RUN_COMMAND = "\n".join(
    [
        "def run(command):",  # 1
        '    """跑一条命令，超时就杀掉。"""',  # 2
        "    return command",  # 3
    ]
) + "\n"

CLIENT = "\n".join(
    [
        "class LlmClient:",  # 1
        "    def send(self, prompt):",  # 2
        "        return prompt",  # 3
    ]
) + "\n"


class RetrievalTestBase(unittest.TestCase):
    """临时目录当 root，`make_fixture()` 摆好那三个文件。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def make(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def make_fixture(self) -> None:
        self.make("agent/agent.py", AGENT)
        self.make("tools/run_command.py", RUN_COMMAND)
        self.make("llm/client.py", CLIENT)

    def build(self, scorers=DEFAULT_SCORERS) -> Retriever:
        """摆好文件、建好索引的检索器。"""
        self.make_fixture()
        retriever = Retriever(root=self.root, scorers=scorers)
        retriever.index.build()
        return retriever

    def ids(self, results) -> list[str]:
        return [item.chunk.id for item in results]

    def symbols(self, results) -> list[str]:
        return [item.chunk.symbol for item in results]

    def kinds(self, result: SearchResult) -> list[str]:
        return [item.kind for item in result.matches]


class SymbolHitTest(RetrievalTestBase):
    """符号名命中：精确、限定名、部分、名字里的一个词。"""

    def test_exact_name(self):
        results = self.build().search("Agent")
        self.assertEqual(results[0].chunk.symbol, "Agent")
        self.assertEqual(results[0].chunk.path, "agent/agent.py")
        self.assertIn("symbol", self.kinds(results[0]))

    def test_exact_name_case_insensitive(self):
        results = self.build().search("agent")
        self.assertEqual(results[0].chunk.symbol, "Agent")

    def test_qualified_name(self):
        results = self.build().search("Agent.run")
        self.assertEqual(results[0].chunk.symbol, "Agent.run")
        self.assertEqual(results[0].score, sum(m.score for m in results[0].matches))

    def test_qualified_beats_simple_and_content(self):
        results = self.build().search("Agent.run")
        self.assertEqual(self.symbols(results)[0], "Agent.run")
        # 正文里也出现 agent / run 的那些，排在后面
        self.assertGreater(results[0].score, results[1].score)

    def test_exact_symbol_match_cannot_be_outscored(self):
        """查询就是某个符号的整名时，它永远第 1——别的块就算靠自定义 scorer
        硬塞一个更大的分数也不行。整名是最强的一种证据，这里比的是证据的
        「种类」，不是分数多少（_order 里那一档）。"""
        def inflate(query, chunk):
            if chunk.symbol == "Agent":  # 同名的类块，硬塞 999 分
                return [Match("code", "硬塞的分数", 999.0)]
            return []

        retriever = self.build(DEFAULT_SCORERS + (inflate,))
        results = retriever.search("Agent.run")
        self.assertEqual(results[0].chunk.symbol, "Agent.run")
        self.assertEqual(results[1].chunk.symbol, "Agent")

    def test_partial_name(self):
        results = self.build().search("agen")
        self.assertEqual(self.symbols(results)[0], "Agent")
        for result in results:
            self.assertEqual(result.chunk.path, "agent/agent.py")
            self.assertIn("symbol", self.kinds(result))

    def test_name_word(self):
        """`_ask` 拆成词就是 ask——查询不必写全 `_ask`。"""
        results = self.build().search("ask")
        self.assertEqual(self.symbols(results)[0], "Agent._ask")
        self.assertIn("symbol", self.kinds(results[0]))

    def test_no_symbol_hit_is_not_a_hit(self):
        """符号上没有、正文里也没有的词，不给结果。"""
        self.assertEqual(self.build().search("quit_now"), ())


class PathHitTest(RetrievalTestBase):
    """路径命中：整条路径、文件名、目录名。"""

    def test_file_name(self):
        results = self.build().search("run_command.py")
        self.assertEqual(results[0].chunk.path, "tools/run_command.py")
        self.assertIn("path", self.kinds(results[0]))

    def test_file_stem(self):
        results = self.build().search("client.py")
        self.assertEqual(results[0].chunk.path, "llm/client.py")

    def test_directory(self):
        results = self.build().search("llm")
        self.assertTrue(all(item.chunk.path == "llm/client.py" for item in results))
        self.assertIn("path", self.kinds(results[0]))

    def test_whole_path(self):
        results = self.build().search("agent/agent.py")
        self.assertTrue(all(item.chunk.path == "agent/agent.py" for item in results))
        self.assertIn("path", self.kinds(results[0]))


class CodeHitTest(RetrievalTestBase):
    """正文命中：整句出现过、或者里面的词出现过。"""

    def test_word_in_code(self):
        results = self.build().search("command")
        self.assertEqual(self.symbols(results)[0], "run")
        self.assertIn("code", self.kinds(results[0]))

    def test_chinese_word_in_code(self):
        results = self.build().search("超时就杀掉")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.path, "tools/run_command.py")
        self.assertIn("code", self.kinds(results[0]))

    def test_word_only_in_code(self):
        """`prompt` 不是任何符号名，只在正文里。类块和方法都命中时，方法在前
        （8.7 的粒度奖励）——类块的正文里那份是从 `send` 的正文里含过来的。"""
        results = self.build().search("prompt")
        self.assertTrue(results)
        self.assertTrue(all(item.chunk.path == "llm/client.py" for item in results))
        self.assertEqual(self.symbols(results), ["LlmClient.send", "LlmClient"])
        self.assertEqual(self.kinds(results[0]), ["code", "specific"])


class RankingTest(RetrievalTestBase):
    """排序：分数高的在前，同分的顺序钉死。"""

    def test_named_symbol_beats_code_content(self):
        results = self.build().search("run")
        self.assertEqual(self.symbols(results)[0], "run")  # 名字就叫 run 的函数
        self.assertEqual(results[0].chunk.path, "tools/run_command.py")
        self.assertGreater(results[0].score, results[1].score)

    def test_ties_are_deterministic(self):
        """`question` 只出现在 Agent 那几个 chunk 的正文里。

        两个方法同分（6 + 粒度奖励 10），类块没有奖励（6）——同分的两个按
        路径 → 起始行：Agent.run(2) 在 Agent._ask(6) 前面。"""
        retriever = self.build()
        first = self.ids(retriever.search("question"))
        second = self.ids(retriever.search("question"))
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            [
                "agent/agent.py:Agent.run:2-4",
                "agent/agent.py:Agent._ask:6-7",
                "agent/agent.py:Agent:1-10",
            ],
        )

    def test_prefix_ordering_is_by_line(self):
        results = self.build().search("Agent")
        self.assertEqual(
            self.ids(results),
            [
                "agent/agent.py:Agent:1-10",
                "agent/agent.py:Agent.run:2-4",
                "agent/agent.py:Agent._ask:6-7",
                "agent/agent.py:Agent.quit:9-10",
            ],
        )

    def test_results_all_come_from_the_index(self):
        retriever = self.build()
        everything = set(retriever.index.chunks())
        for result in retriever.search("run", top_k=None):
            self.assertIn(result.chunk, everything)


class GranularityTest(RetrievalTestBase):
    """粒度奖励（8.7）：同一句话在类块和它里面的方法都命中时，方法在前。"""

    def test_a_method_beats_the_class_that_contains_it(self):
        """`question` 出现在 Agent 两个方法的正文里，也出现在类块正文里——
        类块的正文本来就含着这两个方法的正文（8.5 诊断的那个毛病）。
        要的是方法，不是包着它的类。"""
        results = self.build().search("question")
        self.assertEqual(self.symbols(results), ["Agent.run", "Agent._ask", "Agent"])

    def test_the_container_is_not_dropped(self):
        """类块不是被判没了：正文里真有这句话，它还在结果里（只是排在后面），
        而且拿不到粒度奖励——奖励只给更具体的块。"""
        results = self.build().search("question")
        container = next(item for item in results if item.chunk.symbol == "Agent")
        self.assertEqual(self.kinds(container), ["code"])

    def test_the_reward_shows_up_in_the_reason(self):
        result = self.build().search("command")[0]
        self.assertEqual(result.chunk.symbol, "run")
        self.assertIn("specific", self.kinds(result))
        self.assertIn("更具体", result.reason)

    def test_the_reward_never_rides_alone(self):
        """奖励只跟着正文命中一起出现，不会自己造出一条结果。"""
        for result in self.build().search("question", top_k=None):
            self.assertIn("code", self.kinds(result))

    def test_a_name_hit_still_beats_the_biggest_body_hit(self):
        """奖励不让正文命中翻档：正文最多 3 个词（18）+ 奖励 10 = 28，还是
        低于「名字里的词」（30）——名字命中永远比正文命中可信。"""
        self.assertLess(
            MAX_CODE_HITS * SCORE_CODE_HIT + SCORE_SPECIFIC, SCORE_SYMBOL_WORD
        )


class TopKTest(RetrievalTestBase):
    """top_k：给几条就几条，0 / 负数是一条都没有，None 是不限。"""

    def test_limit(self):
        retriever = self.build()
        everything = retriever.search("run", top_k=None)
        self.assertGreater(len(everything), 2)
        self.assertEqual(self.ids(retriever.search("run", top_k=1)), self.ids(everything[:1]))
        self.assertEqual(self.ids(retriever.search("run", top_k=2)), self.ids(everything[:2]))

    def test_zero_and_negative(self):
        retriever = self.build()
        self.assertEqual(retriever.search("run", top_k=0), ())
        self.assertEqual(retriever.search("run", top_k=-3), ())

    def test_more_than_available(self):
        retriever = self.build()
        self.assertEqual(
            self.ids(retriever.search("run", top_k=100)),
            self.ids(retriever.search("run", top_k=None)),
        )

    def test_default_is_ten(self):
        retriever = self.build()
        self.assertEqual(len(retriever.search("question")), 3)  # 没到 10 条就给这么多


class EmptyQueryTest(RetrievalTestBase):
    """查不到、或者查询本身没意义时：老老实实空手回来。"""

    def test_no_match(self):
        self.assertEqual(self.build().search("没有这个词zzz"), ())

    def test_empty(self):
        retriever = self.build()
        self.assertEqual(retriever.search(""), ())
        self.assertEqual(retriever.search("   "), ())

    def test_only_punctuation(self):
        retriever = self.build()
        self.assertEqual(retriever.search("!!!"), ())
        self.assertEqual(retriever.search("()[]{}"), ())

    def test_special_characters_are_stripped(self):
        """`Agent.run()` 和 `Agent.run` 是一回事。"""
        retriever = self.build()
        for query in ("Agent.run()", "Agent.run(", "  Agent.run  ", "'Agent.run'"):
            self.assertEqual(self.symbols(retriever.search(query))[0], "Agent.run")


class ResultShapeTest(RetrievalTestBase):
    """SearchResult 的形状：分数是各项之和、理由是给人看的一句话。"""

    def test_score_is_sum_of_matches(self):
        for result in self.build().search("Agent.run", top_k=None):
            self.assertEqual(result.score, sum(item.score for item in result.matches))

    def test_reason_is_readable(self):
        result = self.build().search("Agent.run")[0]
        self.assertIn("Agent.run", result.reason)
        self.assertIn("+", result.reason)
        self.assertTrue(all(isinstance(item.detail, str) for item in result.matches))

    def test_kinds_are_known(self):
        # 默认三个打分器能给的种类（embedding 是加进来的，不在默认里）
        known = {"symbol", "path", "code", "specific"}
        for result in self.build().search("run", top_k=None):
            self.assertTrue(set(self.kinds(result)) <= known)

    def test_results_are_frozen(self):
        result = self.build().search("Agent")[0]
        with self.assertRaises(FrozenInstanceError):
            result.score = 1.0
        with self.assertRaises(FrozenInstanceError):
            result.matches[0].kind = "other"

    def test_no_matches_no_result(self):
        """一条理由都没有的 chunk 不进结果——宁可少给，不给凑数的。"""
        for result in self.build().search("run", top_k=None):
            self.assertTrue(result.matches)


class ExtensionTest(RetrievalTestBase):
    """打分机制是可换的：加一个 scorer、或者整个换掉。"""

    def test_custom_scorer_can_add_results(self):
        """默认打分器找不到的 chunk，自定义 scorer 能把它捞出来、还能顶到第 1。

        embedding 就是这个形状：对每个 chunk 都给一条 Match，分数是相似度
        乘权重，排序和 top_k 完全不用改。（查询不能用整名——整名命中的那条
        单独一档，任何自定义分数都翻不过去，那条规矩另有测试。）
        """

        def always_like_quit(query, chunk):
            if chunk.symbol == "Agent.quit":
                return [Match("embedding", "假装的相似度", 999.0)]
            return []

        retriever = self.build(DEFAULT_SCORERS + (always_like_quit,))
        results = retriever.search("question")
        self.assertEqual(results[0].chunk.symbol, "Agent.quit")
        self.assertEqual(self.kinds(results[0])[0], "embedding")

    def test_scorers_can_be_replaced(self):
        self.assertEqual(self.build(scorers=()).search("Agent"), ())

    def test_scorer_sees_every_chunk(self):
        """每个 scorer 对每个 chunk 都会被问一遍——不然语义打分没法给所有块打分。"""
        seen = []

        def spy(query, chunk):
            seen.append(chunk.id)
            return []

        retriever = self.build(scorers=(spy,))
        retriever.search("Agent")
        self.assertEqual(len(seen), len(retriever.index.chunks()))

    def test_query_is_prepared_once(self):
        """自定义 scorer 收到的 Query 是拆好的：原文、剥边小写、词表。"""
        seen = []

        def spy(query, chunk):
            seen.append(query)
            return []

        self.build(scorers=(spy,)).search("  Agent.run()  ")
        prepared = seen[0]
        self.assertEqual(prepared.text, "Agent.run()")
        self.assertEqual(prepared.stripped, "agent.run")
        self.assertEqual(prepared.tokens, ("agent", "run"))


class IndexLifecycleTest(RetrievalTestBase):
    """索引归谁管：没建过替它建一次，建过的不重扫，刷新得自己说。"""

    def test_search_builds_a_fresh_index(self):
        """只想搜一句、不想管索引——不该静默返回空。"""
        self.make_fixture()
        retriever = Retriever(root=self.root)
        self.assertFalse(retriever.index.built)
        self.assertTrue(retriever.search("Agent"))
        self.assertTrue(retriever.index.built)

    def test_search_does_not_rescan(self):
        retriever = self.build()
        self.make("tools/newcomer.py", "def newcomer():\n    pass\n")
        self.assertEqual(retriever.search("newcomer"), ())
        retriever.index.build()  # 刷新是显式的
        self.assertEqual(self.symbols(retriever.search("newcomer")), ["newcomer"])

    def test_index_and_root_are_exclusive(self):
        with self.assertRaises(ToolError):
            Retriever(index=ChunkIndex(root=self.root), root=self.root)

    def test_external_index_is_not_rebuilt(self):
        self.make_fixture()
        index = ChunkIndex(root=self.root)
        index.build()
        retriever = Retriever(index=index)
        before = index.chunks()
        retriever.search("Agent")
        self.assertEqual(index.chunks(), before)

    def test_search_does_not_touch_files(self):
        retriever = self.build()
        before = (self.root / "agent" / "agent.py").read_bytes()
        retriever.search("Agent.run")
        self.assertEqual((self.root / "agent" / "agent.py").read_bytes(), before)


class RealProjectRetrievalTest(unittest.TestCase):
    """拿真项目跑：查自己身上的符号，看是不是头一条就中。"""

    @classmethod
    def setUpClass(cls):
        cls.retriever = Retriever(root=PROJECT_ROOT)

    def test_class_name(self):
        results = self.retriever.search("ProjectIndex")
        self.assertEqual(results[0].chunk.path, "index/project.py")
        self.assertEqual(results[0].chunk.symbol, "ProjectIndex")
        self.assertIn("symbol", [item.kind for item in results[0].matches])

    def test_function_name(self):
        results = self.retriever.search("digest_of")
        self.assertEqual(results[0].chunk.symbol, "digest_of")

    def test_method_qualified_name(self):
        results = self.retriever.search("SymbolIndex.build")
        self.assertEqual(results[0].chunk.symbol, "SymbolIndex.build")

    def test_content_word(self):
        """正文关键词：命中的每一条，正文里都真的有这个词——不猜也不凑。

        sha256 在项目里到处都是（好几条并列 40 分），所以这条只断言「没有
        假命中」，不问第一条是谁：关键词检索本来就说不清哪一处更要紧。

        文件名命中那组在临时目录里测（PathHitTest）——真项目上这类查询躲不开
        「这个测试文件自己的源码也在索引里」：断言里写着文件名，这串字就成了
        测试文件自己的正文命中。
        """
        results = self.retriever.search("sha256", top_k=None)
        self.assertGreater(len(results), 1)
        for result in results:
            self.assertIn("sha256", result.chunk.code)
            self.assertIn("code", [item.kind for item in result.matches])

    def test_stable_across_calls(self):
        first = [item.chunk.id for item in self.retriever.search("chunk", top_k=20)]
        second = [item.chunk.id for item in self.retriever.search("chunk", top_k=20)]
        self.assertEqual(first, second)

    def test_top_k(self):
        self.assertLessEqual(len(self.retriever.search("agent", top_k=3)), 3)
        self.assertEqual(self.retriever.search("agent", top_k=0), ())

    def test_no_match(self):
        # 还是那个原因，写死的词会命中这个测试文件自己。现拼一个带时间戳
        # 的名字：源码里不可能有，跑一百遍也不可能撞上。
        self.assertEqual(self.retriever.search(f"zqxv{time.time_ns()}"), ())

    def test_results_come_from_index(self):
        everything = set(self.retriever.index.chunks())
        for result in self.retriever.search("timeout", top_k=None):
            self.assertIn(result.chunk, everything)


class RealProjectGranularityTest(unittest.TestCase):
    """真项目：8.7 之前被类块压住的那两条 case，现在目标排第 1。

    查询从自带 case 集里取（JSON 不在语料里）——原样写进这个文件的话，那
    串字就成了语料的一部分，这条 case 会命中测试文件自己（8.4 起的坑）。
    """

    @classmethod
    def setUpClass(cls):
        # 9.5 起自带 case 分两份语料（项目 + C++ fixture），各配一个 retriever
        cls.report = Evaluator(baseline_retrievers(build_corpora())).run(load_cases())

    def test_the_blocked_description_now_ranks_first(self):
        """两条 description 都指向同一个函数（8.5 的诊断：答案被它自己的
        类挡住）。带字面词的那条现在第 1；另一条一个词都不在代码里，依然
        查不到——那是留给真 embedding 的靶子。"""
        items = [item for item in self.report.results if item.case.scenario == "description"]
        self.assertEqual(len(items), 2)
        self.assertEqual(sorted(item.rank for item in items if item.rank), [1])
        self.assertEqual(sum(1 for item in items if item.rank is None), 1)

    def test_the_blocked_keyword_now_ranks_first(self):
        """`fsync` 那条：命中它的方法（连同另一个文件里的同名方法）现在
        排在各自的类块前面，目标从第 2 名升到第 1。"""
        item = next(item for item in self.report.results if item.case.scenario == "keyword")
        self.assertEqual(item.rank, 1, "类块不该再压着它自己的方法")


if __name__ == "__main__":
    unittest.main()
