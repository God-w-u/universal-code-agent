"""Retrieval Evaluation：给检索打分的尺子。

8.4 把检索做出来了，但「检索好不好」不能靠感觉。这一步加的是**离线评估**：
一组手写的 case（一句查询 + 期望命中的目标），挨条喂给 Retriever，看期望的
目标落在第几名，再汇总成 Hit@1 / Hit@3 / Hit@5 和平均耗时。以后加 embedding
也是拿这套数字比——好没好，看同一组 case 的命中率涨没涨。

几条刻意的边界：

- **只调用，不改动**：评估模块只用 `Retriever.search()` 这个公开入口，打分、
  排序、索引全都不碰。指标是从结果里数出来的（`rank` 第几名命中），不是
  从 Retriever 内部抠出来的——这样它测的才是「检索给人看到的东西」。
- **Hit@k 只统计「该找到」的 case**：本来就查不到的查询（`expected_nothing`）
  是另一类——它们的正确答案是「一条都不返回」，混进命中率里只会把分数搅浑。
  两类分开统计，小结里都写着。
- **建索引不算检索耗时**：第一次 `search()` 会顺手把索引建起来（8.4 的懒建），
  那一两秒混进 `elapsed_time` 就成了假数据。评估开始前先建好，计时只包住
  `search()` 本身。
- **case 存在 JSON 里，不写在 py 里**：这个项目把自己也索引了。case 写成
  Python 常量，那些查询字符串就会跟着模块的 chunk 一起进索引——等于把答案
  提前埋进考卷（8.4 的真项目测试就踩过这个坑）。JSON 没有符号、不产 chunk。
- **指标计算独立**：Hit@1/3/5、平均耗时都在这一层算，`rank` 是唯一的事实，
  命中率是现推的——不会出现「明细说命中、汇总说没中」。

9.5 把 case 集扩成「能区分语义检索好坏」的规模（原来的 7 正 2 反很快就被
字面检索打满，量不出 embedding 有没有用），同时给评估加了两件东西：

- **元数据 + 分组统计**：case 可以标 `language` / `query_lang` /
  `query_type` / `corpus`（都可以不标）；`EvaluationReport.group_by()` 和
  `grouped_summary()` 把同一份结果按这些字段切片印出来——整体、按语言、
  按查询类型、反例各一行。**Hit@1/3/5 的定义一个字没改**，切出来的每组
  用的还是同一套算法。
- **多语料**：C/C++ 的评估目标放在 `tests/fixtures/cpp/`（很小、真实、
  能被检索到，且不在生产索引里——`tools/tree.py` 不扫 `fixtures/`）。
  `Evaluator` 收 `{语料名: Retriever}`，case 用 `corpus` 指到哪一份；
  没写的走默认语料 `project`。

不做 embedding、不做向量库、不做自动生成测试问题、不做 Agent Tool；也**不改
Retriever 的评分规则和排序逻辑**，不改 Agent / RepairLoop / Verifier / tools。
"""

import json
import time
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from index.chunks import CodeChunk
from index.retrieval import Retriever, SearchResult
from tools.base import ToolError


# 自带的那组 case。和本模块放在一起，但**不是** py 文件——见上面那条边界。
DEFAULT_CASES_PATH = Path(__file__).with_name("evaluation_cases.json")

# 默认取回几条：要算 Hit@5，至少得有 5 条可比。
DEFAULT_TOP_K = 5

# 没写 `corpus` 的 case 落在哪个语料上。项目自己的索引就叫这个名字。
DEFAULT_CORPUS = "project"

# case 元数据的取值范围（9.5）。空串 = 没标，不进分组；标了就得是这几个之一
# ——写错（比如 "py" / "python3" / "中文"）会静默地把这条 case 挪出分组，
# 看起来像「这组命中率怪怪的」，其实是元数据写错了，所以这里直接报错。
_VOCABULARY: dict[str, tuple[str, ...]] = {
    "language": ("python", "cpp"),
    "query_lang": ("zh", "en"),
    "query_type": ("lexical", "semantic"),
}

# 分组统计按这几个字段切，打印时的中文标签。顺序就是打印顺序。
GROUP_LABELS: tuple[tuple[str, str], ...] = (
    ("language", "语言"),
    ("query_lang", "查询语言"),
    ("query_type", "查询类型"),
    ("corpus", "语料"),
)


@dataclass(frozen=True)
class EvaluationCase:
    """一条评估用例：问什么、期望在哪儿找到。

    - `query`：喂给 Retriever 的那句话。
    - `expected_path` / `expected_symbol`：期望命中的目标。可以只给一个，
      也可以都给（都给就是「必须同时满足」）。symbol 简单名（`build`）和
      限定名（`ChunkIndex.build`）都认。
    - `expected_nothing`：反例——这条查询**本来就该查不到**，查到反而算错。
      它和上面两个只能给一边。
    - `scenario` / `note`：给人看的。`scenario` 是短标签（`file_name`、
      `keyword` 这些），`note` 说明这条在测什么。要加字段（难度、来源……）
      就从这儿加，不动评估逻辑。
    - `language` / `query_lang` / `query_type` / `corpus`（9.5）：可选元数据，
      **只影响分组统计，不影响这条 case 怎么判分**。空串 = 没标，这条就不
      进对应的分组（老 case 文件没有这些字段，照常能读）。取值：
      `language` ∈ {python, cpp}（目标代码的语言）、`query_lang` ∈ {zh, en}
      （查询用的语言）、`query_type` ∈ {lexical, semantic}（查询是照名字查
      还是说人话）、`corpus` = 语料名（`None` = 默认语料，见 `DEFAULT_CORPUS`）。
    """

    query: str
    expected_path: str | None = None
    expected_symbol: str | None = None
    expected_nothing: bool = False
    scenario: str = ""
    note: str = ""
    language: str = ""
    query_lang: str = ""
    query_type: str = ""
    corpus: str | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ToolError("case 的 query 不能是空的")
        if self.expected_nothing:
            if self.expected_path is not None or self.expected_symbol is not None:
                raise ToolError(
                    "expected_nothing 和 expected_path / expected_symbol 只能给一边"
                )
        elif self.expected_path is None and self.expected_symbol is None:
            raise ToolError(
                "case 得说清期望什么：expected_path / expected_symbol，或者 expected_nothing"
            )
        for name, allowed in _VOCABULARY.items():
            value = getattr(self, name)
            if value and value not in allowed:
                raise ToolError(
                    f"case 的 {name} 只认 {sorted(allowed)}，收到 {value!r}"
                )
        if self.corpus is not None and not self.corpus.strip():
            raise ToolError("case 的 corpus 要么不写，要么给个非空的名字")

    @property
    def target(self) -> str:
        """期望的目标写成一行，给人看。"""
        if self.expected_path and self.expected_symbol:
            return f"{self.expected_path}:{self.expected_symbol}"
        return self.expected_path or self.expected_symbol or "（本来就该查不到）"

    def matches(self, chunk: CodeChunk) -> bool:
        """这个 chunk 算不算这条 case 期望命中的目标。

        给了几个条件就要同时满足几个：只给 path 就是「这个文件里的哪条都算」，
        只给 symbol 就是「哪个文件里的都算」，两个都给就都要对上。
        """
        if self.expected_path is not None and chunk.path != self.expected_path:
            return False
        if self.expected_symbol is not None:
            if chunk.symbol != self.expected_symbol and _simple(chunk.symbol) != self.expected_symbol:
                return False
        return True


@dataclass(frozen=True)
class EvaluationResult:
    """一条 case 跑完的结果。

    存下来的事实只有 `rank`（第几名命中）和 `elapsed_time`（这一条花了多少秒）；
    `hit1` / `hit3` / `hit5` 是从 `rank` 现推的——不会出现「明细和汇总对不上」。
    """

    case: EvaluationCase
    results: tuple[SearchResult, ...]
    rank: int | None
    elapsed_time: float

    @property
    def query(self) -> str:
        return self.case.query

    @property
    def hit1(self) -> bool:
        """期望的目标是不是就在第 1 条（Hit@1）。"""
        return self.rank is not None and self.rank <= 1

    @property
    def hit3(self) -> bool:
        return self.rank is not None and self.rank <= 3

    @property
    def hit5(self) -> bool:
        return self.rank is not None and self.rank <= 5

    @property
    def found_nothing(self) -> bool:
        """一条结果都没返回。"""
        return not self.results

    @property
    def verdict(self) -> str:
        """为什么算命中 / 算什么没命中，一句话。"""
        if self.case.expected_nothing:
            if self.found_nothing:
                return "确实没查到"
            first = self.results[0].chunk
            return f"不该查到，却查到 {len(self.results)} 条（第一条 {first.path}:{first.symbol}）"
        if not self.results:
            return f"一条都没查到（期望 {self.case.target}）"
        if self.rank is None:
            first = self.results[0].chunk
            return f"前 {len(self.results)} 条都没命中（期望 {self.case.target}；第一条是 {first.path}:{first.symbol}）"
        return f"第 {self.rank} 名命中 {self.case.target}"


@dataclass(frozen=True)
class EvaluationReport:
    """一组 case 跑完的汇总。命中率都是现算的，不是另存的一份数。"""

    results: tuple[EvaluationResult, ...]

    @property
    def answerable(self) -> tuple[EvaluationResult, ...]:
        """「该找到」的那些 case——Hit@k 只统计它们。"""
        return tuple(item for item in self.results if not item.case.expected_nothing)

    @property
    def nothing_cases(self) -> tuple[EvaluationResult, ...]:
        """「本来就该查不到」的反例。"""
        return tuple(item for item in self.results if item.case.expected_nothing)

    @property
    def hit1(self) -> float:
        """Hit@1：该找到的 case 里，期望的目标排第一的比例（没 case 就是 0）。"""
        return _rate(self.answerable, lambda item: item.hit1)

    @property
    def hit3(self) -> float:
        return _rate(self.answerable, lambda item: item.hit3)

    @property
    def hit5(self) -> float:
        return _rate(self.answerable, lambda item: item.hit5)

    @property
    def average_time(self) -> float:
        """平均检索耗时（秒），所有 case 一起算——反例也是一次真查询。"""
        if not self.results:
            return 0.0
        return sum(item.elapsed_time for item in self.results) / len(self.results)

    @property
    def misses(self) -> tuple[EvaluationResult, ...]:
        """该找到、前 top_k 里却没命中的——看失败先看这里。"""
        return tuple(item for item in self.answerable if item.rank is None)

    def group_by(self, field: str) -> dict[str, "EvaluationReport"]:
        """把报告按 case 元数据切片：{取值: 那一小份报告}（9.5）。

        `field` 是 `EvaluationCase` 上带元数据的字段名（`language` /
        `query_lang` / `query_type` / `corpus`）。**只切正例**：反例的答案
        是「查不到」，混进语言 / 类型的分组里只会把每组的数字搅浑，它们
        自己那组在 `nothing_cases` / `grouped_summary()` 里。没标（空串）
        的 case 不进任何一组——分组是可选元数据，没标就是没得可切。
        """
        groups: dict[str, list[EvaluationResult]] = {}
        for item in self.answerable:
            value = getattr(item.case, field)
            if value:
                groups.setdefault(value, []).append(item)
        return {
            value: EvaluationReport(tuple(items)) for value, items in groups.items()
        }

    def tally(self) -> str:
        """一份报告的一行总账（现算的），分组里每条都复用它——两边不会对不上。"""
        answerable = self.answerable
        parts = [
            f"该找到 {len(answerable)} 条："
            f"Hit@1 {_count(answerable, lambda i: i.hit1)}/{len(answerable)}"
            f"（{_rate(answerable, lambda i: i.hit1):.1%}）、"
            f"Hit@3 {_count(answerable, lambda i: i.hit3)}/{len(answerable)}"
            f"（{_rate(answerable, lambda i: i.hit3):.1%}）、"
            f"Hit@5 {_count(answerable, lambda i: i.hit5)}/{len(answerable)}"
            f"（{_rate(answerable, lambda i: i.hit5):.1%}）"
        ]
        if self.nothing_cases:
            ok = _count(self.nothing_cases, lambda item: item.found_nothing)
            parts.append(f"反例 {len(self.nothing_cases)} 条、{ok} 条确实没查到")
        parts.append(f"平均 {self.average_time * 1000:.1f} ms")
        return "；".join(parts)

    def grouped_summary(self) -> str:
        """整体 + 各分组的总账（9.5）。

        没标元数据、也没有反例时返回空串——老 case 文件跑出来的输出和 8.5
        逐字一样，分组是「有料才印」的加分项，不是必印的表头。
        """
        groups = {
            field: self.group_by(field)
            for field in ("language", "query_lang", "query_type", "corpus")
        }
        if not any(groups.values()) and not self.nothing_cases:
            return ""

        lines = [f"整体：{self.tally()}"]
        for field, label in GROUP_LABELS:
            for value, report in groups[field].items():
                lines.append(f"  {label}={value}：{report.tally()}")
        if self.nothing_cases:
            ok = _count(self.nothing_cases, lambda item: item.found_nothing)
            false_positives = len(self.nothing_cases) - ok
            lines.append(
                f"  negative：反例 {len(self.nothing_cases)} 条、{ok} 条确实没查到、"
                f"假阳性 {false_positives} 条"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """逐条 + 总账，拼成一段可以直接打印的文本。"""
        lines = []
        for item in self.results:
            found = item.rank is not None or (
                item.case.expected_nothing and item.found_nothing
            )
            label = item.case.scenario or "-"
            lines.append(
                f"  {'✓' if found else '✗'} [{_pad(label, 11)}] {_pad(item.query, 26)} {item.verdict}"
            )

        total = len(self.results)
        answerable = len(self.answerable)
        lines.append("")
        lines.append(
            f"共 {total} 条 case：{answerable} 条该找到"
            f"（Hit@1 {_count(self.answerable, lambda i: i.hit1)}/{answerable}、"
            f"Hit@3 {_count(self.answerable, lambda i: i.hit3)}/{answerable}、"
            f"Hit@5 {_count(self.answerable, lambda i: i.hit5)}/{answerable}）"
        )
        if self.nothing_cases:
            ok = _count(self.nothing_cases, lambda item: item.found_nothing)
            lines.append(
                f"另外 {len(self.nothing_cases)} 条本来就该查不到，{ok} 条确实没查到"
            )
        lines.append(f"平均检索耗时 {self.average_time * 1000:.1f} ms（只算检索，不含建索引）")
        return "\n".join(lines)


class Evaluator:
    """拿一个（或几个语料的）Retriever 和一组 case，跑一遍，给一份报告。

        report = Evaluator(Retriever()).run(load_cases())
        print(report.summary())
        report.hit1, report.hit3, report.hit5, report.average_time

    只做三件事：调 `retriever.search()`、拿 case 去对结果、把结果加起来。
    指标一律在报告里现算，Retriever 那边一个字都不知道自己在被测。

    给的可以是一个 Retriever（= 默认语料，9.4 的老形状），也可以是
    `{语料名: Retriever}`（9.5：项目的 C++ fixture 有自己的索引）。带
    `corpus` 的 case 走那张表里对应的 retriever；指了不存在的语料直接报错
    ——查不到和「根本没建这份索引」是两回事，不能混成一条 miss。
    """

    def __init__(
        self,
        retriever: Retriever | Mapping[str, Retriever],
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if isinstance(retriever, Mapping):
            if not retriever:
                raise ToolError("语料表是空的：至少得有一个 retriever")
            self._retrievers = dict(retriever)
        else:
            self._retrievers = {DEFAULT_CORPUS: retriever}
        self.top_k = top_k

    @property
    def retrievers(self) -> dict[str, Retriever]:
        """语料名 → retriever（只读一份副本，改它不影响评估）。"""
        return dict(self._retrievers)

    def retriever_for(self, case: EvaluationCase) -> Retriever:
        """这条 case 该用哪个 retriever：case.corpus 或默认语料。"""
        name = case.corpus or DEFAULT_CORPUS
        try:
            return self._retrievers[name]
        except KeyError:
            known = "、".join(sorted(self._retrievers))
            raise ToolError(
                f"case「{case.query}」要的语料 {name!r} 不在这次评估里"
                f"（有的：{known}）"
            ) from None

    def run(self, cases: Iterable[EvaluationCase]) -> EvaluationReport:
        """把 case 挨条跑一遍。查询本身出错（路径没了之类）由 Retriever 抛，
        评估不替它兜——那是真出事了，不是「没命中」。"""
        # 建索引不算检索耗时：第一次 search() 会懒建一次，先在这儿做掉，
        # 免得第一条 case 的 elapsed_time 里混进建索引的那一两秒。
        for retriever in self._retrievers.values():
            if not retriever.index.built:
                retriever.index.build()

        results = []
        for case in cases:
            retriever = self.retriever_for(case)
            start = time.perf_counter()
            found = retriever.search(case.query, top_k=self.top_k)
            elapsed = time.perf_counter() - start
            results.append(EvaluationResult(case, found, _rank(case, found), elapsed))
        return EvaluationReport(tuple(results))


def load_cases(path: Path | None = None) -> tuple[EvaluationCase, ...]:
    """从 JSON 读一组 case，默认读项目自带的那份。

    字段名写错会直接报错，不静默忽略：写错的期望等于那条 case 永远不命中，
    看起来像「检索不行」，其实是 case 写错了。字段能写哪些见 EvaluationCase。
    """
    path = DEFAULT_CASES_PATH if path is None else Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ToolError(f"读不了 case 文件 {path}：{error}") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ToolError(f"case 文件不是合法 JSON：{path}（{error}）") from error
    if not isinstance(raw, list):
        raise ToolError(f"case 文件的最外层该是一个数组：{path}")
    return tuple(_case_from(item, path) for item in raw)


_FIELDS = (
    "query",
    "expected_path",
    "expected_symbol",
    "expected_nothing",
    "scenario",
    "note",
    "language",
    "query_lang",
    "query_type",
    "corpus",
)

_TEXT_FIELDS = (
    "query",
    "expected_path",
    "expected_symbol",
    "scenario",
    "note",
    "language",
    "query_lang",
    "query_type",
    "corpus",
)


def _case_from(item: object, path: Path) -> EvaluationCase:
    """JSON 里的一条 → EvaluationCase，顺手校验字段名和类型。"""
    if not isinstance(item, dict):
        raise ToolError(f"{path} 里有一条不是对象：{item!r}")
    unknown = sorted(set(item) - set(_FIELDS))
    if unknown:
        raise ToolError(
            f"{path} 里有不认识的字段 {unknown}；能写的只有 {list(_FIELDS)}"
        )
    for name in _TEXT_FIELDS:
        if name in item and not isinstance(item[name], str):
            raise ToolError(f"{path} 里的 {name} 应该是字符串：{item[name]!r}")
    if "expected_nothing" in item and not isinstance(item["expected_nothing"], bool):
        raise ToolError(f"{path} 里的 expected_nothing 应该是 true / false：{item['expected_nothing']!r}")
    try:
        return EvaluationCase(**item)
    except TypeError as error:  # 少写了 query 之类
        raise ToolError(f"{path} 里有条 case 写不完整：{error}") from error


def _rank(case: EvaluationCase, results: tuple[SearchResult, ...]) -> int | None:
    """期望的目标第一次出现在第几名（1 起）；前 top_k 里没有就是 None。

    反例（expected_nothing）没有「第几名」可言，永远是 None。
    """
    if case.expected_nothing:
        return None
    for position, item in enumerate(results, start=1):
        if case.matches(item.chunk):
            return position
    return None


def _rate(items: tuple[EvaluationResult, ...], hit) -> float:
    if not items:
        return 0.0
    return _count(items, hit) / len(items)


def _count(items: tuple[EvaluationResult, ...], hit) -> int:
    return sum(1 for item in items if hit(item))


def _pad(text: str, width: int) -> str:
    """按**显示宽度**补空格：汉字占两格，不这么补的话中文查询会把后面的列顶歪。"""
    shown = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
    return text + " " * max(0, width - shown)


def _simple(qualified: str) -> str:
    """限定名的最后一段：`ChunkIndex.build` → `build`。

    和 index/chunks.py、index/retrieval.py 里那两个是同一条规则。几行的小东西，
    与其跨模块去 import 别人的私有函数，不如各自留着。
    """
    return qualified.rsplit(".", 1)[-1]


# 想在真项目上跑一遍自带 case 集，用 `python -m index`（入口在 index/__main__.py）。
# 这里不写 `if __name__ == "__main__"`：index/__init__.py 会导入本模块，再用
# -m 直接跑它就成了「同一个模块进两次」，Python 会打一行 RuntimeWarning。
