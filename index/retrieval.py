"""Code Retrieval：拿一句查询，在现成的 chunk 里找最像的那些，排好序。

检索这条线的第四步，也是最后一步（再往后才是 embedding 那条路）。输入一个
字符串，输出一串 SearchResult——每条带着命中的 chunk、总分、以及**为什么
命中**：命中是符号名？是路径？还是正文里出现了这个词。

四种匹配全是「字面」的，没有语义：symbol 简单名、qualified symbol、文件
路径、chunk 正文。分数是拍的权重，全在下面那张表里，要调只有这一处。

几条刻意的边界：

- **只在现成的 chunk 上找**：进来的是 ChunkIndex，不重扫源码、不重解析、
  不重切块——扫描、提取符号、切块分别是 8.1 / 8.2 / 8.3 的事，这儿一件都
  不做。要刷新就调 index.refresh()（或者再 build() 一次），检索只读不写。
- **排序是确定的**：整名命中（查询就是一个限定名，和某个符号一模一样）
  永远排最前；其余先比分数，同分再按 路径 → 起始行 → 符号名 排。同一句
  查询跑一百遍顺序一样，top_k 截出来的也就是同一批。
- **块越具体越可信**（8.7）：同一条正文命中，落在 function / method 上会
  多给一档 `SCORE_SPECIFIC`——类块的正文本来就含着成员函数的正文，同一句
  话两边都命中时，要的是里面那个函数，不是包着它的类。这一档（10）刻意小
  于「名字里的词」（30）：正文命中再怎么攒也压不过名字命中。
- **打分可以换**：scorers 是一串 `(Query, CodeChunk) -> list[Match]` 的函数，
  默认三个（符号 / 路径 / 正文）。以后加 embedding 就是再写一个这样的函数
  塞进 Retriever——它对每个 chunk 返回一条 `Match("embedding", ..., 相似度)`，
  排序、去重、top_k 统统不用动。
- **不产生假命中**：查不到就是空结果，不凑数、不猜同义词。检索结果最终是
  要给人或模型看的，凭空多一条「可能相关」比少一条更贵。

不做 embedding、不做向量库、不做语义搜索、不做 Hybrid Search / RAG Tool，
也不改 Agent / RepairLoop / Verifier / 编译测试 / EditFile 的行为。
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from index.chunks import ChunkIndex, CodeChunk
from index.symbols import FUNCTION, METHOD
from tools.base import ToolError


# 命中的方式不同，可信程度差很多：名字就叫 run 的，比正文里恰好出现过
# "run" 的有用得多。所以分档——整名 > 名字里的词 > 名字的一部分 > 路径 >
# 正文。这些数是拍的，觉得排序不顺手就调这一张表。
SCORE_QUALIFIED = 100.0  # 查询就是限定名（Agent.run）
SCORE_NAME = 80.0        # 查询就是简单名（run）
SCORE_PATH = 70.0        # 查询就是整条路径（agent/agent.py）
SCORE_FILE_NAME = 60.0   # 查询就是文件名（client.py），或文件名的词干（client）
SCORE_PATH_PART = 40.0   # 查询是路径里的一段（llm、tools）
SCORE_CODE = 40.0        # 整条查询在正文里出现过
SCORE_SYMBOL_WORD = 30.0  # 查询是名字里的一个词（max_repair_rounds ← repair）
SCORE_SYMBOL_PART = 15.0  # 查询是名字的一部分（agen ← Agent）
SCORE_PATH_HIT = 12.0     # 路径里出现了查询词
SCORE_SPECIFIC = 10.0     # 正文命中落在 function / method 上（8.7）：比落在包着它的类 / 结构体上更具体
SCORE_CODE_HIT = 6.0      # 单个查询词在正文里出现过

# 「具体」的块：函数 / 方法。类 / 结构体是容器，正文里含着成员——不算。
SPECIFIC_KINDS = (FUNCTION, METHOD)

# 封顶：同一个词出现几次都只算一次，而且最多算这么多个词。不封的话，
# 一句五六个常用词的查询能从正文里叠出一大堆分，盖过真正的符号命中。
MAX_SYMBOL_HITS = 2
MAX_PATH_HITS = 2
MAX_CODE_HITS = 3

# 拆词：字母数字下划线一串，或者汉字一串（区间两端是「一」U+4E00 和
# 「鿿」U+9FFF，就是常用汉字那一整块）。其余全是分隔符。
# 拆出来的词要多长才够格做模糊匹配，见 _long_enough()。
_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]+")

# 整名比较前要剥掉的首尾标点：`Agent.run()` 和 `Agent.run` 是一回事。
_EDGE = " \t\r\n()[]{}<>'\"`,;:.*&#@!?/\\|+-="

# 名字拆成词用：ContextManager → context + manager，max_repair_rounds →
# max + repair + rounds。只用于「名字里的词」这一档。
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")


@dataclass(frozen=True)
class Query:
    """拆好的查询。所有比较都在它上面做，一句查询只拆一次。

    - `text`：原样的查询（只去了首尾空白），留着给日志 / 报错看。
    - `stripped`：剥掉首尾标点、转小写，整名比较用（`Agent.run()` → `agent.run`）。
    - `tokens`：词表，小写、去重、保序。部分匹配都从这儿取词。
    """

    text: str
    stripped: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Match:
    """一条命中理由：哪种命中、命中的是什么、这一项值多少分。

    `kind` 是英文短 id，目前四种：`symbol` / `path` / `code` / `specific`
    （8.7 加的粒度奖励：正文命中落在 function / method 上）。embedding 那条
    用 `embedding`（8.6 已经有一个假实现走这条路）——`detail` 写模型或参数，
    `score` 写相似度乘完权重的分。分数是浮点，接口现在就是照后面的用法留的。
    """

    kind: str
    detail: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    """一条检索结果：命中的 chunk、总分、命中的理由（可能不止一条）。

    `score` 就是 `matches` 里各项分数之和，不另算。排序用的也是它。
    """

    chunk: CodeChunk
    score: float
    matches: tuple[Match, ...]

    @property
    def reason(self) -> str:
        """把命中理由拼成一句话，给人 / 给模型看。

        `符号名 Agent.run（+80）；正文里的 timeout（+6）`
        """
        return "；".join(f"{item.detail}（+{item.score:g}）" for item in self.matches)


# 一个打分函数：拿拆好的查询和一个 chunk，返回它命中了哪几条（没命中就是 []）。
# 加 embedding 就是照着这个签名再写一个，塞进 Retriever(scorers=...)。
Scorer = Callable[[Query, CodeChunk], list[Match]]


def symbol_matches(query: Query, chunk: CodeChunk) -> list[Match]:
    """符号名命中：限定名整名 > 简单名整名 > 名字里的一个词 > 名字的一部分。"""
    lowered = chunk.symbol.lower()

    if query.stripped == lowered:
        return [Match("symbol", f"限定名 {chunk.symbol}", SCORE_QUALIFIED)]
    if query.stripped == _simple(lowered):
        return [Match("symbol", f"符号名 {chunk.symbol}", SCORE_NAME)]

    words = _words(chunk.symbol)
    matches: list[Match] = []
    for token in query.tokens:
        if token in words:
            matches.append(Match("symbol", f"名字里的词 {token}", SCORE_SYMBOL_WORD))
        elif _long_enough(token) and token in lowered:
            matches.append(Match("symbol", f"名字里的 {token}", SCORE_SYMBOL_PART))
        if len(matches) == MAX_SYMBOL_HITS:
            break
    return matches


def path_matches(query: Query, chunk: CodeChunk) -> list[Match]:
    """路径命中：整条路径 > 文件名 / 目录名 > 路径里出现查询词。"""
    path = chunk.path.lower()
    if query.stripped == path:
        return [Match("path", f"路径 {chunk.path}", SCORE_PATH)]

    parts = path.split("/")
    name = parts[-1]        # client.py
    stem = name.rsplit(".", 1)[0]   # client
    if query.stripped and query.stripped in (name, stem):
        return [Match("path", f"文件名 {chunk.path}", SCORE_FILE_NAME)]
    if query.stripped and query.stripped in parts[:-1]:
        return [Match("path", f"路径里的 {query.stripped}", SCORE_PATH_PART)]

    hits = [token for token in query.tokens if _long_enough(token) and token in path]
    return [
        Match("path", f"路径里的 {token}", SCORE_PATH_HIT)
        for token in hits[:MAX_PATH_HITS]
    ]


def code_matches(query: Query, chunk: CodeChunk) -> list[Match]:
    """正文命中：整条查询出现过 > 里面的词挨个出现过。

    整条命中就不再数词了——词本来就是那句话里的，数一遍是重复记分。

    有正文命中的 function / method 再补一档 SCORE_SPECIFIC（8.7）：类块的
    正文里含着自己所有成员的正文，同一句话在类块里出现，多半只是因为它出现
    在某个方法里——要的是那个方法。类 / 结构体自己不给这一档。
    """
    code = chunk.code.lower()

    if _long_enough(query.stripped) and query.stripped in code:
        matches = [Match("code", f"正文里的 {query.stripped}", SCORE_CODE)]
    else:
        hits = [token for token in query.tokens if _long_enough(token) and token in code]
        matches = [
            Match("code", f"正文里的 {token}", SCORE_CODE_HIT)
            for token in hits[:MAX_CODE_HITS]
        ]

    if matches and chunk.kind in SPECIFIC_KINDS:
        matches.append(Match("specific", f"更具体的 {chunk.kind}", SCORE_SPECIFIC))
    return matches


DEFAULT_SCORERS: tuple[Scorer, ...] = (symbol_matches, path_matches, code_matches)

# search() 不传 top_k 时给几条。10 是拍的上限，不是「只能给 10 条」。
DEFAULT_TOP_K = 10


class Retriever:
    """检索器。用法：建一个 → search()。

        retriever = Retriever()                    # 默认整个项目
        retriever = Retriever(root=some_dir)       # 指定目录（测试里用）
        retriever.search("max_repair_rounds")      # 前 10 条
        retriever.search("重试", top_k=5)           # 前 5 条
        retriever.search("Agent.run", top_k=None)  # 不限条数

    结果里整名命中永远最前，其余按分数从高到低，同分按 路径 → 起始行 →
    符号名——同一句查询的结果顺序永远一样。

    索引由谁管，和前面几层一个规矩：`root=`（或默认）是自己管的，`index=`
    传来的归调用方管。search() 发现索引从没建过，会先替它建一次——「只想
    搜一句，不想管索引」是正常用法，不该静默返回空。建过的（哪怕已经过时）
    不碰：检索不负责刷新，要刷新自己调 index.refresh()；刷完不用做别的——
    search() 每次都从索引现取 chunk。
    """

    def __init__(
        self,
        index: ChunkIndex | None = None,
        root: Path | None = None,
        scorers: tuple[Scorer, ...] = DEFAULT_SCORERS,
    ) -> None:
        if index is not None and root is not None:
            raise ToolError("index 和 root 只能给一个：传了 index 就由调用方管索引")
        self.index = index if index is not None else ChunkIndex(root=root)
        self.scorers = tuple(scorers)

    def search(self, query: str, top_k: int | None = DEFAULT_TOP_K) -> tuple[SearchResult, ...]:
        """查一句，返回排好序的结果。

        `top_k` 是最多给几条：`None` 不限，0 或负数就是一条都不给（不是
        「倒数几条」）。空查询、只有标点的查询，都老老实实返回空。
        """
        if not self.index.built:
            self.index.build()

        prepared = _prepare(query)

        results: list[SearchResult] = []
        for chunk in self.index.chunks():
            matches = [
                match
                for scorer in self.scorers
                for match in scorer(prepared, chunk)
            ]
            if matches:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=sum(item.score for item in matches),
                        matches=tuple(matches),
                    )
                )

        results.sort(key=_order)
        if top_k is not None:
            results = results[: max(0, top_k)]
        return tuple(results)


def _order(result: SearchResult) -> tuple[bool, float, str, int, str]:
    """排序键：整名命中最前，然后分数从高到低，同分再按位置——顺序钉死。

    整名命中（查询就是一个限定名，和这条结果的符号整名相同）单独拎成第一
    档：这是最强的一种证据。别的块靠各种命中凑出来的总分再高，也不该盖过
    它——分数比的是「同类证据的多少」，整名是「证据的种类」不同；不单独
    成档的话，理论上总有一个堆得出来的分数能翻过去（有测试拿硬塞的高分
    钉着这条）。同一档里仍然先比分数。
    """
    return (
        not _has_exact_symbol_match(result),
        -result.score,
        result.chunk.path,
        result.chunk.start_line,
        result.chunk.symbol,
    )


def _has_exact_symbol_match(result: SearchResult) -> bool:
    """这条结果里有没有「整名命中」——symbol 那一档的最高分。"""
    return any(
        item.kind == "symbol" and item.score >= SCORE_QUALIFIED
        for item in result.matches
    )


def _prepare(text: str) -> Query:
    """把查询拆成 Query：剥边、转小写、拆词、去重（保序）。"""
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if token not in tokens:
            tokens.append(token)
    return Query(
        text=text.strip(),
        stripped=text.strip(_EDGE).lower(),
        tokens=tuple(tokens),
    )


def _long_enough(token: str) -> bool:
    """够不够格做「扫一遍路径 / 正文」这种模糊匹配。

    两个字母的纯英文词（py、id、os）扫出来全是噪声——查 `agent/agent.py`
    不该把全项目每个 .py 都算一条命中，所以纯 ASCII 三个字符起。汉字两个字
    已经是一个词（超时），两个就够。精确匹配不受这条管：短词照样能精确命中
    符号名 / 文件名 / 目录名。
    """
    if any(ord(char) > 127 for char in token):
        return len(token) >= 2
    return len(token) >= 3


def _words(name: str) -> list[str]:
    """名字拆成词：`max_repair_rounds` → max / repair / rounds。"""
    return [word.lower() for word in _WORD.findall(name)]


def _simple(qualified: str) -> str:
    """限定名的最后一段：`Agent.Inner.run` → `run`。"""
    return qualified.rsplit(".", 1)[-1]
