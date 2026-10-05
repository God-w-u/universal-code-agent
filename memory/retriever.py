"""MemoryRetriever：拿当前 query 在历史里找相关的那几条（Phase 11.1）。

**为什么先做轻量检索**：这一步的目标不是「找得最准」，是「不接向量库、
不接模型也能用，而且结果确定」——同样一批记录 + 同样的 query，永远给出
同样的顺序。等它证明有用、量出真不够用了，再谈 embedding。

**拆词**（`tokenize`）：小写、去重、保序。字母数字下划线一串是一个词；
汉字串切 **bigram**（修复缓存 → 修复 / 复缓 / 缓存）——为什么不整段
匹配见 _TOKEN 上面的注释。

**排序（deterministic）**：

- 一条记录的分 = 命中的查询词个数，task 里的命中算 2 分、summary 里的
  算 1 分（task 是「当时要干的事」，比摘要里顺带提一句更相关）。
- 同分时**后写入的排在前面**——store 的顺序就是时间顺序，更近的历史
  更有参考价值。
- 排序键只有（分数，写入顺序）两个整数：不掺随机、不去解析时间戳，
  顺序永远是定的。

**进 context 的边界**（这一层最要紧的约束）：只返回 top_k 条，每条
summary 截到 summary_chars 以内——历史攒得再多，也不会把 context 撑爆。
没有命中的记录**不出现在结果里**（不是排在后面），免得模型把不相干的
历史当成线索。
"""

import re
from dataclasses import dataclass

from memory.base import MemoryRecord, MemoryStore
from tools.base import ToolError

# 默认返回几条、每条 summary 最长多少字符（构造时都能改）
DEFAULT_TOP_K = 5
MAX_SUMMARY_CHARS = 200

# 打分权重：task 命中比 summary 命中重（见模块开头）
SCORE_TASK = 2
SCORE_SUMMARY = 1

# 拆词：字母数字下划线一串，或者汉字一串（区间两端是「一」U+4E00 和
# 「鿿」U+9FFF，就是常用汉字那一整块）。ASCII 这一段和
# index/retrieval.py 的 _TOKEN 一致；汉字这一段**故意不同**——代码检索
# 那边汉字多出现在注释 / 字符串里，整段匹配够用；而 Memory 的 task /
# summary 是自然语言，整段「修复缓存过期」只有当查询一字不差时才命中，
# 等于查不了。切成 bigram 之后查「缓存」也能命中——不引分词库、不加载
# 模型，规则确定，同输入同输出。
_TOKEN = re.compile(r"[a-z0-9_]+|[一-鿿]+")


def is_cjk(char: str) -> bool:
    """是不是汉字（用来判断一整段是汉字还是 ASCII）。"""
    return "一" <= char <= "鿿"


def tokenize(text: str) -> tuple[str, ...]:
    """把 text 拆成检索用的词：小写、去重（保序）；汉字串切成 bigram。

    凑不成 bigram 的单个汉字原样留成一个词（短查询也要能用；但「存」
    和别的记录里的「缓存」是不同的词——单字查询本来就噪声大，不值得
    为它再加一条规则）。空串 / 全是标点 → 空元组。
    """
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    for piece in _TOKEN.findall(text.lower()):
        if not is_cjk(piece[0]):
            add(piece)
        elif len(piece) == 1:
            add(piece)
        else:
            for start in range(len(piece) - 1):
                add(piece[start : start + 2])
    return tuple(tokens)


def _score(record: MemoryRecord, query_tokens: set[str]) -> int:
    """一条记录命中多少个查询词：task 里命中算 SCORE_TASK 分、summary
    里算 SCORE_SUMMARY 分；同一个词在两处都出现就都算。"""
    task_hits = len(query_tokens & set(tokenize(record.task)))
    summary_hits = len(query_tokens & set(tokenize(record.summary)))
    return SCORE_TASK * task_hits + SCORE_SUMMARY * summary_hits


@dataclass(frozen=True)
class MemoryHit:
    """一条命中：记录（summary 已按上限裁成副本）+ 匹配得分。

    `record` 就是要进 context 的那份（不是 store 里的原文）；要全文
    自己去 `store.load()` 找——模型看到的是裁过的短版。
    """

    record: MemoryRecord
    score: int


class MemoryRetriever:
    """按 query 检索历史；结果条数和 summary 长度都有上限（见模块开头）。"""

    def __init__(
        self,
        store: MemoryStore,
        *,
        top_k: int = DEFAULT_TOP_K,
        summary_chars: int = MAX_SUMMARY_CHARS,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k 至少为 1，收到 {top_k}")
        if summary_chars < 1:
            raise ValueError(f"summary_chars 至少为 1，收到 {summary_chars}")
        self.store = store
        self.top_k = top_k
        self.summary_chars = summary_chars

    def retrieve(
        self, query: str, *, top_k: int | None = None
    ) -> list[MemoryHit]:
        """找和 query 相关的历史，最多 top_k 条（不传就用构造时的默认）。

        没有命中就返回空列表——query 拆不出词、历史是空的、一个词都对
        不上，都走这条路；这是正常结果，不是错误。
        """
        limit = self.top_k if top_k is None else top_k
        if limit < 1:
            raise ToolError(f"top_k 至少为 1，收到 {limit}")

        wanted = set(tokenize(query))
        if not wanted:
            return []

        scored: list[tuple[int, int, MemoryRecord]] = []
        for order, record in enumerate(self.store.load()):
            score = _score(record, wanted)
            if score > 0:
                scored.append((score, order, record))

        # 分高的在前；同分后写入的在前（order 大的在前）。排序键只有
        # 这两个整数，顺序是定的。
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [
            MemoryHit(record=record.clipped(self.summary_chars), score=score)
            for score, _, record in scored[:limit]
        ]
