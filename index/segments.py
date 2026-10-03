"""把紧凑文本切成模型窗口装得下的 segment（9.5）。

**为什么要切**：multilingual 模型的窗口只有 128 token，9.2/9.3 量出全库
近一半的块超窗被截——代码后半段等于没喂给模型。切块本身不能动
（`chunk.code` 是给工具看的原文），所以把 `compact_text` 拼出来的那段
文本再切成几段，每段都塞得进窗口；一个块的语义表示由它的各段一起给
（打分时取「和查询最像的那一段」，见 `EmbeddingScorer`）。

**怎么切**（确定性，同输入同输出）：

- 第一行是位置行（`path:symbol [kind]`），**每一段都带上它**——路径 /
  符号名 / 类型的线索一段都不丢。
- 其余部分按行贪心往里装；单行太长（一行就超预算）就按字符硬切。
- 装得下就原样一段返回：短块切出来的文本和 9.4 喂的**一字不差**。

**token 数是估计，不是真分词**。规则见 `estimate_tokens`。宁可切小
（多几段），也不能切大——切大的那部分会被模型悄悄截掉，正是要修的毛病。
实测（2793 块语料，multilingual 模型真实分词对照）：ASCII 每 2 字符算
1 token 时，估计值 **100% 不低于**真实分词数；每 3 字符只剩 31% 覆盖。
代价是段数偏多（约 1.8 段/块），冷跑嵌入量约为块数的 1.8 倍。

**为什么不用真 tokenizer**：缓存 key 由切分结果决定，而查缓存必须能在
**不加载模型**的前提下算出 key（9.4 立下的好处：热缓存时模型连加载都
不加载）。真 tokenizer 要先加载模型，这条就没了。估计规则只依赖文本
本身，冷跑热跑两边算出来的一模一样。

**版本**：切法一改就把 `SEGMENT_STRATEGY` 加一版；它和 `TEXT_STRATEGY`
一起拼进缓存命名空间（见 `embedding.cache_for`），旧向量自动作废。
"""

import math

from index.chunks import CodeChunk
from index.compact import TEXT_STRATEGY, compact_text

# 切法版本：改了 split / estimate / 预算规则就 +1。
SEGMENT_STRATEGY = "segment-v1"

# 给特殊 token（[CLS] / [SEP] 之类）留的余量：模型的截断是在含特殊 token
# 的总长上算的，估计值顶到窗口上限时得给它们让位。
SEGMENT_SAFETY_MARGIN = 2

# ASCII 部分每多少字符算 1 token。见模块开头的实测：2 是「估计永不小于
# 真实」的那一档；调小 = 段数更多更安全，调大 = 省段数但可能被截。
ASCII_CHARS_PER_TOKEN = 2


def estimate_tokens(text: str) -> int:
    """估计一段文本占多少 token：汉字 1 字 1 个，其余每 2 个字符算 1 个。

    只求**不小于**真实分词数（宁多勿少），具体见模块开头。空串是 0。
    """
    cjk = sum(1 for char in text if is_cjk(char))
    return cjk + math.ceil((len(text) - cjk) / ASCII_CHARS_PER_TOKEN)


def is_cjk(char: str) -> bool:
    """是不是汉字（CJK 统一表意文字；本项目的中文都在这一段）。"""
    return "一" <= char <= "鿿"


def segment_text(text: str, max_tokens: int) -> tuple[str, ...]:
    """紧凑文本 → 装得进 `max_tokens` 的若干段（确定性）。

    - `max_tokens` 是模型的窗口；预算 = 窗口 - `SEGMENT_SAFETY_MARGIN`。
    - 装得下（含只有一行）就原样返回一段。
    - 第一行当位置行，每一段都带；其余按行装，单行超预算按字符硬切。
    - 空串返回一个空段（保持「一个块至少一段」的形状，调用方不用特判）。
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens 至少为 1，收到 {max_tokens}")
    budget = max(1, max_tokens - SEGMENT_SAFETY_MARGIN)

    if estimate_tokens(text) <= budget:
        return (text,)

    header, body = _split_header(text)
    if estimate_tokens(header) > budget:
        # 位置行自己就超预算：连它一起按字符硬切（极端情况，metadata 保不住）
        return _hard_split(text, budget)

    segments: list[str] = []
    current = header
    has_body = False

    for line in body.split("\n"):
        candidate = f"{current}\n{line}"
        if estimate_tokens(candidate) <= budget:
            current = candidate
            has_body = True
            continue

        if has_body:
            segments.append(current)
        pieces = _fit_line(header, line, budget)
        # 最后一片留作 current，后面的行还能接着往里装；其余的先出
        segments.extend(pieces[:-1])
        current = pieces[-1]
        has_body = True

    segments.append(current)
    return tuple(segments)


def segment_chunk(chunk: CodeChunk, max_tokens: int) -> tuple[str, ...]:
    """一个 chunk 的 embedding 文本 → 若干段：compact_text 之后再切。"""
    return segment_text(compact_text(chunk), max_tokens)


def strategy_id(segmented: bool) -> str:
    """缓存命名空间里的「文本管线版本」：切分开关 + 两个拼法版本。

    分段和不分段是两条管线（向量含义不同），必须落在不同命名空间；切法
    或拼法任一改版，字符串就变，旧向量自动作废。
    """
    if segmented:
        return f"{TEXT_STRATEGY}+{SEGMENT_STRATEGY}"
    return TEXT_STRATEGY


def _split_header(text: str) -> tuple[str, str]:
    """第一行（位置行）和其余部分。没有换行就是「一行 + 空正文」。"""
    first, newline, rest = text.partition("\n")
    return (first, rest) if newline else (first, "")


def _fit_line(header: str, line: str, budget: int) -> list[str]:
    """把一行装进「位置行 + 这一段」的预算里；装不下就按字符硬切。

    返回至少一段，顺序和原文一致。字符切法：预算还剩多少 token，就取
    多少个字符——汉字 1 字 ≤ 1 token、ASCII 2 字 ≤ 1 token，取「剩余
    token 数」个字符永远不会超（保守方向，可能切得碎一点）。
    """
    candidate = f"{header}\n{line}"
    if estimate_tokens(candidate) <= budget:
        return [candidate]

    room = max(1, budget - estimate_tokens(f"{header}\n"))
    pieces: list[str] = []
    remaining = line
    while remaining and estimate_tokens(f"{header}\n{remaining}") > budget:
        take = min(len(remaining), room)
        pieces.append(f"{header}\n{remaining[:take]}")
        remaining = remaining[take:]
    if remaining:
        pieces.append(f"{header}\n{remaining}")
    return pieces or [header]


def _hard_split(text: str, budget: int) -> tuple[str, ...]:
    """整段按字符硬切（位置行自己都超预算时的兜底）。"""
    pieces = []
    remaining = text
    while remaining:
        take = max(1, min(len(remaining), budget))
        pieces.append(remaining[:take])
        remaining = remaining[take:]
    return tuple(pieces)
