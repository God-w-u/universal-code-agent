"""Fake Embedding Interface：给检索接一层语义打分——先用假货把管线跑通。

8.4 说好了「加 embedding 就是再写一个 `(Query, CodeChunk) -> list[Match]`
塞进 Retriever」。这一步真写了一个，但**不是真的 embedding**：没有模型、
没有 API、没有向量库，只有一个 `FakeEmbedder`——一张手写的「同义词小抄」
加上余弦相似度。它只有一个用途：把 hybrid 检索这条管线（接口 → 打分 →
排序 → 评估）整个跑一遍，证明以后真模型接进来时不用改架构。

三层，各管各的：

- `Embedder`：文本 → 向量的接口（`embed(texts) -> 向量们`）。真接模型的
  时候写一个子类实现它，其余一律不动。
- `FakeEmbedder`：假的实现。构造时给一张词表，每一组写着「这些说法是
  一回事」（`("校验", "核对", "verify")`）。它不懂语言，只会拿表里的词
  去文本里做子串查找：撞上哪几组，向量的那几个位置就是 1。
- `EmbeddingScorer`：把 Embedder 包成 retrieval 认的 scorer——相似度和
  阈值比大小，过了就给一条 `Match("embedding", ..., 相似度 × 权重)`。
  加它是可选的：`Retriever(scorers=hybrid_scorers(embedder))` 就是在字面
  三个打分器后面接上它；不加，检索行为和 8.4 一模一样。

几条刻意的边界：

- **只是一张小抄，不是语义**：表里没写的说法它一律不认识，查不到就是查
  不到。真模型的本事是从语料里自己学出这张表（连没写过的说法也认）；
  假货的本事是手抄，抄得对不对全看写表的人。所以它只能验证架构，不能
  拿来回答「语义检索到底有没有用」——那个问题得等真模型。
- **分数照 embedding 的形状来**：相似度是余弦，分数 = 相似度 × 权重，
  `Match.kind` 是 `embedding`——和 8.4 文档里留的形状一致。权重默认和
  「整句正文」同档（40）：语义像，和正文里原样出现一样可信，但都压不过
  名字命中。
- **低于阈值不给分**：没有阈值的话每个 chunk 都会有一点点相似度，
  「查不到」这件事就不存在了。阈值的道理和 8.4 的 `_long_enough` 一样：
  宁可少给，不给凑数的。
- **缓存是接口的一部分**：Retriever 对每个 chunk 都问一遍 scorer，不缓存
  的话一句查询就是几千次模型调用。查询按 `Query` 记；chunk 按
  `(id, digest)` 记——digest 是文件级的，文件一改就重新算，不会拿着旧
  向量给新代码打分。真接模型时这一条就是钱。
- **不碰检索本身**：Retriever、打分表、排序全都没动，这一层只是又一个
  scorer。也不改 Agent / RepairLoop / Verifier / tools。

9.3 给 scorer 加了三个纯加速的口子，**检索结果一个字不动**：

- **compact text**：喂给模型的文本从「路径 + 符号名 + 整段代码」换成
  `compact.py` 拼的紧凑文本（位置行 + docstring + 代码开头）——9.2 量出
  45.7% 的块超窗口被截断，信号还被整段代码稀释。
- **批量**：第一次打分前把全部 chunk 一次性 `prefetch`（构造时给
  `chunks=` 来源）。Retriever 是逐个 chunk 问 scorer 的，不预热就是
  N 个 chunk N 次模型调用；预热后是**一次**。
- **缓存**：`cache=VectorCache(...)` 把向量按 `id|digest` / 查询文本存到
  磁盘（`vector_cache.py`），第二次运行不惊动模型；换模型、文件改动
  自动失效。

9.4 把这条线接进 Agent 的 `search_code_chunks` 时加了两个小件：

- `cache_for(embedder, 目录)`：9.3 的命名空间规矩（模型身份 + 文本拼法
  版本）在这里只写一份，评估入口和工具共用——免得两边各抄一份、哪天
  只改一边。
- `DegradingScorer`：给语义打分器套一层「坏了就当它不存在」。Agent 的
  工具不能因为模型没装 / 下不动 / 缓存坏了就整体失败——降级回字面检索，
  错因记下来给人看。`hybrid_scorers(..., guard=True)` 就是顺手套上它。

9.5 加**分段**（默认开，`segment=False` 关）：embedder 声明了 `max_tokens`
（模型窗口）时，一个 chunk 的紧凑文本按 `segments.py` 切成几段，每段一个
向量，打分取和查询最像的那一段——超窗的代码后半段不再被模型直接截掉。
没声明窗口的 embedder（FakeEmbedder）、或关了开关，走的还是 9.4 的老路
（一个块一个向量）。见 `EmbeddingScorer` 的类文档。

不做真实 embedding API、FAISS、Chroma、Vector DB。
"""

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

from index.chunks import CodeChunk
from index.compact import TEXT_STRATEGY, compact_text
from index.retrieval import DEFAULT_SCORERS, Match, Query, Scorer
from index.segments import segment_chunk, strategy_id
from index.vector_cache import (
    VectorCache,
    cache_file,
    chunk_key,
    namespace_of,
    query_key,
)
from tools.base import ToolError


# 向量：一串浮点数。真的假的都是这个形状——假货短（几个概念维度），真模型
# 长（几百上千维），接口不关心长短。
Vector = tuple[float, ...]

# 语义分和「整句正文」同一档：像，但压不过名字命中。
DEFAULT_SEMANTIC_WEIGHT = 40.0

# 相似度低于这个数就不给 Match。这个数是拍的，先按假货的量纲来；真接
# 模型时得拿真模型的相似度分布重新标一次。
DEFAULT_SEMANTIC_THRESHOLD = 0.35


def cosine(left: Vector, right: Vector) -> float:
    """两个向量的余弦相似度：方向越一致越接近 1。

    有一个是零向量（比如词表里一个词都没撞上）就返回 0.0——不是出错，
    就是「什么都没对上」。长度不一样是接错线了，直接报错。
    """
    if len(left) != len(right):
        raise ToolError(f"两个向量长度不一样：{len(left)} 和 {len(right)}")

    dot = sum(a * b for a, b in zip(left, right))
    left_len = math.sqrt(sum(a * a for a in left))
    right_len = math.sqrt(sum(b * b for b in right))
    if left_len == 0.0 or right_len == 0.0:
        return 0.0
    return dot / (left_len * right_len)


class Embedder(ABC):
    """文本 → 向量的接口。真接 embedding 模型时继承它，只实现这两个。

        class SomeModel(Embedder):
            @property
            def dimension(self): return 1536

            def embed(self, texts):
                return tuple(call_the_api(list(texts)) ...)

    批量进、批量出、顺序一一对应：真模型一次调用能塞一串文本，逐条调是
    最贵的用法。`dimension` 是向量长度，用来挡住「模型换了、向量维度对
    不上」这类接错线。
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量长度。"""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """把一串文本变成等长的向量串，顺序对应。"""

    @property
    def max_tokens(self) -> int | None:
        """模型单次输入的 token 窗口；`None` = 没声明，不切分（9.5）。

        声明了它，`EmbeddingScorer` 就会把过长的紧凑文本切成多段
        （`index/segments.py`），每段单独一个向量、打分取最大相似度——
        模型窗口装不下整块时，代码后半段不再被悄悄截掉。

        **刻意不查模型**：查了就得先加载，而缓存 key 由切分结果决定，
        必须能在不加载模型的前提下算出来（热缓存时模型连加载都不加载）。
        值该是常量：LocalEmbedder 由 provider 按模型名给（见 providers.py）。
        默认 `None`：9.0 起的老形状，一个块一个向量。
        """
        return None


class FakeEmbedder(Embedder):
    """假的 embedding：一张手写的词表，查表出向量。

    每一组词是一个「概念」；向量长度 = 组数，第 i 个位置是 1 表示这组词
    里至少有一个出现在文本里（出现几次都一样）。两个向量越重合，余弦越
    接近 1——「说的是一回事」在它这儿就是「撞上同一组词」。

    匹配规则土得掉渣：全转小写，拿组里的每个词去文本里做**子串**查找。
    英文按词写、中文按说法写，都当子串扫——没有分词器，只能这么撞。
    """

    def __init__(self, aliases: Sequence[Sequence[str]] = ()) -> None:
        groups: list[tuple[str, ...]] = []
        for group in aliases:
            cleaned = tuple(alias.lower() for alias in group)
            if not cleaned or any(not alias.strip() for alias in cleaned):
                raise ToolError("词表里每一组至少要有一个非空的词")
            groups.append(cleaned)
        self._groups = tuple(groups)

    @property
    def dimension(self) -> int:
        return len(self._groups)

    @property
    def aliases(self) -> tuple[tuple[str, ...], ...]:
        """词表本身，留着给人看、给测试查。"""
        return self._groups

    def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        return tuple(self._one(text) for text in texts)

    def _one(self, text: str) -> Vector:
        lowered = text.lower()
        return tuple(
            1.0 if any(alias in lowered for alias in group) else 0.0
            for group in self._groups
        )


# 给例子和 `python -m index --hybrid` 用的小抄：手写的，就这么多。每一组是
# 一行——这些说法是一回事。它是照着自带的 case 和这个项目的常用词挑的，
# 表里没写的说法这假货一律不认识；它证明的是管线通，不是它懂语义。真接模型
# 以后，这张表和 FakeEmbedder 一起退休。
DEMO_ALIASES: tuple[tuple[str, ...], ...] = (
    ("校验", "核对", "验证", "verify", "check"),
    ("改过", "被改", "对不上", "changed"),
    ("文件", "file"),
    ("读回来", "读", "read"),
    ("索引", "index"),
    ("符号", "symbol"),
    ("代码块", "chunk"),
    ("摘要", "指纹", "digest"),
    ("刷新", "重建", "refresh", "rebuild"),
    ("超时", "timeout"),
    ("修复", "repair"),
)


def segmentation_active(embedder: Embedder, segment: bool = True) -> bool:
    """这个 embedder 在这档开关下会不会分段：开关开着、且它声明了窗口。

    判定只看 embedder 声明的 `max_tokens`（常量，不加载模型），所以
    scorer 和 `cache_for` 能各自算出同一答案——缓存命名空间和实际切法
    必须对得上，不然会拿没分段的旧向量当分段结果用。
    """
    return segment and getattr(embedder, "max_tokens", None) is not None


class EmbeddingScorer:
    """把一个 Embedder 包成 retrieval 认的 scorer：加进 scorers 就是 hybrid。

        scorer = EmbeddingScorer(FakeEmbedder(DEMO_ALIASES))
        retriever = Retriever(scorers=DEFAULT_SCORERS + (scorer,))

    形状和别的 scorer 一模一样：`(Query, CodeChunk) -> list[Match]`。相似度
    过了阈值就返回一条 `Match("embedding", 相似度说明, 相似度 × weight)`，
    不然空列表。

    9.3 起多两个可选的加速口子，**给不给都不改检索结果**：

    - `chunks=`：全部 chunk 的来源（序列，或拿回序列的函数——比如绑定的
      `index.chunks`）。给了它，第一次打分前 `prefetch()` 把全部 chunk
      一次批量 embed：模型调用从「N 个 chunk N 次」变成「一次」。不给就
      退回逐条算的老路，结果一样，只是慢。
    - `cache=`：一个 `VectorCache`。chunk 向量按 `id|digest`、查询向量按
      文本存取，命中就不惊动模型。不给也行；读写失败只是没有缓存。

    9.5 起默认多一层**分段**（`segment=False` 关掉）：embedder 声明了
    `max_tokens`（模型窗口，见 `Embedder.max_tokens`）时，过长的紧凑文本
    由 `segments.py` 切成几段，**每段一个向量**，打分取「和查询最像的那
    一段」（max，不取平均——平均会把「有一段很像」稀释成「整体一般像」，
    掉到阈值下面）。一个块只有一个向量时，超窗的部分会被模型悄悄截掉，
    代码后半段等于没喂给模型——分段就是修这个。embedder 没声明窗口
    （比如 FakeEmbedder）、或者 `segment=False`，走的都是 9.4 的老路：
    一个块一个向量，喂的文本一字不差。

    分段之后缓存 key 变成 `id|digest#s0`、`#s1`…（一段一条），命名空间里
    也跟着多一段 `SEGMENT_STRATEGY`（见 `cache_for`）——切法改版、开关
    换档，都落到另一个文件，旧向量不会冒充新结果。分段与否都能算出 key：
    切分只依赖文本和声明的窗口，**不加载模型**（9.4 立下的规矩）。

    统计几个数（`embed_calls` / `texts_embedded` / `cache_hits` /
    `prefetch_seconds` / 分段的 `segment_total` / `segmented_chunks`）只
    记账，打分不看它们；评估入口拿它印耗时。
    """

    kind = "embedding"

    def __init__(
        self,
        embedder: Embedder,
        weight: float = DEFAULT_SEMANTIC_WEIGHT,
        threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        *,
        cache: VectorCache | None = None,
        chunks: Sequence[CodeChunk] | Callable[[], Sequence[CodeChunk]] | None = None,
        segment: bool = True,
    ) -> None:
        if weight <= 0:
            raise ToolError("weight 得是正数：语义分是要加分的")
        if not 0.0 <= threshold <= 1.0:
            raise ToolError("threshold 得在 0~1 之间：它和余弦相似度比大小")
        self.embedder = embedder
        self.weight = float(weight)
        self.threshold = float(threshold)
        self.cache = cache
        self._chunk_source = chunks
        self.segmented = segmentation_active(embedder, segment)
        self._max_tokens = getattr(embedder, "max_tokens", None)
        self._queries: dict[Query, Vector] = {}
        self._chunks: dict[tuple[str, str], tuple[Vector, ...]] = {}
        self._warmed = False
        self.embed_calls = 0
        self.texts_embedded = 0
        self.cache_hits = 0
        self.prefetch_seconds = 0.0
        self.segment_total = 0
        self.segmented_chunks = 0

    @property
    def max_tokens(self) -> int | None:
        """切分用的窗口（embedder 声明的那个）；没分段就是 None。"""
        return self._max_tokens

    def __call__(self, query: Query, chunk: CodeChunk) -> list[Match]:
        self._warm()
        query_vector = self._query_vector(query)
        similarity = max(
            cosine(query_vector, vector) for vector in self._chunk_vectors(chunk)
        )
        # 零相似就是什么都没对上；threshold 给到 0 也不给这种「0 分结果」。
        if similarity <= 0.0 or similarity < self.threshold:
            return []
        return [Match(self.kind, f"语义相近 {similarity:.2f}", similarity * self.weight)]

    def prefetch(self, chunks=None) -> int:
        """把还没算过的 chunk 向量一次批量算齐，返回这次新嵌了几段文本。

        `chunks` 不给就用构造时给的来源；两边都没有就什么都不做（老路是
        `_chunk_vectors` 逐条算）。顺序：内存 → 磁盘缓存 → 模型——只有在
        最后一步才调用 `embed()`，而且只调一次：分段后所有块的所有段拼进
        同一批，段数多了也不会变成 N 次调用。
        """
        source = self._chunk_source if chunks is None else chunks
        if source is None:
            return 0
        items = tuple(source() if callable(source) else source)
        self._warmed = True

        started = time.perf_counter()
        pending = [
            chunk
            for chunk in items
            if (chunk.id, chunk.digest) not in self._chunks
        ]
        texts = {self._identity(chunk): self._texts_for(chunk) for chunk in pending}
        self.segment_total += sum(len(segment) for segment in texts.values())
        self.segmented_chunks += sum(
            1 for segment in texts.values() if len(segment) > 1
        )

        known: dict[tuple[str, str], dict[int, Vector]] = {}
        if self.cache is not None and pending:
            by_key = {
                self._cache_key(chunk, index): (self._identity(chunk), index)
                for chunk in pending
                for index in range(len(texts[self._identity(chunk)]))
            }
            for key, vector in self.cache.get_many(by_key).items():
                identity, index = by_key[key]
                known.setdefault(identity, {})[index] = vector
                self.cache_hits += 1

        missing = [
            (chunk, index, text)
            for chunk in pending
            for index, text in enumerate(texts[self._identity(chunk)])
            if index not in known.get(self._identity(chunk), {})
        ]
        embedded = self._embed_many([text for _, _, text in missing])

        if self.cache is not None and missing:
            # 批量一次写盘；再攒下的（查询向量）由 flush_cache() 收尾
            self.cache.put_many(
                {
                    self._cache_key(chunk, index): vector
                    for (chunk, index, _), vector in zip(missing, embedded)
                }
            )

        for chunk in pending:
            identity = self._identity(chunk)
            merged = dict(known.get(identity, {}))
            for (other, index, _), vector in zip(missing, embedded):
                if self._identity(other) == identity:
                    merged[index] = vector
            self._chunks[identity] = tuple(
                merged[index] for index in range(len(texts[identity]))
            )
        # 顺手打扫：这份语料里已经查不到的旧向量（9.6）
        self._prune_stale_vectors(items)
        self.prefetch_seconds += time.perf_counter() - started
        return len(embedded)

    def _prune_stale_vectors(self, items) -> None:
        """把这份语料里已经查不到的旧向量从缓存清掉（9.6）。

        digest 一改，老向量就永远查不到；分段后 key 还带段号，段数一变
        也会留下够不着的旧段——留着只占地方。`scope` 卡在本次语料自己的
        chunk id 上：评估入口里项目 + C++ fixture 共用一份缓存文件，两个
        scorer 各扫各的，谁也不动谁；查询向量由 `prune` 自己保证不碰。
        缓存坏了 / 写不进去，prune 和老规矩一样只降级不炸。
        """
        if self.cache is None:
            return
        self.cache.prune(
            (
                self._cache_key(chunk, index)
                for chunk in items
                for index in range(len(self._chunks[self._identity(chunk)]))
            ),
            scope={chunk.id for chunk in items},
        )

    def flush_cache(self) -> None:
        """把攒着没落盘的向量写掉（没有缓存 / 没攒着 / 写不成都不报错）。"""
        if self.cache is not None:
            self.cache.flush()

    def stats_line(self) -> str:
        """一行统计，给评估入口印着看。"""
        return combined_stats_line((self,))

    def _warm(self) -> None:
        """第一次打分前把批量预热做掉（只做一次）。"""
        if not self._warmed:
            self.prefetch()

    def _query_vector(self, query: Query) -> Vector:
        vector = self._queries.get(query)
        if vector is None:
            vector = self._from_cache(query_key(query.text))
            if vector is None:
                vector = self._embed_many((query.text,))[0]
                self._remember(query_key(query.text), vector)
            self._queries[query] = vector
        return vector

    def _chunk_vectors(self, chunk: CodeChunk) -> tuple[Vector, ...]:
        """一个块的全部向量（没分段就一个），顺序即段序。

        老路（没预热时逐条算）：内存 → 缓存 → 模型；一个块缺几段就补几段，
        一笔批量里补齐。
        """
        identity = self._identity(chunk)
        vectors = self._chunks.get(identity)
        if vectors is not None:
            return vectors

        texts = self._texts_for(chunk)
        by_index: dict[int, Vector] = {}
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cached = self._from_cache(self._cache_key(chunk, index))
            if cached is None:
                missing.append((index, text))
            else:
                by_index[index] = cached

        if missing:
            embedded = self._embed_many([text for _, text in missing])
            for (index, _), vector in zip(missing, embedded):
                by_index[index] = vector
                self._remember(self._cache_key(chunk, index), vector)

        vectors = tuple(by_index[index] for index in range(len(texts)))
        self._chunks[identity] = vectors
        return vectors

    def _texts_for(self, chunk: CodeChunk) -> tuple[str, ...]:
        """这个块要喂给模型的文本：分段时是切好的几段，否则就是整段紧凑文本。"""
        if not self.segmented or self._max_tokens is None:
            return (compact_text(chunk),)
        return segment_chunk(chunk, self._max_tokens)

    def _identity(self, chunk: CodeChunk) -> tuple[str, str]:
        """内存里的块身份：和磁盘 key 同一个意思（id | digest）。"""
        return (chunk.id, chunk.digest)

    def _cache_key(self, chunk: CodeChunk, index: int) -> str:
        """一段文本的缓存 key：分段时 `id|digest#s0`，否则还是 9.4 的形状。"""
        key = chunk_key(chunk)
        return f"{key}#s{index}" if self.segmented else key

    def _from_cache(self, key: str) -> Vector | None:
        if self.cache is None:
            return None
        vector = self.cache.get(key)
        if vector is None:
            return None
        self.cache_hits += 1
        return vector

    def _remember(self, key: str, vector: Vector) -> None:
        """单条往缓存里记：先攒在内存（整份重写不便宜），收尾 `flush_cache()`。"""
        if self.cache is not None:
            self.cache.put_many({key: vector}, flush=False)

    def _embed_many(self, texts) -> tuple[Vector, ...]:
        """一次 embed 一批，顺序一一对应；顺手挡住两种接错线：条数不对、
        长度不对。空批次直接回空——不为空批次惊动模型。"""
        batch = tuple(texts)
        if not batch:
            return ()
        vectors = tuple(
            tuple(float(value) for value in vector)
            for vector in self.embedder.embed(batch)
        )
        if len(vectors) != len(batch):
            raise ToolError(
                f"embedder 收到 {len(batch)} 条文本，却回了 {len(vectors)} 个向量"
            )
        dimension = self.embedder.dimension
        for vector in vectors:
            if len(vector) != dimension:
                raise ToolError(
                    f"embedder 说向量长 {dimension}，实际回了 {len(vector)}：接错模型了？"
                )
        self.embed_calls += 1
        self.texts_embedded += len(batch)
        return vectors


def combined_stats_line(scorers: Sequence[EmbeddingScorer]) -> str:
    """几个 scorer 的统计合成一行（9.5）。

    多语料时（项目 + C++ fixture）每个语料一个 scorer、各记各的账，报告里
    要的是这组合起来的一行——不然「嵌入了几条」会被拆成两行，没法跟单语料
    的 9.4 直接比。单语料就是 `EmbeddingScorer.stats_line()`。
    """
    scorers = tuple(scorers)
    embed_calls = sum(scorer.embed_calls for scorer in scorers)
    texts = sum(scorer.texts_embedded for scorer in scorers)
    hits = sum(scorer.cache_hits for scorer in scorers)
    seconds = sum(scorer.prefetch_seconds for scorer in scorers)
    line = (
        f"[embedding] embed() 调用 {embed_calls} 次；"
        f"嵌入 {texts} 条文本；"
        f"缓存命中 {hits} 条；"
        f"批量预热 {seconds:.1f} s"
    )
    segmented = [scorer for scorer in scorers if scorer.segmented]
    if segmented:
        # 同一组里几个语料共用同一个 embedder，窗口必然一致，取第一个说就行
        line += (
            f"；分段 {sum(s.segment_total for s in segmented)} 段"
            f"（其中 {sum(s.segmented_chunks for s in segmented)} 块切成多段，窗口 "
            f"{segmented[0].max_tokens} token）"
        )
    return line


class DegradingScorer:
    """把 scorer 包成「坏了就当它不存在」：检索永不因它失败（9.4）。

    Agent 的 `search_code_chunks` 里，embedding 是锦上添花——模型没装 /
    没下动 / 跑起来报错时，整次检索不能跟着失败，得安安静静退回字面。
    包一层就够：第一次出错（不管是查询还是 chunk 那一步）就永久关上
    自己，之后每条都返回空命中，剩下的字面 scorer 照常打分；错因记在
    `error` 里，工具转出去给人看。

    刻意**不重试**：模型加载失败重试一次多半还是失败，而每次重试都要
    先等一遍超时——宁可当场降级。`disable()` 是给外面的开口，比如批量
    预热那一步失败了（它的失败发生在 scorer 之外），手动关上。
    """

    def __init__(self, scorer: Scorer) -> None:
        self.scorer = scorer
        self.error: str | None = None

    @property
    def disabled(self) -> bool:
        return self.error is not None

    def disable(self, reason: str) -> None:
        """记下第一个错因；已经关上了就不再改（第一现场最有诊断价值）。"""
        if self.error is None:
            self.error = reason

    def __call__(self, query: Query, chunk: CodeChunk) -> list[Match]:
        if self.disabled:
            return []
        try:
            return self.scorer(query, chunk)
        except Exception as exc:  # 模型 / 缓存 / 维度对不上，一律降级
            self.disable(f"{type(exc).__name__}: {exc}")
            return []


def hybrid_scorers(
    embedder: Embedder,
    weight: float = DEFAULT_SEMANTIC_WEIGHT,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    *,
    cache: VectorCache | None = None,
    chunks: Sequence[CodeChunk] | Callable[[], Sequence[CodeChunk]] | None = None,
    guard: bool = False,
    segment: bool = True,
) -> tuple[Scorer, ...]:
    """字面三个打分器 + 一个语义打分器，直接喂给 Retriever。

        Retriever(root=..., scorers=hybrid_scorers(FakeEmbedder(DEMO_ALIASES)))

    `cache` / `chunks` 是 9.3 的加速口子，原样转给 `EmbeddingScorer`；
    不传就是 9.0 的行为。`guard=True`（9.4）再套一层 `DegradingScorer`：
    语义分出了任何问题都自动当它不存在——Agent 的工具用这个形状。
    `segment=False`（9.5）关掉分段，复现 9.4 的「一个块一个向量」，
    用来做前后对比；默认开，embedder 没声明窗口时本来也不切。
    """
    scorer: Scorer = EmbeddingScorer(
        embedder,
        weight=weight,
        threshold=threshold,
        cache=cache,
        chunks=chunks,
        segment=segment,
    )
    if guard:
        scorer = DegradingScorer(scorer)
    return DEFAULT_SCORERS + (scorer,)


def cache_for(
    embedder: Embedder, directory: Path, *, segmented: bool = True
) -> VectorCache:
    """按 9.3 的命名空间规矩给 embedder 配一个落盘缓存（9.4 / 9.5）。

    命名空间 = 模型身份（`namespace_of`：类名 + 模型名 / 词表指纹）+ 文本
    管线版本：不分段是 `compact-v1`，分段是 `compact-v1+segment-v1`
    （`segments.strategy_id`）——换模型、改 compact 拼法、改切法、换开关，
    都落到另一个文件，旧向量自动作废。评估入口和 Agent 工具共用这一处
    规则，`segmented` 要和给 `EmbeddingScorer` 的 `segment` 一致。
    """
    cutting = segmentation_active(embedder, segmented)
    namespace = f"{namespace_of(embedder)}#{strategy_id(cutting)}"
    return VectorCache(cache_file(directory, namespace), namespace)
