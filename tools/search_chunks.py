"""工具：按查询检索最相关的代码块，只给位置，不给正文。

检索本身全在 index 那条线里（字面打分、8.7 的粒度与整名排序），这里只做
两件事：把查询交给 Retriever，把每条结果转成一行元信息——路径、符号、
起止行号、类型、分数、为什么命中。

13.6 起可以显式刷新（`refresh()`）：文件增删改之后调它，索引整个重扫，
下一次检索用的就是最新代码，删掉的文件不留下任何 chunk；语义那一侧同时
把「连 id 都没了的块」的向量从内存和磁盘缓存里清掉，下一次检索按新索引
重新预热。

刻意不返回代码正文：正文会把上下文撑爆，而模型拿到「路径:起止行号」用
read_file 自己就能读原文——返回位置比返回内容更省；而且它读到的永远是
磁盘上当前的代码，不是索引里缓存的那份副本。

9.4 起可以选配 embedding（语义检索），四种模式 `off / fake / local /
multilingual`，默认 `off`——不选就是 8.8 起的老行为，一个字节不差：

- **复用 9.3 的整条加速管线**：紧凑文本、一次批量预热、用户目录下的
  向量缓存，全走 `index` 里现成的件，这里不重写一套。
- **模型是懒加载的**：构造工具、注册工具都不碰模型；第一次真正检索
  才加载（见 `_prepare`）。
- **坏了自己退**：模型没装 / 下不动 / 缓存炸了，一律降级回字面检索，
  工具照常出结果——语义分是锦上添花，不该把检索拖下水。
- **并发只准备一次**（13.3）：工具实例被各会话共用，第一次检索可能
  同时进来——建索引和预热收在同一把锁里，只做一遍。
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tools.base import Tool, ToolError
from tools.paths import check_root

if TYPE_CHECKING:
    # 只在做类型检查时导入，运行时不导——index 自己也 import tools.base，
    # 模块顶层真去 import index 就会和 tools/__init__.py 撞成循环（见 __init__）。
    # DegradingScorer 不在这一串里：__init__ 里那句 from index import 已经把它
    # 绑成局部名字了，注解用的就是那一个
    from index import Embedder, EmbeddingScorer, SearchResult, VectorCache

# 默认模式：纯字面。off 不在 index.providers 里——它的意思是「这条线
# 根本不接」，不是某个 provider。
OFF_MODE = "off"


class SearchChunksTool(Tool):
    """按查询找代码块（类 / 函数 / 方法），返回位置和分数。"""

    name = "search_code_chunks"
    description = (
        "用一句查询找出项目里最相关的代码块（类 / 函数 / 方法），按相关度排序；"
        "每条给出「相对路径:起止行号、符号名、类型、分数」和命中原因，不返回代码正文。"
        "查询可以是限定名（Agent.run）、简单名（run）、文件名（client.py）或普通关键词。"
        "想知道「某个功能在哪儿实现」用它比 search_code 直接：search_code 只按关键词找"
        "单行，这个会综合符号名、路径、正文排序，并把整段（整个方法 / 类）作为一条给出。"
        "拿到结果后用 read_file 按行号去读原文。"
    )
    parameters = {
        "query": {
            "type": "string",
            "description": "要查的内容：符号名、文件名或关键词",
            "required": True,
        },
        "top_k": {
            "type": "integer",
            "description": "最多返回几条，省略默认 10 条",
            "required": False,
        },
    }

    # top_k 的上限：一条结果就是一行，50 行已经够挑了，再多只是往上下文里灌位置
    max_top_k = 50

    def __init__(
        self,
        root: Path | None = None,
        embedding: str = OFF_MODE,
        *,
        embedder: "Embedder | None" = None,
        cache: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        """root：项目根；`embedding`：`off`（默认）/ `fake` / `local` / `multilingual`。

        后三个就是 `index.providers` 里的 provider 名字，由它去造 embedder
        ——这里不认识任何模型。构造**不加载模型**：加载推迟到第一次真正
        检索（和 `LocalEmbedder` 的懒加载是一回事）。

        `embedder=` 是给测试的注入口：模式照选，但用传进来的对象代替
        按名字造的那个。`cache=False` 不落盘（测试默认走这条）；
        `cache_dir=` 换缓存目录，默认和评估入口同一个位置（用户目录下，
        不在项目里、不会被索引扫到）。
        """
        self.root = check_root(root)

        # 到这儿才 import index，而不是模块顶层：tools/__init__.py 会把这个
        # 模块连同所有工具一起拉起来，而 index 自己也 import tools.base——
        # 顶层 import 会让「先 import index」的那条路撞上只加载了一半的
        # index（循环 import）。走到构造函数时两边都已加载完，怎么进来都不会炸。
        from index import (
            DEFAULT_TOP_K,
            PROVIDERS,
            ChunkIndex,
            DegradingScorer,
            Retriever,
            cache_for,
            create_embedder,
            default_cache_dir,
            hybrid_scorers,
        )

        self.index = ChunkIndex(root=self.root)
        self.default_top_k = DEFAULT_TOP_K

        mode = embedding.strip().lower()
        valid = (OFF_MODE,) + tuple(PROVIDERS)
        if mode not in valid:
            raise ToolError(
                f"不认识的 embedding 模式：{embedding!r}（可选：{'、'.join(valid)}）"
            )

        # off 的三个属性都是 None：下面 _prepare / _flush_embedding 靠它
        # 认路，测试也拿它区分「没开」和「开了」
        self.scorer: EmbeddingScorer | None = None  # 批量预热 / 统计用
        self.guard: DegradingScorer | None = None  # 降级状态看它
        self.vector_cache: VectorCache | None = None
        self._warmed = False
        # 13.3：工具实例被各会话共用——建索引 + 预热收在这一把锁里
        # （见 _prepare）
        self._prepare_lock = threading.Lock()

        if mode == OFF_MODE:
            if embedder is not None:
                raise ToolError(
                    "embedding 模式是 off，却又给了 embedder："
                    "要注入口就选一个非 off 的模式"
                )
            self.mode = OFF_MODE
            self.embedder = None
            # 老样子：不传 scorers，Retriever 用默认的字面那三个
            self.retriever = Retriever(index=self.index)
            return

        self.mode = mode
        self.embedder = create_embedder(mode) if embedder is None else embedder
        if cache:
            directory = default_cache_dir() if cache_dir is None else Path(cache_dir)
            self.vector_cache = cache_for(self.embedder, directory)

        scorers = hybrid_scorers(
            self.embedder,
            cache=self.vector_cache,
            # 批量预热的来源：第一次检索前把全部 chunk 一次算齐（9.3）。
            # 传的是绑定方法，prefetch 那一刻索引已经建好（见 _prepare）。
            chunks=self.index.chunks,
            # 出问题自动退回字面，别让检索跟着失败（9.4）
            guard=True,
        )
        self.guard = next(s for s in scorers if isinstance(s, DegradingScorer))
        # DegradingScorer.scorer 的声明类型是通用的 Scorer 可调用对象（它谁都
        # 能包）；这里包的一定是 EmbeddingScorer——hybrid_scorers 造的——预热
        # 和清缓存三个口子（prefetch / forget / flush_cache）只有它有
        self.scorer = cast("EmbeddingScorer | None", self.guard.scorer)
        self.retriever = Retriever(index=self.index, scorers=scorers)

    @property
    def embedding_error(self) -> str | None:
        """语义检索降级了没有：没降级是 None，降了是第一个错因。"""
        return None if self.guard is None else self.guard.error

    def execute(self, query: str, top_k: int | None = None) -> str:
        """查一句，返回最多 top_k 行位置信息；查不到就明说查不到。"""
        if not query.strip():
            raise ToolError("查询不能为空")

        asked = self.default_top_k if top_k is None else top_k
        if asked < 1:
            raise ToolError(f"top_k 至少为 1，收到 {top_k}")

        self._prepare()
        try:
            results = self.retriever.search(query, top_k=min(asked, self.max_top_k))
        finally:
            # 查询向量是攒在内存里的：不管这次查成没查成，都落一次盘
            self._flush_embedding()

        if not results:
            return f"（没有找到和 {query!r} 匹配的代码块）"

        text = self._render(results)
        if asked > self.max_top_k:
            # 截断了就得说：不说的话模型分不清「就这么多」和「只给了前几条」
            text += f"\n…（top_k 最多给到 {self.max_top_k}）"
        return text

    def refresh(self) -> None:
        """显式刷新索引：文件增、删、改之后调它，下次检索就用最新的代码。

        全量重建（不是增量）：删掉的文件不会留下 chunk，改过的文件行号、
        digest 都是新的，新加的文件也在。语义模式还要多做一步——**把连
        id 都没了的块的向量清掉**（已删的文件、没了的符号；id 自己都没了，
        磁盘上那些条目不在任何 prune 的 scope 里，不清就是永远查不到的
        残留）。改过内容但 id 没变的块不用管：key 里带着 digest，旧向量
        查不到，接着的预热会顺手把过时的那几条从缓存里打扫掉。

        刷新不动模型的可用状态：已经降级（模型坏了）的工具不会借这次刷新
        重试模型——那是「一次为限，失败不重试」那条规矩。字面检索照常。

        和 _prepare 共用一把锁：多会话同用一个工具实例时，刷新和第一次
        准备不会撞在一起。
        """
        with self._prepare_lock:
            before = (
                {chunk.id for chunk in self.index.chunks()}
                if self.index.built
                else set()
            )
            self.index.refresh()
            # 这两个属性要么都是 None（off 模式），要么都建好了——一起判，
            # 免得只认一个、另一个撞上 None
            if self.scorer is None or self.guard is None:
                return
            self.scorer.forget(before - {chunk.id for chunk in self.index.chunks()})
            if not self.guard.disabled:
                # 下一次检索重新预热（用新索引的 chunk）；降级过的就维持原样
                self._warmed = False

    def _prepare(self) -> None:
        """检索前把索引和语义分准备好（一次为限）。

        索引：off 模式下原来由 Retriever.search 懒建，提前到这里——和
        语义模式的建索引合成同一步。先建索引，批量预热的来源是 chunks，
        空索引等于什么都没预热。

        预热：让 scorer 一次批量算齐全部 chunk 向量；**模型就是在这一步
        加载的**，构造和注册都不碰它。

        多会话共用一个工具实例（13.3）：先无锁快查（都齐了直接过），没齐
        才进锁，进锁后再查一次——同时进来的第一次检索只有一个人在准备。
        预热是「一次为限，失败不重试」：_warmed 进锁后立刻立起来，出的
        任何问题（没装库 / 模型下不动 / 缓存坏了）都不往外抛，只关掉降级
        开关，检索退回字面，工具照常出结果。建索引失败不在此列——那是
        baseline 也会遇到的整体失败，不该假装降级成功。
        """
        if self.index.built and (self.scorer is None or self._warmed):
            return

        with self._prepare_lock:
            if not self.index.built:
                self.index.build()

            if self.scorer is None or self.guard is None or self._warmed:
                return
            self._warmed = True

            try:
                self.scorer.prefetch()
            except Exception as exc:
                self.guard.disable(f"{type(exc).__name__}: {exc}")

    def _flush_embedding(self) -> None:
        """把攒着的查询向量落盘；没新的、没缓存、写不成，都什么都不做。"""
        if self.scorer is not None:
            self.scorer.flush_cache()

    def _render(self, results: "tuple[SearchResult, ...]") -> str:
        """每条一行：位置 + 符号 + 类型 + 分数 + 命中原因。

        `reason` 是分数的拆解（名字命中还是正文命中），留着它模型才知道
        同样 40 分的两条为什么不同——它引用的都是查询里的词和符号名，
        不会把代码正文漏出来。
        """
        return "\n".join(
            f"{position}. {item.chunk.path}:{item.chunk.start_line}-{item.chunk.end_line}"
            f"  {item.chunk.symbol}  [{item.chunk.kind}]  {item.score:g}"
            f"  {item.reason}"
            for position, item in enumerate(results, start=1)
        )
