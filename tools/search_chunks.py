"""工具：按查询检索最相关的代码块，只给位置，不给正文。

检索本身全在 index 那条线里（字面打分、8.7 的粒度与整名排序），这里只做
两件事：把查询交给 Retriever，把每条结果转成一行元信息——路径、符号、
起止行号、类型、分数、为什么命中。

刻意不返回代码正文：正文会把上下文撑爆，而模型拿到「路径:起止行号」用
read_file 自己就能读原文——返回位置比返回内容更省；而且它读到的永远是
磁盘上当前的代码，不是索引里缓存的那份副本。

9.4 起可以选配 embedding（语义检索），四种模式 `off / fake / local /
multilingual`，默认 `off`——不选就是 8.8 起的老行为，一个字节不差：

- **复用 9.3 的整条加速管线**：紧凑文本、一次批量预热、用户目录下的
  向量缓存，全走 `index` 里现成的件，这里不重写一套。
- **模型是懒加载的**：构造工具、注册工具都不碰模型；第一次真正检索
  才加载（见 `_prepare_embedding`）。
- **坏了自己退**：模型没装 / 下不动 / 缓存炸了，一律降级回字面检索，
  工具照常出结果——语义分是锦上添花，不该把检索拖下水。
"""

from pathlib import Path
from typing import TYPE_CHECKING

from tools.base import Tool, ToolError
from tools.paths import check_root

if TYPE_CHECKING:
    # 只在做类型检查时导入，运行时不导——index 自己也 import tools.base，
    # 模块顶层真去 import index 就会和 tools/__init__.py 撞成循环（见 __init__）。
    from index import Embedder, SearchResult

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

        # off 的三个属性都是 None：下面 _prepare_embedding / _flush_embedding
        # 靠它认路，测试也拿它区分「没开」和「开了」
        self.scorer = None  # EmbeddingScorer：批量预热 / 统计用
        self.guard = None  # DegradingScorer：降级状态看它
        self.vector_cache = None
        self._warmed = False

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
            # 传的是绑定方法，prefetch 那一刻索引已经建好（见 _prepare_embedding）。
            chunks=self.index.chunks,
            # 出问题自动退回字面，别让检索跟着失败（9.4）
            guard=True,
        )
        self.guard = next(s for s in scorers if isinstance(s, DegradingScorer))
        self.scorer = self.guard.scorer
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

        self._prepare_embedding()
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

    def _prepare_embedding(self) -> None:
        """第一次真要用语义分时，把 embedding 准备好（只做一次）。

        先建索引——批量预热的来源是 chunks，空索引等于什么都没预热；再让
        scorer 一次批量算齐全部 chunk 向量。**模型就是在这一步加载的**，
        构造和注册都不碰它。

        这一步出的任何问题（没装库 / 模型下不动 / 缓存坏了）都不往外抛：
        关掉降级开关，检索退回字面，工具照常出结果。建索引失败不在此列
        ——那是 baseline 也会遇到的整体失败，不该假装降级成功。
        """
        if self.scorer is None or self._warmed:
            return
        self._warmed = True

        if not self.index.built:
            self.index.build()

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
