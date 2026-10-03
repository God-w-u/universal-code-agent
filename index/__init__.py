"""代码索引与检索：文件清单（Project Index）→ 符号位置（Symbol Index）→
代码块（Code Chunk）→ 检索（Code Retrieval）→ 评估（Retrieval Evaluation）。

前三层是地基：ProjectIndex 管项目里有哪些文件，SymbolIndex 管每个文件里
有哪些类 / 函数 / 方法，ChunkIndex 按符号边界把真实代码切出来。第四层
Retriever 才是入口——拿一句查询，在这些 chunk 上做符号 / 路径 / 正文的
字面匹配，排序后返回。第五层 Evaluator 拿一组手写 case 给检索打分
（Hit@1/3/5 + 平均耗时），以后换 embedding 就靠这套数字比好坏。
embedding.py 是第四层的可选扩展：Embedder 接口 + 一个假实现（FakeEmbedder）
+ EmbeddingScorer——先用假货把 hybrid 检索的管线跑通。9.0 起真模型从同一个
接口接进来：local_embedding.py 包一个本地 sentence-transformers 模型
（可选依赖、惰性加载），providers.py 是「名字 → Embedder」的工厂。
9.3 的加速件：compact.py 拼喂给模型的紧凑文本；vector_cache.py 把向量
存到用户目录下的 JSON（换模型 / 文件改动自动失效）；EmbeddingScorer
支持一次批量预热，不再 N 个 chunk N 次模型调用。9.4 起 Agent 的
search_code_chunks 也能选配这条线（off / fake / local / multilingual，
默认 off），embedding 出问题由 DegradingScorer 兜住、自动退回字面。
9.5 的 segments.py 把过长的紧凑文本切成装得进模型窗口的几段，一个块可以有
多个向量、打分取最像的那一段——模型窗口装不下的代码后半段不再被悄悄截掉；
评估集同时扩成多语言 / 多查询类型 / 多语料（含一个 C++ fixture）并加了
分组统计。给后面的 Hybrid Search / Code RAG 打底。
不含向量库（FAISS / Chroma 这类），也不改 Agent 的行为。
"""

from index.chunks import ChunkIndex, CodeChunk
from index.compact import (
    MAX_CODE_CHARS,
    MAX_DOC_CHARS,
    TEXT_STRATEGY,
    compact_text,
)
from index.embedding import (
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_SEMANTIC_WEIGHT,
    DEMO_ALIASES,
    DegradingScorer,
    Embedder,
    EmbeddingScorer,
    FakeEmbedder,
    Vector,
    cache_for,
    combined_stats_line,
    cosine,
    hybrid_scorers,
    segmentation_active,
)
from index.evaluation import (
    DEFAULT_CORPUS,
    GROUP_LABELS,
    EvaluationCase,
    EvaluationReport,
    EvaluationResult,
    Evaluator,
    load_cases,
)
from index.languages import UNKNOWN_LANGUAGE, language_of
from index.local_embedding import (
    DEFAULT_LOCAL_MAX_TOKENS,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_MULTILINGUAL_MAX_TOKENS,
    DEFAULT_MULTILINGUAL_MODEL,
    EmbeddingUnavailable,
    LocalEmbedder,
)
from index.project import (
    BINARY,
    INDEXED,
    TOO_LARGE,
    UNREADABLE,
    IndexedFile,
    ProjectIndex,
    digest_of,
)
from index.providers import PROVIDERS, create_embedder
from index.retrieval import (
    DEFAULT_SCORERS,
    DEFAULT_TOP_K,
    MAX_CODE_HITS,
    SCORE_CODE_HIT,
    SCORE_SPECIFIC,
    SCORE_SYMBOL_WORD,
    Match,
    Query,
    Retriever,
    Scorer,
    SearchResult,
)
from index.segments import (
    SEGMENT_SAFETY_MARGIN,
    SEGMENT_STRATEGY,
    estimate_tokens,
    segment_chunk,
    segment_text,
    strategy_id,
)
from index.symbols import (
    CHANGED,
    CLASS,
    FUNCTION,
    METHOD,
    STRUCT,
    UNPARSABLE,
    Skipped,
    Symbol,
    SymbolIndex,
)
from index.vector_cache import (
    CACHE_VERSION,
    VectorCache,
    cache_file,
    chunk_key,
    default_cache_dir,
    namespace_of,
    query_key,
)

__all__ = [
    "ProjectIndex",
    "IndexedFile",
    "INDEXED",
    "BINARY",
    "TOO_LARGE",
    "UNREADABLE",
    "digest_of",
    "SymbolIndex",
    "Symbol",
    "Skipped",
    "CLASS",
    "STRUCT",
    "FUNCTION",
    "METHOD",
    "CHANGED",
    "UNPARSABLE",
    "ChunkIndex",
    "CodeChunk",
    "Embedder",
    "FakeEmbedder",
    "LocalEmbedder",
    "EmbeddingUnavailable",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_MULTILINGUAL_MODEL",
    "create_embedder",
    "PROVIDERS",
    "EmbeddingScorer",
    "DegradingScorer",
    "hybrid_scorers",
    "cache_for",
    "combined_stats_line",
    "segmentation_active",
    "DEMO_ALIASES",
    "DEFAULT_SEMANTIC_WEIGHT",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "Vector",
    "cosine",
    "compact_text",
    "MAX_DOC_CHARS",
    "MAX_CODE_CHARS",
    "TEXT_STRATEGY",
    "segment_text",
    "segment_chunk",
    "estimate_tokens",
    "strategy_id",
    "SEGMENT_STRATEGY",
    "SEGMENT_SAFETY_MARGIN",
    "DEFAULT_LOCAL_MAX_TOKENS",
    "DEFAULT_MULTILINGUAL_MAX_TOKENS",
    "VectorCache",
    "CACHE_VERSION",
    "cache_file",
    "chunk_key",
    "query_key",
    "namespace_of",
    "default_cache_dir",
    "Retriever",
    "SearchResult",
    "Match",
    "Query",
    "Scorer",
    "DEFAULT_SCORERS",
    "DEFAULT_TOP_K",
    "SCORE_SPECIFIC",
    "SCORE_CODE_HIT",
    "SCORE_SYMBOL_WORD",
    "MAX_CODE_HITS",
    "Evaluator",
    "EvaluationCase",
    "EvaluationResult",
    "EvaluationReport",
    "load_cases",
    "DEFAULT_CORPUS",
    "GROUP_LABELS",
    "language_of",
    "UNKNOWN_LANGUAGE",
]
