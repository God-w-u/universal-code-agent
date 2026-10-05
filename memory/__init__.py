"""Memory：任务历史的持久化、检索与 Agent 集成（Phase 11）。

- base：MemoryRecord（id / timestamp / task / summary / outcome /
  metadata）+ MemoryStore 抽象 + 摘要裁剪规则。
- store：LocalMemoryStore —— 本地一个 JSONL 文件，追加即写、坏行跳过、
  残句不连累下一条；默认路径在用户目录下（项目之外）。
- retriever：MemoryRetriever —— 确定性的轻量检索：token 匹配、
  top_k、summary 上限，不接 embedding。
- context（11.2）：MemoryContext —— Agent 一轮任务前后各问一次的那层：
  before() 检索出「仅供参考」的 system 消息，after() 存一条最终记录；
  两个方向都失败降级，绝不影响任务本身。
- maintenance（11.3）：清理与统计 —— 上限（最多留几条 / 文件多大）、
  去重（三元组完全一样只留最新）、prune 的算账逻辑、stats。真正重写
  文件的是 LocalMemoryStore.prune / compact（临时文件 + 换名）。

**给模型看的唯一入口是 `MemoryContext.before`（内部走 retriever）**：
它只返回有限的几条、每条都裁过，还带着抬头；`store.load()` / `stats()`
读的是全部历史，那是给测试 / 统计 / 维护用的，别直接塞进 context。
"""

from memory.base import (
    FAILURE,
    OUTCOMES,
    SUCCESS,
    MemoryRecord,
    MemoryStore,
    clip_text,
)
from memory.context import (
    MAX_STORED_SUMMARY_CHARS,
    MAX_TASK_CHARS,
    MemoryContext,
    render,
)
from memory.maintenance import (
    DEFAULT_LIMITS,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_RECORDS,
    MemoryLimits,
    MemoryStats,
    PruneResult,
)
from memory.retriever import (
    DEFAULT_TOP_K,
    MAX_SUMMARY_CHARS,
    MemoryHit,
    MemoryRetriever,
    tokenize,
)
from memory.store import DEFAULT_FILENAME, LocalMemoryStore, default_memory_path

__all__ = [
    "SUCCESS",
    "FAILURE",
    "OUTCOMES",
    "MemoryRecord",
    "MemoryStore",
    "clip_text",
    "LocalMemoryStore",
    "DEFAULT_FILENAME",
    "default_memory_path",
    "MemoryHit",
    "MemoryRetriever",
    "DEFAULT_TOP_K",
    "MAX_SUMMARY_CHARS",
    "tokenize",
    "MemoryContext",
    "render",
    "MAX_TASK_CHARS",
    "MAX_STORED_SUMMARY_CHARS",
    "MemoryLimits",
    "MemoryStats",
    "PruneResult",
    "DEFAULT_LIMITS",
    "DEFAULT_MAX_RECORDS",
    "DEFAULT_MAX_BYTES",
]
