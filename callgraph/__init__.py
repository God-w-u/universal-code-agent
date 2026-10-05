"""Call Graph：项目里「函数 A 调用了函数 B」这张关系网（Phase 12.1）。

- base：CallEdge（一条调用边）、CallGraph（边 + callers / callees 查询 +
  find_path 调用链搜索）、CallPath（一条从起点到终点的链）——语言无关，
  全是字符串和数字，不绑定任何一种 AST。
- python / cpp：两种语言的**调用点**提取器，只回答「第几行在调谁」。
- index（CallGraphIndex）：在 ProjectIndex / SymbolIndex 之上把调用点认到
  函数头上，建出整份图并提供查询；加语言就是往 SITE_EXTRACTORS 补一条。

第一阶段只有离线索引和查询：不接 Agent Tool、不改 Agent Loop。
"""

from callgraph.base import DEFAULT_MAX_DEPTH, CallEdge, CallGraph, CallPath, CallSite
from callgraph.index import SITE_EXTRACTORS, CallGraphIndex

__all__ = [
    "CallEdge",
    "CallGraph",
    "CallPath",
    "CallSite",
    "CallGraphIndex",
    "DEFAULT_MAX_DEPTH",
    "SITE_EXTRACTORS",
]
