"""Memory 的清理与统计（Phase 11.3）：历史别无限长下去。

**要解决的问题**：memory.jsonl 是只追加的——每跑一个任务写一条，跑得
越久文件越大；同一条任务做出同样结果，还会留下一串一模一样的记录。

这一层给出两条线（`MemoryLimits`：最多留几条、文件最多多少字节）和
三件事：

- `dedupe`：完全一样的历史（task + outcome + summary 三个字段全相同）
  只留最新的一条。不接 embedding、不算相似度——只认一模一样。
- `plan_prune`：去重 + 按上限丢最旧的，**永远保留最新**。它只负责算出
  「留哪些、各类丢多少」，真正动文件的是 store（临时文件 + 换名，见
  memory/store.py）。
- `stats_of`：数一数历史（总数 / 成 / 败 / 最早最晚的时间戳），给人和
  排查用——这是**读**的辅助，不进模型的 context。

**按「写入顺序」判新旧**：store 里的顺序就是写入顺序，正常使用下也就是
时间顺序。不解析时间戳、不拿时间戳字符串比大小——从别处导进来的记录，
时间戳格式不保证一样，比出来的结果没有意义。

**清理只从最旧的下手**：先丢最旧的，直到不超上限；要留的完整保留。
单条记录自己就超大小上限时留它一条——上限是防止无限增长，不是把历史
抹干净（写入时本来就有长度裁剪，正常到不了这一步）。
"""

import json
from dataclasses import dataclass

from memory.base import FAILURE, SUCCESS, MemoryRecord

# 默认上限：500 条 / 2 MiB。一条记录几百字节，500 条也就是几百 KB——
# 够攒一段时间的参考；2 MiB 是文件大小的兜底。检索每次扫全量，这个
# 量级扫一遍是瞬间的事。config.py 里有一份同值的默认（那边是配置的
# 默认值），有测试钉着两处相等。
DEFAULT_MAX_RECORDS = 500
DEFAULT_MAX_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class MemoryLimits:
    """清理的两条线：最多留几条、文件最多多少字节。

    两个都必须 ≥ 1：想完全不限就别用这个（LocalMemoryStore 的
    limits=None 才是「不限」），别用 0 表示——0 和「一条都不留」长得
    太像，容易误读。
    """

    max_records: int = DEFAULT_MAX_RECORDS
    max_bytes: int = DEFAULT_MAX_BYTES

    def __post_init__(self) -> None:
        if self.max_records < 1:
            raise ValueError(f"max_records 至少为 1，收到 {self.max_records}")
        if self.max_bytes < 1:
            raise ValueError(f"max_bytes 至少为 1，收到 {self.max_bytes}")


# store 构造时不传 limits 就用它；传 None 明确表示「不设上限」。
DEFAULT_LIMITS = MemoryLimits()


@dataclass(frozen=True)
class MemoryStats:
    """历史有几条、成 / 败各多少、最早 / 最晚的时间戳（空历史时是 None）。

    oldest / newest 取**文件里的第一条和最后一条**（写入顺序），不是拿
    时间戳算出来的（理由见模块开头）。
    """

    total: int
    success: int
    failure: int
    oldest: str | None
    newest: str | None


def stats_of(records: list[MemoryRecord]) -> MemoryStats:
    """数一数这批记录：总数、成功、失败、最早 / 最晚的时间戳。"""
    return MemoryStats(
        total=len(records),
        success=sum(1 for record in records if record.outcome == SUCCESS),
        failure=sum(1 for record in records if record.outcome == FAILURE),
        oldest=records[0].timestamp if records else None,
        newest=records[-1].timestamp if records else None,
    )


def dedupe(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """去掉重复的历史：三元组一模一样只留最新那条，其余原样保序。

    从后往前扫（先碰到的是更新的），见过的三元组就跳过。**只认完全
    一样**：task 相同但 outcome / summary 不同的不是重复——那是同一件
    事的两次不同结果，两条都留着。不接 embedding，不用相似度。
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[MemoryRecord] = []
    for record in reversed(records):
        key = (record.task, record.outcome, record.summary)
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    kept.reverse()
    return kept


def record_size(record: MemoryRecord) -> int:
    """一条记录在文件里占多少字节：JSON 那行的字节数 + 换行符。

    和 store 真正写出去的一行同一个算法（ensure_ascii=False、普通的
    \\n），大小上限的账才和 path.stat() 对得上。
    """
    line = json.dumps(record.to_dict(), ensure_ascii=False)
    return len(line.encode("utf-8")) + 1


@dataclass(frozen=True)
class PruneResult:
    """一次清理的账（store.prune / compact 的返回值）。

    - kept：清理后留下的记录，写入顺序（最新在最后）——store 拿它重写
      文件；调用方通常只关心下面的计数。
    - before / after：清理前后各多少条（after == len(kept)）。
    - deduped / over_count / over_bytes：分别因为「重复」「超过条数上限」
      「超过大小上限」去掉多少条。
    - rewritten：文件是不是真被重写过了。没超限也没重复时是 False——
      那种情况下文件一个字节都没动。
    """

    kept: list[MemoryRecord]
    before: int
    after: int
    deduped: int = 0
    over_count: int = 0
    over_bytes: int = 0
    rewritten: bool = False


def _fit_bytes(records: list[MemoryRecord], max_bytes: int) -> list[MemoryRecord]:
    """从最新往回留，装不下就停；至少留一条（见模块开头）。"""
    kept: list[MemoryRecord] = []
    total = 0
    for record in reversed(records):
        size = record_size(record)
        if kept and total + size > max_bytes:
            break
        kept.append(record)
        total += size
    kept.reverse()
    return kept


def plan_prune(
    records: list[MemoryRecord], limits: MemoryLimits | None
) -> PruneResult:
    """算一算按上限该留哪些：先去重，再超条数丢最旧的，再超大小丢最旧的。

    limits=None（store 没设上限）时只去重，不因为「太多太大」丢历史。
    这个函数不碰文件，`rewritten` 留给真正重写的 store 填。
    """
    unique = dedupe(records)
    deduped = len(records) - len(unique)

    kept = unique
    over_count = 0
    over_bytes = 0
    if limits is not None:
        if len(kept) > limits.max_records:
            over_count = len(kept) - limits.max_records
            kept = kept[over_count:]  # 顺序就是新旧：切掉前面最旧的
        fitted = _fit_bytes(kept, limits.max_bytes)
        over_bytes = len(kept) - len(fitted)
        kept = fitted

    return PruneResult(
        kept=kept,
        before=len(records),
        after=len(kept),
        deduped=deduped,
        over_count=over_count,
        over_bytes=over_bytes,
    )
