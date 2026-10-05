"""LocalMemoryStore：把历史存成本地一个 JSONL 文件（Phase 11.1）。

**存哪**：默认 `~/.cache/universal-code-agent/memory.jsonl`——和向量缓存
放同一个目录，**在项目目录之外**：运行产生的数据不进仓库、不会被代码
索引扫到（和向量缓存同一个理由）。要换位置就在构造时给 path，测试
永远给临时路径。

**为什么是 JSONL**（一行一条 JSON 记录）：追加就是往文件尾巴上加一行，
不用重写整个文件；一行坏了只坏那一条，其余照读。只有标准库，没有新
依赖。

**长期运行怎么办**（11.3）：文件不能无限长。默认最多留 500 条 / 2 MiB
（`MemoryLimits`，见 memory/maintenance.py）：每次 append 写完顺手看
一眼，超了就去重 + 丢掉最旧的（`prune`），**永远保留最新**。清理是
「附带的一步」——没做成只记 `last_error`，绝不影响已经写进去的记录；
要显式整理就调 `prune()` / `compact()`，那两个会把错误抛出来。传
`limits=None` 表示不设上限（append 退回成「只加一行」）。

**坏记录怎么办**：读的时候逐行解析，坏行直接跳过（不抛异常），好记录
一条不少——单条损坏不能让整个历史读不出来。平时坏行留在原地不清
（读是只读的）；清理重写文件时它自然消失（那正是 `compact` 的活）。

**写到一半怎么办**：一条记录一行、一次 write、写完就关（关文件等于
flush），半写最多毁掉**最后一行**，下次读跳过它就是。还有一种更隐蔽的
半写：上次断在行中间，文件结尾没有换行符——这时直接追加，新记录
会粘在残句后面，本来只坏一条、现在坏两条。所以 append 前先看文件
最后一个字节：不是换行就先补一个，让残句自己成一行去坏。

**重写为什么是安全的**：`prune` / `compact` 会把整个文件重写，走的是
「先写同目录的临时文件，再换名过来」（`_rewrite`）：写到一半断了，原
文件一个字节没动，坏的只是临时文件。平时 append 不读全文、只加一行
（见上），没有这个风险。

**换行统一写成 \\n**（写文件时 newline=""）：大小上限的账（一条记录占
多少字节，见 maintenance.record_size）才算得准；读的时候 \\r\\n 一样
认，老文件照读。

**并发**（13.3）：多个会话共用同一个 store，append 可能撞上另一个
append。append / prune / compact 用一把可重入锁护着（append 里还会调
prune，所以要能重入）；load 只读、不加锁——_rewrite 走「临时文件 +
换名」，读的人要么看到旧文件要么看到新文件，没有半份。
"""

import json
import os
import threading
from dataclasses import replace
from pathlib import Path

from memory.base import MemoryRecord, MemoryStore
from memory.maintenance import (
    DEFAULT_LIMITS,
    MemoryLimits,
    MemoryStats,
    PruneResult,
    dedupe,
    plan_prune,
    stats_of,
)

# 默认文件名。目录和向量缓存共用（~/.cache/universal-code-agent）。
DEFAULT_FILENAME = "memory.jsonl"


def default_memory_path() -> Path:
    """默认位置：用户目录下——不进项目、不会被代码索引扫到。

    和 index.vector_cache.default_cache_dir() 是同一个目录；那边是缓存
    （删了只是慢一点），这边是数据（删了历史就没了），所以文件分开、
    名字分开。
    """
    return Path.home() / ".cache" / "universal-code-agent" / DEFAULT_FILENAME


class LocalMemoryStore(MemoryStore):
    """一个 JSONL 文件就是一个 store。

    文件**懒建**：构造、读一个不存在的文件都不会碰磁盘；第一次 append
    才把目录和文件建出来。这样「新建 store」不需要任何初始化步骤。

    **上限**：默认最多留 DEFAULT_MAX_RECORDS 条 / DEFAULT_MAX_BYTES
    字节，append 之后自动清理（见模块开头）；limits=None 表示不限。
    `last_error` 是最近一次**自动清理**没做成的原因，正常时是 None——
    显式 prune / compact 的错误直接抛给调用方，不写在这里。
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        limits: MemoryLimits | None = DEFAULT_LIMITS,
    ) -> None:
        self.path = (
            Path(path).expanduser() if path is not None else default_memory_path()
        )
        self.limits = limits  # None = 不限：append 只加一行，什么也不检查
        self.last_error: str | None = None
        # 13.3：多会话共用——写侧（append / prune / compact）串行化；
        # RLock 是因为 append → prune 是同一个线程再拿一次（见模块开头）
        self._lock = threading.RLock()

    def append(self, record: MemoryRecord) -> None:
        """追加一条记录；然后看一眼上限，超了就顺手清理（见类开头）。

        一行一次写，不做「读全部 → 改写 → 写回去」那套——那样写一半
        坏的是整个文件。超限时的清理是另一回事：那由 _rewrite 用临时
        文件 + 换名来做，原文件同样是安全的。
        """
        if not isinstance(record, MemoryRecord):
            raise TypeError(
                f"append 需要 MemoryRecord，收到 {type(record).__name__}"
            )

        # 写侧串行化（13.3）：查残句 → 追写 → 顺手清理，这几步之间不能
        # 夹进另一个 append，不然「补换行 / 写新行」和别的写交错会写乱
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record.to_dict(), ensure_ascii=False)
            # 残句判定要在打开写句柄之前做（下面那个句柄只写不读）
            torn = self._has_torn_tail()
            # newline=""：换行统一写成 \n，大小上限的账才算得准（见模块开头）
            with self.path.open("a", encoding="utf-8", newline="") as file:
                if torn:
                    # 上次留下的半行没有换行结尾：先补一个，不然新记录会
                    # 粘在残句后面，两条一起坏
                    file.write("\n")
                file.write(line + "\n")

            self._enforce_limits()

    def load(self) -> list[MemoryRecord]:
        """读出所有**能读的**记录（写入顺序 = 文件顺序）；坏行跳过，不抛。

        这是给测试 / 统计 / 排查用的**全量**读；给模型看的入口是
        `MemoryRetriever.retrieve`——它按 top_k 和 summary 上限裁好，
        不会把全部历史倒进 context。

        文件不存在 = 还没有历史，返回空列表（正常初始化）。文件在但
        读不动（路径是个目录、没有权限）才算错误，照样往外抛。
        """
        if not self.path.exists():
            return []

        records: list[MemoryRecord] = []
        # errors="replace"：半写留下的坏字节不当致命错误，交给
        # json.loads 判——解析不过去就是一条坏记录，跳过
        with self.path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(MemoryRecord.from_dict(json.loads(line)))
                except (ValueError, TypeError):
                    # 坏记录（JSON 解析失败 / 字段不对 / 类型不对）：
                    # 单条跳过，其余的照读。JSONDecodeError 是
                    # ValueError 的子类，一起接住了。
                    continue
        return records

    def prune(self) -> PruneResult:
        """按上限清理：去重 + 只留最新的，直到不超条数和大小。

        文件没有变化就不重写（返回的 rewritten=False，文件一个字节没
        动）。显式调用时错误照抛，由调用方决定怎么办；append 之后那次
        自动清理走的是 _enforce_limits，失败只记 last_error。

        没设上限（limits=None）时只去重，不因为「太多太大」丢历史。
        """
        # 「读 → 判断 → 重写」整段进锁（13.3）：不然两个 prune 交错，
        # 后一个会拿着旧快照把前一个刚写进去的记录盖掉
        with self._lock:
            self.last_error = None
            records = self.load()
            result = plan_prune(records, self.limits)
            if result.after == result.before:
                # 一条没丢：没重复、也没超限，文件保持原样（坏行也不动）
                return result
            self._rewrite(result.kept)
            return replace(result, rewritten=True)

    def compact(self) -> PruneResult:
        """把文件重写成「只剩能读的、去重后的记录」——不按上限丢历史。

        和 prune 的分工：prune 管「太多太大」（没超限就不动文件）；
        compact 管「文件脏不脏」——坏行、写到一半的残句、重复记录都在
        这次重写里消失，条数不因为上限减少。文件本来不存在就不动手
        （读不存在的文件是正常状态，也不把文件建出来）。
        """
        with self._lock:
            self.last_error = None
            records = self.load()
            if not self.path.exists():
                return PruneResult(kept=[], before=0, after=0)
            kept = dedupe(records)
            self._rewrite(kept)
            return PruneResult(
                kept=kept,
                before=len(records),
                after=len(kept),
                deduped=len(records) - len(kept),
                rewritten=True,
            )

    def stats(self) -> MemoryStats:
        """数一数历史：多少条、成 / 败各多少、最早 / 最晚的时间戳。

        给人和排查用的（比如启动时打一行），**不进模型的 context**——
        进模型的路只有 MemoryContext.before → retriever。
        """
        return stats_of(self.load())

    def _rewrite(self, records: list[MemoryRecord]) -> None:
        """把全部记录原子地写回去：先写同目录的临时文件，再换名过来。

        写临时文件时原文件一个字节没动；写坏了、换名失败，原文件也还是
        原样——最多是这次清理没做成（临时文件清掉，下一次再来）。换名
        用 Path.replace（同一目录内是原子的），谁也不会读到重写到一半
        的文件。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_name(self.path.name + ".tmp")
        try:
            with scratch.open("w", encoding="utf-8", newline="") as file:
                for record in records:
                    file.write(
                        json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                    )
            scratch.replace(self.path)
        except Exception:  # noqa: BLE001 —— 清理现场之后原样抛出
            try:
                scratch.unlink(missing_ok=True)
            except OSError:
                pass  # 临时文件清不掉，不该盖住真正的失败原因
            raise

    def _enforce_limits(self) -> None:
        """写完顺手清理：超限才动文件；清理本身失败不往外抛。

        记录已经写进去了——清理是「附带的一步」，没做成不该把一次成功
        的写入变成失败（和 MemoryContext 的降级同一条规矩）。原因留在
        last_error 里，下一次 append 还会再试。
        """
        if self.limits is None:
            return
        try:
            self.prune()
        except Exception as exc:  # noqa: BLE001 —— 自动清理绝不往外抛
            self.last_error = f"Memory 自动清理没做成：{exc}"

    def _has_torn_tail(self) -> bool:
        """文件非空、且最后一个字节不是换行 → 上次写到一半断了。"""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as file:
            file.seek(-1, os.SEEK_END)
            return file.read(1) != b"\n"
