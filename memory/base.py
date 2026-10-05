"""Memory 的基础抽象（Phase 11.1）：一条记录长什么样、一个 store 该会什么。

这一层只回答两个问题：「历史里存的是什么」和「存取历史的最小接口是
什么」。怎么落盘在 store.py、怎么按 query 找在 retriever.py——以后
换存储方式（比如上数据库）时改的是 store，记录结构和接口都不动。

**记录为什么是这几个字段**：id 用来去重 / 引用一条历史；timestamp 记下
这事什么时候发生；task 是当时的任务 / 查询原文、summary 是结果摘要——
检索只读这两个字段；outcome 一眼看出是成了还是败了；metadata 留给以后
（模型名、轮次、耗时……），现在不定义里面有什么，但类型先钉成
「字符串 → 字符串」，免得读回来的历史一半是数字一半是字符串。

**为什么 summary 要能截短**：历史是给模型当参考的，不是把原文倒回去。
记录本身存全文，`clipped()` 只产出一份进 context 用的副本——那条边界
由 retriever 执行（见 memory/retriever.py 的模块开头）。
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

# outcome 只认这两个值。写成常量而不是到处用裸字符串：拼错了在构造
# 的时候就报错，而不是等检索时把一条 failure 当成 success 用。
SUCCESS = "success"
FAILURE = "failure"
OUTCOMES = (SUCCESS, FAILURE)


def now_timestamp() -> str:
    """当前时间：UTC、秒级 ISO 8601。

    用 UTC 是为了不带歧义（机器换时区、跨机器读同一份历史都不受影响）；
    秒级够用，字符串还短。返回的是字符串不是 datetime——JSONL 里存的
    就是它，读回来一字不差，不用解析。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clip_text(text: str, limit: int) -> str:
    """把文本截到不超过 limit 个字符：超了就留前 limit - 1 个字符 + 省略号。

    裁的是**尾巴**：开头才是「这次任务在说什么」，结尾丢掉的损失最小。
    省略号自己占一个字符，所以结果长度恰好不超过 limit。
    """
    if limit < 1:
        raise ValueError(f"limit 至少为 1，收到 {limit}")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass(frozen=True)
class MemoryRecord:
    """一条任务历史：字段含义见模块开头。

    frozen=True：记录是「已经发生过的事」，存进去之后不该被改（要截短
    摘要用 `clipped()`，它返回副本）。校验放在 __post_init__：坏记录在
    构造时就报错，而不是先写进文件、下次读的时候才发现。
    """

    id: str
    timestamp: str
    task: str
    summary: str
    outcome: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError(f"id 不能为空，收到 {self.id!r}")
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError(f"timestamp 不能为空，收到 {self.timestamp!r}")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task 不能为空：一条不知道是什么任务的历史没有用")
        if not isinstance(self.summary, str):
            raise ValueError(
                f"summary 必须是字符串，收到 {type(self.summary).__name__}"
            )
        if self.outcome not in OUTCOMES:
            raise ValueError(
                f"outcome 只能是 {'、'.join(OUTCOMES)}，收到 {self.outcome!r}"
            )
        if not isinstance(self.metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise ValueError("metadata 必须是「字符串 → 字符串」的字典")

    @classmethod
    def create(
        cls,
        task: str,
        summary: str,
        outcome: str,
        metadata: dict[str, str] | None = None,
        *,
        id: str | None = None,
        timestamp: str | None = None,
    ) -> "MemoryRecord":
        """造一条记录：不传 id / timestamp 就现生成。

        日常用法就是 `MemoryRecord.create(task=..., summary=..., outcome=...)`
        再加 `store.append(...)`；id / timestamp 显式传是为了测试和
        从别处导入历史。
        """
        return cls(
            id=id or uuid.uuid4().hex,
            timestamp=timestamp or now_timestamp(),
            task=task,
            summary=summary,
            outcome=outcome,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """转成能直接 json.dumps 的普通字典（字段顺序固定，方便人看）。"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "task": self.task,
            "summary": self.summary,
            "outcome": self.outcome,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MemoryRecord":
        """从 JSON 读回来的那条路；形状不对一律 ValueError。

        JSON 是外部输入（文件可能被人手改过、被写坏过），所以这里宁可
        判成坏记录也不拼一条半真的历史：字段一个不能少、类型必须是
        声明的那种。metadata 的值统一转成字符串（JSON 里可能是数字 /
        布尔），保持记录类型简单。坏记录怎么处理由 store 决定——它是
        「跳过这一条」，不是「整个历史读不出来」。
        """
        if not isinstance(data, dict):
            raise ValueError(f"记录必须是 JSON 对象，收到 {type(data).__name__}")

        fields: dict[str, str] = {}
        for key in ("id", "timestamp", "task", "summary", "outcome"):
            value = data.get(key)
            if not isinstance(value, str):
                raise ValueError(f"字段 {key!r} 必须是字符串")
            fields[key] = value

        raw_meta = data.get("metadata")
        if raw_meta is None:
            raw_meta = {}
        if not isinstance(raw_meta, dict):
            raise ValueError("metadata 必须是对象")
        return cls(
            **fields,
            metadata={str(key): str(value) for key, value in raw_meta.items()},
        )

    def clipped(self, limit: int) -> "MemoryRecord":
        """一份把 summary 截到 limit 字以内的**副本**（原记录不动）。

        进模型 context 的是副本：全文留在 store 里，喂模型的是裁过的
        这一段。裁剪规则见 clip_text。
        """
        summary = clip_text(self.summary, limit)
        if summary == self.summary:
            return self
        return replace(self, summary=summary)


class MemoryStore(ABC):
    """历史存取的接口：只管「追加一条」和「读出全部」。

    现在只有 LocalMemoryStore（JSONL 文件）一个实现；以后换数据库也是
    实现这两个方法，上层的 retriever 不用动。注意 `load()` 读的是
    **全部**记录——哪些、以多长进模型 context 由 MemoryRetriever 决定，
    别把 load() 的结果直接塞给模型。
    """

    @abstractmethod
    def append(self, record: MemoryRecord) -> None:
        """追加一条记录；文件 / 目录不存在就从这里建出来。"""

    @abstractmethod
    def load(self) -> list[MemoryRecord]:
        """读出所有能读的记录（顺序 = 写入顺序）；单条坏记录不影响其余的。"""
