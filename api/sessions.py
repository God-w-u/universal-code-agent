"""会话管理：一个 session_id 一只 Agent，互不串话（Phase 13.3）。

**为什么要有它**：Agent 的 messages 就是会话历史，一份历史只能服务一个
会话——两个标签页共用一个 Agent 会串话（A 问完的上下文进了 B 的请求）。
这层把「按会话隔离」收在一处：session_id → Session（里面一只独立 Agent），
创建 / 拿取 / 删除 / 过期清理都从这里走。Agent 本身完全不认识会话这回事，
不该由它管生命周期。

**共享的是资源，不是历史**：每只 Agent 由同一个工厂造（main.AgentFactory）——
工具注册表、模型客户端、Memory store 都是同一份。懒建的索引（调用图、
代码块、embedding）挂在那些共享件上，所以「新建一个会话」= 造一只轻量
Agent，**不重扫项目**；会话被清理也只丢掉它自己的 messages，共享件一个
都不碰。

**清理两条线**：
- 过期：距上次使用超过 ttl 的，下次有人来（create / get / acquire）时
  顺手清掉。**握着锁的不清**——它正在跑，刚被用过；
- 超量：会话数已到 max_sessions 还要新建，就先丢「最久没用」的那个
  （LRU）。正在跑任务的（锁被握着）先不丢；全都忙才退回丢最久那个
  （见 _evict_locked，有测试钉着）。

两种清理都只动本层的表；正在跑的请求手里还攥着 Session 对象，不受影响。
「上次使用」= 请求进来的那一刻（get / create），外加一轮跑完时的
`touch()`——不然一个比 ttl 还长的任务，跑完就已经算过期了。

**并发**：本层自己的表用一把锁护着（创建 / 查找 / 删除都是临界区）；
每个 Session 另带一把自己的锁，给「同一个会话别同时跑两个任务」用——
两把锁各管各的，不同 Session 之间不互相阻塞。
"""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 只为类型检查：运行时不 import，api 和 agent 之间少一条运行期依赖
    from agent import Agent

# 默认值：一小时没人用就过期；本地服务 32 个会话足够，超了丢最久没用的。
# 都能被环境变量覆盖，见 config.load_api_config() / .env.example。
DEFAULT_TTL = 3600.0
DEFAULT_MAX_SESSIONS = 32


@dataclass
class Session:
    """一个会话：id + 它自己的 Agent + 自己那把 run 锁。

    lock 是给「同一个会话的两次 run」用的：messages 是共享的会话历史，
    同时跑会互相踩；不同会话各拿各的锁，不互堵。
    """

    id: str
    agent: "Agent"
    created_at: float
    last_used: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    """session_id → Session 的表：创建、拿取、删除、过期清理都在这儿。

    用法（api/server.py 就是这么用的）：

        manager = SessionManager(factory.new_agent, ttl=..., max_sessions=...)
        session = manager.acquire(request 里带的 session_id 或 None)
        with session.lock:
            session.agent.run(task)

    - factory：不带参数、每调一次造一只新 Agent 的可调用对象；
    - ttl / max_sessions：见模块开头，默认值和 config 里的默认对齐；
    - clock：现在几点，默认 time.monotonic——测试注入假时钟就能「快进」。
    """

    def __init__(
        self,
        factory: "Callable[[], Agent]",
        *,
        ttl: float = DEFAULT_TTL,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl 必须大于 0，收到 {ttl}")
        if max_sessions < 1:
            raise ValueError(f"max_sessions 至少为 1，收到 {max_sessions}")

        self._factory = factory
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        """新建一个会话（factory 现造一只 Agent）。"""
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            return self._create_locked(now)

    def get(self, session_id: str) -> Session | None:
        """按 id 取会话（顺手记一次「刚用过」）；没有 / 已过期给 None。"""
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_used = now
            return session

    def acquire(self, session_id: str | None = None) -> Session | None:
        """接口层的入口：给了 id 就取（没有 → None），不给就新建。

        不认识的 id 返回 None，**不会**顺手新建一个顶着这个 id 的会话：
        session_id 一律由服务端发（uuid），客户端只负责带回来；认不出来
        就是过期或拼错了，让上层回 404 说清楚，别悄悄换一只。
        """
        if session_id is None:
            return self.create()
        return self.get(session_id)

    def delete(self, session_id: str) -> bool:
        """删掉一个会话：删到了 True，本来就没有 False。

        只从表里摘掉（Agent 随之可回收），共享的工具、索引、Memory store
        一个都不动。
        """
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def purge_expired(self) -> list[str]:
        """清掉过期的会话，返回被清掉的 id（给测试和日志看）。"""
        now = self._clock()
        with self._lock:
            return self._purge_locked(now)

    def touch(self, session: Session) -> None:
        """记一次「这个会话刚被用过」。

        跑任务的时候要单独调一次：`get()` / `create()` 只在**请求进来的
        那一刻**刷新 last_used，而一轮 run 可能跑很久。不刷的话，一个比
        ttl 还长的任务跑完那一刻，它自己就已经算过期了——客户端接着追问
        （还是同一个 session_id）会先被 purge 掉，然后拿到一个 404，
        整段对话凭空消失。

        只在这条会话还在表里时刷新：已经被删掉的会话不该被这次调用复活。
        """
        now = self._clock()
        with self._lock:
            if self._sessions.get(session.id) is session:
                session.last_used = now

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------
    # 临界区内部：调用方已经握着 self._lock
    # ------------------------------------------------------------------

    def _purge_locked(self, now: float) -> list[str]:
        """清过期的会话。**握着锁的不清**——它正在跑。

        和 _evict_locked 同一条理由，这里以前漏了：把一条正在跑任务的
        会话从表里摘掉，跑的那一轮手里的对象还在、会照常跑完，但客户端
        再也拿不到它（下一个请求 404），那段对话等于凭空消失；而这条
        会话其实刚被用过（有请求正在它身上跑）。

        `last_used` 由 `get()` / `create()` 刷新，另外跑完一轮还会
        `touch()` 一次，所以「刚跑完的任务」不会被下一问判成过期。
        """
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_used > self.ttl and not session.lock.locked()
        ]
        for session_id in expired:
            del self._sessions[session_id]
        return expired

    def _create_locked(self, now: float) -> Session:
        if len(self._sessions) >= self.max_sessions:
            self._evict_locked()
        session = Session(
            id=uuid.uuid4().hex,
            agent=self._factory(),
            created_at=now,
            last_used=now,
        )
        self._sessions[session.id] = session
        return session

    def _evict_locked(self) -> None:
        """超量了：丢最久没用的一个。

        优先丢**空闲**的（正在跑任务的先不丢，理由见模块开头）。如果全都
        在跑，就退回「还是丢最久那个」——这是在「表无限涨」和「踢掉一条
        正在跑的会话」之间做的取舍：上限是硬要求，而踢掉的那条跑完自己
        会看到 404。这一条有测试钉着（MaxSessionsTest），是有意为之。
        """
        idle = [s for s in self._sessions.values() if not s.lock.locked()]
        victim = min(idle or list(self._sessions.values()), key=lambda s: s.last_used)
        del self._sessions[victim.id]
