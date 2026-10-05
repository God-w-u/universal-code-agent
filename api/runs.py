"""Run 的登记簿：run_id、状态、取消开关（13.4）。

一次 POST /api/run（或 /api/run/stream）就是一条 Run：服务端给它发一个
run_id，登记在这儿；跑完（成 / 败 / 取消 / 超时）挪到「最近结束的」那一
小格里。取消接口靠这张表找到目标——**它不碰会话锁**，所以正在跑的那一轮
（还攥着会话锁）绝卡不住取消请求。

状态机（13.4 规格）：

    created ──> running ──┬──> completed
                          ├──> failed
                          ├──> cancelled
                          └──> timed_out

状态只在 API 这一层流动：Agent / 工具完全不知道 HTTP 那套；Agent 只会
在检查点上抛 RunCancelled（agent/cancel.py），由路由层翻成终态。

**不永久占内存**：跑着的在 _active 里，结束的挪进 _finished，只留最近
max_finished 条（cancel / status 还查得到），更早的直接丢——一个小 deque
的事，没有 Redis、没有数据库。
"""

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from agent.cancel import CancellationContext

# 五个状态。created 只活一小会儿（受理到开跑之间），前端基本只看得到
# running / 四个终态。
CREATED = "created"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
TIMED_OUT = "timed_out"

# 终态：进了这里就不会再动，重复 cancel 只回状态、不改结果
FINISHED = (COMPLETED, FAILED, CANCELLED, TIMED_OUT)

# 结束的 Run 保留多少条供 cancel / 状态查询；只留最近这些
DEFAULT_MAX_FINISHED = 64


@dataclass
class Run:
    """一条 Run：身份、归属、取消开关、状态、起止时间。"""

    id: str
    session_id: str
    cancellation: CancellationContext
    status: str = CREATED
    created_at: float = 0.0
    finished_at: float | None = None

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class RunRegistry:
    """run_id → Run 的表。所有方法线程安全；锁只在查表时握着，绝不外溢。"""

    def __init__(
        self,
        max_finished: int = DEFAULT_MAX_FINISHED,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_finished < 1:
            raise ValueError("max_finished 至少为 1")
        self.max_finished = max_finished
        self._clock = clock
        self._lock = threading.Lock()
        self._active: dict[str, Run] = {}
        self._finished: dict[str, Run] = {}
        # 结束的先后顺序：超量时从最早的开始丢
        self._order: deque[str] = deque()

    def create(self, session_id: str, *, timeout: float | None = None) -> Run:
        """登记一条新 Run。run_id 服务端发（uuid4）——客户端伪造不了。

        timeout 是整轮预算（秒；None = 不限时）：取消开关构造出来的那一刻
        就开始计时，排队等会话锁的时间也算在里面。
        """
        run = Run(
            id=uuid.uuid4().hex,
            session_id=session_id,
            cancellation=CancellationContext(timeout=timeout),
            created_at=self._clock(),
        )
        with self._lock:
            self._active[run.id] = run
        return run

    def begin(self, run: Run) -> None:
        """created -> running：这一轮受理完了，真的要跑了。"""
        with self._lock:
            if run.status == CREATED:
                run.status = RUNNING

    def finish(self, run: Run, status: str) -> None:
        """running -> 某个终态：挪出活跃表，只留最近 max_finished 条。"""
        if status not in FINISHED:
            raise ValueError(f"不是终态：{status!r}")
        with self._lock:
            run.status = status
            run.finished_at = self._clock()
            self._active.pop(run.id, None)
            self._finished[run.id] = run
            self._order.append(run.id)
            while len(self._order) > self.max_finished:
                self._finished.pop(self._order.popleft(), None)

    def get(self, run_id: str) -> Run | None:
        """按 run_id 找：跑着的、以及最近结束的，都还查得到。"""
        with self._lock:
            return self._active.get(run_id) or self._finished.get(run_id)

    def active_for(self, session_id: str) -> Run | None:
        """这个会话最老的那条活跃 Run——就是「当前这一轮」。

        同一个会话的请求在会话锁上排队，排前面的先跑，所以最老的那条就是
        真正在跑的那条（排在后面的还在锁上等）。没有就是 None。
        """
        with self._lock:
            # dict 保持插入顺序 = 受理顺序
            for run in self._active.values():
                if run.session_id == session_id:
                    return run
        return None

    def cancel(self, run_id: str, session_id: str) -> tuple[Run, bool] | None:
        """按 (run_id, session_id) 发起取消。

        找不到、或者这条 Run 不属于这个会话 → None（调用方回 404；跨会话
        当作不存在，不泄露别人的 run_id）。已经结束的 Run 原样返回、
        第二个值 False——**幂等**，不是错误；返回的第二个值表示「这一枪
        是不是本次真正按下的取消」。

        只碰登记表和那一轮的开关，不取会话锁：取消方永远不会被正在跑的
        那一轮卡住。
        """
        with self._lock:
            run = self._active.get(run_id) or self._finished.get(run_id)
            if run is None or run.session_id != session_id:
                return None
            alive = not run.finished
        # 取消开关自己的锁在锁外拿：这里是纯粹的置位，不等任何人
        started = run.cancellation.cancel() if alive else False
        return run, started

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def finished_count(self) -> int:
        with self._lock:
            return len(self._finished)
