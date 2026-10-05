"""一次 Run 的取消开关：协作式，不杀线程。

Python 没有安全杀线程的办法——硬杀会留下写了一半的文件、半条消息历史。
这里的做法是**协作式**：跑的那一边在几个固定边界上问一句「取消了吗」，
是就抛 RunCancelled 自己停下来。取消方只负责把开关按下去，一下都不等。

两个东西：

- CancellationContext：一次 Run 一个。cancel() 谁都能调（HTTP 的取消接口
  在另一条线程上调），is_cancelled() / raise_if_cancelled() 是跑的那边在
  检查点上调。线程安全：cancel 只写一次（先到的原因算数），读的永远不
  阻塞——**取消请求绝不能被正在跑的那一轮卡住**。
- 整轮超时用的是同一个开关：构造时给 timeout，到点之后的下一次检查就按
  「超时」停。不另起定时线程，到点这件事在下一次检查里现算，正在跑的那
  条线程不会被后台线程打断。

状态只属于这一次 Run：context 用完就丢，下一轮新建一个——取消绝不跨轮
泄漏。所以**别把一次的 context 传给下一轮**。
"""

import threading
import time
from collections.abc import Callable

# 两种取消原因。值会原样进 RunCancelled.reason，API 层靠它把终态分成
# 「客户端主动停」和「整轮超时」两种。
CANCELLED = "cancelled"
TIMEOUT = "timeout"

_REASON_TEXT = {
    CANCELLED: "本次运行已被取消（客户端主动停止）",
    TIMEOUT: "本次运行超过了整轮时间上限",
}


class RunCancelled(RuntimeError):
    """本次 Run 在检查点发现已被取消，正常停下——不是运行出错。

    刻意**不**继承 AgentError：取消是控制流，不是「运行失败了」。Agent 的
    回滚、Memory 记账这些失败路径照样走（它仍是一个 RuntimeError，会被
    Agent.run 的异常分支接住），但 API 层可以在此之前把它认出来，映射成
    cancelled / timed_out，而不是 5xx。
    """

    def __init__(self, reason: str = CANCELLED) -> None:
        self.reason = reason
        super().__init__(_REASON_TEXT.get(reason, f"本次运行已中止（{reason}）"))


class CancellationContext:
    """一次 Run 的取消开关（线程安全；取消方永远不被卡住）。

    timeout 给了就是整轮预算：从构造那一刻（HTTP 层受理请求）起算，
    LLM + 工具 + 修复共享这一段总时间，到点后下一次检查按「超时」停。
    """

    def __init__(
        self,
        timeout: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout 必须是正数，或不给（不给 = 不限时）")

        self._clock = clock
        self._lock = threading.Lock()
        self._reason: str | None = None
        # 到点时刻（None = 不限时）。到点 ≠ 已取消：跑的那边下一次检查才算数
        self.deadline: float | None = (
            clock() + timeout if timeout is not None else None
        )

    def cancel(self, reason: str = CANCELLED) -> bool:
        """发起取消。返回这次是不是**第一枪**；已经取消过就返回 False。

        反复调是幂等的（HTTP 层「已结束的 Run 再 cancel」靠的就是它）；
        先到的原因算数——客户端按了停、随后超时又追上来，报的是「取消」。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("取消原因要是非空字符串")

        with self._lock:
            if self._reason is not None:
                return False
            self._reason = reason
        return True

    def is_cancelled(self) -> bool:
        """取消了吗：主动取消过，或者整轮超时的点已经到了。"""
        return self._reason is not None or self._expired()

    @property
    def reason(self) -> str | None:
        """取消原因；没取消时是 None。主动取消优先于超时。"""
        if self._reason is not None:
            return self._reason
        return TIMEOUT if self._expired() else None

    def raise_if_cancelled(self) -> None:
        """检查点：取消了就抛 RunCancelled，没取消什么都不做。"""
        reason = self.reason
        if reason is not None:
            raise RunCancelled(reason)

    def _expired(self) -> bool:
        """整轮预算用完了吗（到点就算用完）。"""
        return self.deadline is not None and self._clock() >= self.deadline
