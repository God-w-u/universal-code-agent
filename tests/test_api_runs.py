"""Run 的登记表、取消接口、整轮超时（13.4）——HTTP 层与登记表本身。

起真服务、走真 HTTP（沿用 test_api.py 的 ApiCase）；Agent 用测试替身，
不联网。这里钉的是：cancel API 的语义（幂等、跨会话 404、**不被会话锁
卡住**）、run_id 服务端发、/api/run 与 /api/run/stream 上的 cancelled /
timed_out、整轮超时和单工具超时的区别、SSE 事件的字段。

检查点本身的行为（取消后不再调模型 / 不再跑工具、假时钟超时）在
test_cancellation.py；同会话串行 / 跨会话并发沿用 test_api.py 与
test_api_sessions.py 里已有的用例（假 Agent 现在都收得下 cancellation）。
"""

import http.client
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from agent import Agent, RunCancelled
from api.runs import (
    CANCELLED,
    COMPLETED,
    CREATED,
    RUNNING,
    TIMED_OUT,
    RunRegistry,
)
from api.server import ApiHandler, ApiServer
from llm import LLMError
from tests.test_agent import FakeLLMClient, tool_call_reply
from tests.test_api import ApiCase, CountingAgent
from tests.test_api_stream import EchoTool, StreamCase, StubAgent
from tests.test_sessions import FakeClock
from tools import Tool, ToolRegistry

JSON_HEADER = {"Content-Type": "application/json"}


class CooperativeAgent:
    """配合取消的假 Agent：等到被取消就抛 RunCancelled；release 一到正常返回。

    真 Agent 的检查点行为在 test_cancellation.py；这里只用它验 HTTP 层的
    状态映射——「Agent 停下之后 API 回什么」。
    """

    def __init__(self, started=None, release=None, answer: str = "好的") -> None:
        self.started = started
        self.release = release
        self.answer = answer
        self.tasks: list[str] = []

    def run(
        self,
        task,
        on_tool_start=None,
        on_tool_end=None,
        cancellation=None,
        on_llm_start=None,
        on_llm_end=None,
    ):
        self.tasks.append(task)
        if self.started is not None:
            self.started.set()
        deadline = time.monotonic() + 5  # 兜底：既没取消也没人放它，5 秒后炸
        while time.monotonic() < deadline:
            if cancellation.is_cancelled():
                raise RunCancelled(cancellation.reason)
            if self.release is not None and self.release.is_set():
                return self.answer
            time.sleep(0.005)
        raise AssertionError("假 Agent 等不到取消，也没人放它走")


class StubbornAgent:
    """不理会取消的假 Agent：模拟「已经卡在单工具执行里」，只能等它自己回来。"""

    def __init__(self, started, release) -> None:
        self.started = started
        self.release = release

    def run(
        self,
        task,
        on_tool_start=None,
        on_tool_end=None,
        cancellation=None,
        on_llm_start=None,
        on_llm_end=None,
    ):
        self.started.set()
        self.release.wait(5)
        return "跑完了"


class SleepTool(Tool):
    """睡一小会儿的工具：让整轮预算在它跑完之后过期。"""

    name = "slow"
    description = "慢工具"
    parameters = {}

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def execute(self) -> str:
        time.sleep(self.seconds)
        return "慢结果"


class TimeoutTool(Tool):
    """模拟单工具自己超时：返回 run_command 那种文本，不是异常。"""

    name = "slow_command"
    description = "慢命令"
    parameters = {}

    def execute(self) -> str:
        return "命令：慢命令\n结果：超时，已强制终止"


class CancelRequests:
    """取消接口的公共部分：发一个 cancel 请求（ApiCase / StreamCase 共用）。"""

    def cancel(self, session_id: str, run_id: str | None = None):
        payload = {"session_id": session_id}
        if run_id is not None:
            payload["run_id"] = run_id
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.call_json("POST", "/api/runs/cancel", body, JSON_HEADER)


# ---------------------------------------------------------------------------
# 登记表本身：状态机、run_id、取消、容量
# ---------------------------------------------------------------------------


class RunRegistryTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(100.0)
        self.registry = RunRegistry(clock=self.clock)

    def test_a_new_run_is_created_with_a_server_side_id(self):
        first = self.registry.create("会话甲")
        second = self.registry.create("会话甲")

        self.assertEqual(first.status, CREATED)
        self.assertTrue(first.id)  # run_id 服务端发
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(first.session_id, "会话甲")
        self.assertEqual(first.created_at, 100.0)
        self.assertFalse(first.finished)
        self.assertEqual(self.registry.active_count, 2)

    def test_the_lifecycle_moves_created_running_finished(self):
        run = self.registry.create("会话甲")

        self.registry.begin(run)
        self.assertEqual(run.status, RUNNING)

        self.registry.finish(run, COMPLETED)

        self.assertEqual(run.status, COMPLETED)
        self.assertTrue(run.finished)
        self.assertEqual(run.finished_at, 100.0)
        self.assertEqual(self.registry.active_count, 0)
        self.assertEqual(self.registry.finished_count, 1)
        self.assertIs(self.registry.get(run.id), run)  # 结束的也查得到

    def test_finish_rejects_a_status_that_is_not_final(self):
        run = self.registry.create("会话甲")

        with self.assertRaises(ValueError):
            self.registry.finish(run, RUNNING)

    def test_cancel_reaches_an_active_run(self):
        run = self.registry.create("会话甲")
        self.registry.begin(run)

        found, started = self.registry.cancel(run.id, "会话甲")

        self.assertIs(found, run)
        self.assertTrue(started)
        self.assertTrue(run.cancellation.is_cancelled())

    def test_cancel_is_idempotent(self):
        run = self.registry.create("会话甲")
        self.registry.cancel(run.id, "会话甲")

        found, started = self.registry.cancel(run.id, "会话甲")

        self.assertIs(found, run)
        self.assertFalse(started)  # 第二枪不响，但也不是错误

    def test_cancel_a_finished_run_is_idempotent_too(self):
        run = self.registry.create("会话甲")
        self.registry.finish(run, COMPLETED)

        found, started = self.registry.cancel(run.id, "会话甲")

        self.assertIs(found, run)
        self.assertFalse(started)
        self.assertFalse(run.cancellation.is_cancelled())  # 结束的不再动它
        self.assertEqual(found.status, COMPLETED)

    def test_cancel_does_not_cross_sessions(self):
        run = self.registry.create("会话甲")

        self.assertIsNone(self.registry.cancel(run.id, "会话乙"))

        self.assertFalse(run.cancellation.is_cancelled())  # 别人的取消不生效

    def test_cancel_an_unknown_run_is_none(self):
        self.assertIsNone(self.registry.cancel("没有这个", "会话甲"))

    def test_active_for_returns_the_oldest_run_of_the_session(self):
        other = self.registry.create("会话乙")
        first = self.registry.create("会话甲")  # 同一会话：先受理的先跑
        second = self.registry.create("会话甲")

        self.assertIs(self.registry.active_for("会话甲"), first)
        self.assertIs(self.registry.active_for("会话乙"), other)
        self.assertIsNone(self.registry.active_for("没有这个"))
        self.assertIsNotNone(second)  # 排在后面的还在等锁

    def test_finished_runs_are_capped(self):
        registry = RunRegistry(max_finished=2, clock=self.clock)
        runs = []
        for index in range(3):
            run = registry.create(f"会话{index}")
            registry.finish(run, COMPLETED)
            runs.append(run)

        self.assertEqual(registry.finished_count, 2)
        self.assertIsNone(registry.get(runs[0].id))  # 最早的丢了，不占内存
        self.assertIs(registry.get(runs[1].id), runs[1])
        self.assertIs(registry.get(runs[2].id), runs[2])

    def test_max_finished_must_be_at_least_one(self):
        with self.assertRaises(ValueError):
            RunRegistry(max_finished=0)


# ---------------------------------------------------------------------------
# POST /api/runs/cancel
# ---------------------------------------------------------------------------


class CancelApiTest(CancelRequests, ApiCase):
    def test_cancelling_a_running_run_stops_it(self):
        started = threading.Event()
        self.start(CooperativeAgent(started=started))
        session_id = self.sessions.create().id

        results = []
        worker = threading.Thread(
            target=lambda: results.append(self.post_task("任务甲", session_id)),
            daemon=True,
        )
        worker.start()
        self.assertTrue(started.wait(5), "任务没开始跑")

        status, payload = self.cancel(session_id)

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertTrue(payload["cancelled"])  # 这一枪是我们按的
        self.assertEqual(payload["session_id"], session_id)
        self.assertEqual(payload["status"], RUNNING)  # 还没停完，这是正常的

        worker.join(5)
        self.assertEqual(len(results), 1)
        run_status, run_payload = results[0]
        self.assertEqual(run_status, 200)
        self.assertFalse(run_payload["success"])  # 不伪装成功
        self.assertEqual(run_payload["status"], CANCELLED)
        self.assertEqual(run_payload["run_id"], payload["run_id"])
        self.assertEqual(run_payload["session_id"], session_id)
        self.assertIn("取消", run_payload["answer"])

    def test_a_timed_out_run_is_reported_as_timed_out(self):
        self.start(CooperativeAgent(), run_timeout=0.05)
        session_id = self.sessions.create().id

        status, payload = self.post_task("任务甲", session_id)

        self.assertEqual(status, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], TIMED_OUT)
        self.assertTrue(payload["run_id"])
        self.assertIn("上限", payload["answer"])

    def test_a_real_agent_times_out_after_a_slow_tool(self):
        # 真 Agent + 真工具：整轮预算在「工具跑完的下一个检查点」生效
        self.start(
            Agent(
                client=FakeLLMClient(
                    [tool_call_reply(("slow", "{}")), "来不及说的回答"]
                ),
                tools=ToolRegistry([SleepTool(0.3)]),
            ),
            run_timeout=0.05,
        )

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 200)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["status"], TIMED_OUT)

    def test_a_tool_timeout_is_not_a_run_timeout(self):
        # 单工具自己超时只是工具结果，这一轮照常跑完——两个超时不是一回事
        client = FakeLLMClient([tool_call_reply(("slow_command", "{}")), "换个做法。"])
        agent = Agent(client=client, tools=ToolRegistry([TimeoutTool()]))
        self.start(agent)  # 整轮超时不设：单工具的超时不该升级成整轮

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["answer"], "换个做法。")
        # 成功的一轮不带 status：它不是被停的（取消 / 超时才带）
        self.assertNotIn("status", payload)
        observations = [m["content"] for m in agent.messages if m["role"] == "tool"]
        self.assertTrue(any("超时" in text for text in observations))

    def test_a_finished_run_can_be_cancelled_again_idempotently(self):
        self.start(CountingAgent())
        first = self.run_task("任务甲")

        status, payload = self.cancel(first["session_id"], first["run_id"])

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertFalse(payload["cancelled"])  # 没在跑，没什么可按的
        self.assertEqual(payload["status"], COMPLETED)  # 报的是它当时的终态

        again = self.cancel(first["session_id"], first["run_id"])[1]
        self.assertEqual(again, payload)  # 再来一次：一字不差

    def test_every_run_gets_a_fresh_run_id(self):
        self.start(CountingAgent())

        first = self.run_task("任务甲")
        second = self.run_task("任务乙", session_id=first["session_id"])

        self.assertTrue(first["run_id"])
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(second["session_id"], first["session_id"])

    def test_a_run_id_the_client_made_up_is_404(self):
        self.start(CountingAgent())
        first = self.run_task("任务甲")

        status, payload = self.cancel(first["session_id"], "伪造的-run-id")

        self.assertEqual(status, 404)
        self.assertFalse(payload["success"])
        self.assertIn("伪造的-run-id", payload["answer"])

    def test_cancelling_across_sessions_is_404(self):
        self.start(CountingAgent())
        first = self.run_task("任务甲")  # 会话甲的 Run（已结束）
        other = self.run_task("任务乙")  # 另起一个会话乙

        status, payload = self.cancel(other["session_id"], first["run_id"])

        self.assertEqual(status, 404)
        self.assertFalse(payload["success"])
        # 会话甲的 Run 一点没被动过
        run = self.server.runs.get(first["run_id"])
        self.assertEqual(run.status, COMPLETED)
        self.assertFalse(run.cancellation.is_cancelled())

    def test_cancelling_one_session_does_not_touch_another(self):
        events = [threading.Event(), threading.Event()]
        release = threading.Event()
        agents = [
            CooperativeAgent(started=events[0], release=release),
            CooperativeAgent(started=events[1], release=release),
        ]
        self.start(factory=lambda: agents.pop(0))
        first = self.sessions.create().id
        second = self.sessions.create().id

        results = {}
        workers = [
            threading.Thread(
                target=lambda: results.__setitem__(
                    "first", self.post_task("任务甲", first)
                ),
                daemon=True,
            ),
            threading.Thread(
                target=lambda: results.__setitem__(
                    "second", self.post_task("任务乙", second)
                ),
                daemon=True,
            ),
        ]
        for worker in workers:
            worker.start()
        for event in events:
            self.assertTrue(event.wait(5), "有任务没开始跑")

        status, payload = self.cancel(first)

        self.assertEqual(status, 200)
        self.assertTrue(payload["cancelled"])
        second_run = self.server.runs.active_for(second)
        self.assertIsNotNone(second_run)
        self.assertFalse(second_run.cancellation.is_cancelled())  # 乙不受影响

        workers[0].join(5)
        self.assertEqual(results["first"][1]["status"], CANCELLED)

        release.set()  # 放会话乙跑完
        workers[1].join(5)
        self.assertEqual(results["second"][0], 200)
        self.assertTrue(results["second"][1]["success"])

    def test_cancelling_a_session_with_nothing_running_is_404(self):
        self.start(CountingAgent())
        first = self.run_task("任务甲")  # 跑完就结束了

        status, payload = self.cancel(first["session_id"])

        self.assertEqual(status, 404)
        self.assertIn("没有在跑的", payload["answer"])

    def test_the_session_id_is_required(self):
        self.start(CountingAgent())

        status, payload = self.call_json(
            "POST", "/api/runs/cancel", b"{}", JSON_HEADER
        )

        self.assertEqual(status, 400)
        self.assertIn("session_id", payload["answer"])

    def test_a_run_id_that_is_not_a_string_is_400(self):
        self.start(CountingAgent())
        body = json.dumps({"session_id": "甲", "run_id": 123}).encode("utf-8")

        status, payload = self.call_json(
            "POST", "/api/runs/cancel", body, JSON_HEADER
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["success"])

    def test_the_client_cannot_bring_its_own_run_id(self):
        agent = CountingAgent()
        self.start(agent)
        body = json.dumps({"task": "任务甲", "run_id": "我自己编的"}).encode("utf-8")

        status, payload = self.call_json("POST", "/api/run", body, JSON_HEADER)

        self.assertEqual(status, 400)
        self.assertIn("run_id", payload["answer"])
        self.assertEqual(agent.tasks, [])  # 压根没跑

    def test_the_cancel_route_is_post_only(self):
        self.start(CountingAgent())

        status, headers, _ = self.call("GET", "/api/runs/cancel")

        self.assertEqual(status, 405)
        self.assertEqual(headers.get("Allow"), "POST")

    def test_the_cancel_is_not_blocked_by_the_run_lock(self):
        # 会话锁被正在跑的 Run 攥着：cancel 请求不取那把锁，必须立刻回
        started = threading.Event()
        release = threading.Event()
        self.start(StubbornAgent(started=started, release=release))
        session_id = self.sessions.create().id

        results = []
        worker = threading.Thread(
            target=lambda: results.append(self.post_task("任务甲", session_id)),
            daemon=True,
        )
        worker.start()
        self.assertTrue(started.wait(5), "任务没开始跑")

        began = time.monotonic()
        status, payload = self.cancel(session_id)
        elapsed = time.monotonic() - began

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], RUNNING)
        self.assertTrue(payload["cancelled"])
        self.assertLess(elapsed, 1.0)  # 被会话锁卡住的话这里会等到连接超时

        release.set()
        worker.join(5)
        # 取消信号送到了，但这个 Agent 没理它（模拟卡在工具里）：照常跑完
        self.assertEqual(results[0][0], 200)
        self.assertTrue(results[0][1]["success"])


# ---------------------------------------------------------------------------
# 流式接口：cancelled / timed_out 的 SSE 事件与字段
# ---------------------------------------------------------------------------


class StreamCancelTest(CancelRequests, StreamCase):
    def test_a_cancelled_stream_ends_with_a_cancelled_event(self):
        started = threading.Event()
        self.start(CooperativeAgent(started=started))
        session_id = self.sessions.create().id

        results = []
        worker = threading.Thread(
            target=lambda: results.append(self.stream("任务甲", session_id=session_id)),
            daemon=True,
        )
        worker.start()
        self.assertTrue(started.wait(5), "任务没开始跑")

        status, payload = self.cancel(session_id)
        worker.join(5)

        self.assertEqual(status, 200)
        events = self.parse(results[0][2])
        names = [name for name, _ in events]
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-1], "cancelled")  # 有明确的收尾事件
        self.assertNotIn("completed", names)  # 绝不伪装成功
        self.assertEqual(events[-1][1]["session_id"], session_id)
        self.assertEqual(events[-1][1]["run_id"], payload["run_id"])
        self.assertEqual(events[0][1]["run_id"], payload["run_id"])  # 同一个 run

    def test_a_timed_out_stream_ends_with_a_timed_out_event(self):
        self.start(CooperativeAgent(), run_timeout=0.05)
        session_id = self.sessions.create().id

        events = self.stream_events("任务甲", session_id=session_id)

        self.assertEqual([name for name, _ in events], ["start", "timed_out"])
        self.assertEqual(events[-1][1]["session_id"], session_id)
        self.assertEqual(events[-1][1]["run_id"], events[0][1]["run_id"])
        self.assertIn("上限", events[-1][1]["message"])

    def test_every_event_of_a_normal_stream_carries_the_same_run_id(self):
        client = FakeLLMClient(
            [tool_call_reply(("echo", '{"text": "任务甲"}')), "完成了。"]
        )
        self.start(Agent(client=client, tools=ToolRegistry([EchoTool()])))

        events = self.stream_events("任务甲")

        run_ids = {data["run_id"] for _, data in events if "run_id" in data}
        self.assertEqual(run_ids, {events[0][1]["run_id"]})
        self.assertTrue(events[0][1]["run_id"])

    def test_an_error_stream_carries_the_run_id(self):
        self.start(Agent(client=FakeLLMClient([LLMError("模型连不上")])))

        events = self.stream_events("任务甲")

        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["run_id"], events[0][1]["run_id"])

    def test_each_stream_gets_a_fresh_run_id(self):
        self.start(StubAgent())

        first = self.stream_events("任务甲")
        second = self.stream_events("任务乙")

        self.assertNotEqual(first[0][1]["run_id"], second[0][1]["run_id"])

    def test_a_client_disconnect_does_not_cancel_the_run(self):
        # 断线 ≠ 按停：分不清「用户关了页面」和「网断了」，所以断开不自动
        # 取消。要停只能 POST /api/runs/cancel
        started = threading.Event()
        release = threading.Event()
        self.start(StubbornAgent(started=started, release=release))
        session_id = self.sessions.create().id

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST",
            "/api/run/stream",
            json.dumps({"task": "任务甲", "session_id": session_id}).encode("utf-8"),
            JSON_HEADER,
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        response.readline()  # start 帧已经推过来了
        self.assertTrue(started.wait(5), "任务没开始跑")

        conn.close()  # 客户端走人

        run = self.server.runs.active_for(session_id)
        self.assertIsNotNone(run)  # 服务端还跑着
        self.assertFalse(run.cancellation.is_cancelled())  # 开关一动没动

        release.set()  # 让它自己跑完收场（往断开的连接上写失败会被兜底吃掉）


# ---------------------------------------------------------------------------
# 生命周期：哪条出口都不许留下「幽灵 Run」（13 阶段回归）
# ---------------------------------------------------------------------------
#
# 幽灵 Run = 登记表里永远 active、但已经没人跑的那一条。它不只是占内存：
# active_for() 找的是**最老**的那条，所以同会话往后每次「取消当前这一轮」
# 都打在它身上，真正在跑的那一轮反而停不下来。


class Boom(BaseException):
    """`except Exception` 接不住的那种（KeyboardInterrupt / SystemExit 一类）。"""


class ExplodingAgent:
    """一开跑就抛指定异常的假 Agent：专门试「跑挂了会不会留下幽灵」。"""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def run(self, task, **kwargs):
        raise self.error


class GhostRunTest(CancelRequests, StreamCase):
    """每条出口（含「连流都没开成」和 BaseException）都得把 Run 结掉。"""

    TASK = "任务甲"

    def setUp(self):
        # 这几条用例故意制造「服务端处理到一半抛到最外层」的场景，而
        # socketserver / threading 默认把这种异常连同栈打到 stderr（前者
        # handle_error，后者 excepthook）。异常正是被测对象，栈本身没有
        # 信息量——只把两处打印换成空操作：关连接、继续服务、线程退场这些
        # 行为一个字没动，那是 process_request_thread 的事
        for quiet in (
            patch.object(ApiServer, "handle_error"),
            patch("threading.excepthook"),
        ):
            quiet.start()
            self.addCleanup(quiet.stop)

    def wait_finished(self, count: int = 1) -> None:
        """等这一轮在登记表里落定（处理请求的是服务端线程）。"""
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.server.runs.finished_count >= count:
                return
            time.sleep(0.01)
        self.fail(f"等不到 Run 落终态：{self.server.runs.active_count} 条还挂着")

    def raw_post(self, path: str, payload: dict) -> socket.socket:
        """用裸 socket 发一个请求就走：不读响应（响应可能根本不成形）。

        测的是**服务端**收没收尾，客户端读到什么是另一回事——那些写失败
        的用例里，连接本来就断在半路。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = b"".join(
            [
                f"POST {path} HTTP/1.1\r\n".encode(),
                b"Host: 127.0.0.1\r\n",
                b"Content-Type: application/json\r\n",
                f"Content-Length: {len(body)}\r\n\r\n".encode(),
                body,
            ]
        )
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        self.addCleanup(conn.close)
        conn.sendall(request)
        return conn

    def assert_nothing_active(self, session_id: str) -> None:
        """这一轮结了：活跃表里一条不剩，而且取消再也打不到幽灵。"""
        self.assertEqual(self.server.runs.active_count, 0)
        self.assertEqual(self.server.runs.finished_count, 1)
        self.assertIsNone(self.server.runs.active_for(session_id))

        status, payload = self.cancel(session_id)

        self.assertEqual(status, 404)  # 没有「当前这一轮」可取消
        self.assertFalse(payload["success"])

    def test_a_stream_that_cannot_open_still_finishes_its_run(self):
        # 客户端在建流之前就断：写响应头那一下炸在 try 外面的话，下面每个
        # 出口的 finish 一次都轮不到——这条测试盯的就是它
        self.start(StubAgent())
        session_id = self.sessions.create().id

        with patch.object(
            ApiHandler, "_begin_event_stream", side_effect=BrokenPipeError("断了")
        ):
            self.raw_post(
                "/api/run/stream", {"task": self.TASK, "session_id": session_id}
            )
            self.wait_finished()

        self.assert_nothing_active(session_id)

    def test_a_base_exception_does_not_leave_a_ghost_behind(self):
        # except Exception 接不住的那种：兜底的 finally 也得把它结掉
        self.start(ExplodingAgent(Boom("线程要没了")))
        session_id = self.sessions.create().id

        with patch.object(ApiServer, "handle_error", lambda *args: None):
            self.raw_post(
                "/api/run/stream", {"task": self.TASK, "session_id": session_id}
            )
            self.wait_finished()

        self.assert_nothing_active(session_id)

    def test_a_normal_stream_leaves_nothing_active(self):
        # 反过来钉一下：正常跑完的也必须是这个形状（论据是同一个）
        self.start(StubAgent())
        session_id = self.sessions.create().id

        events = self.stream_events(self.TASK, session_id=session_id)

        self.assertEqual(events[-1][0], "completed")
        self.assert_nothing_active(session_id)

    def test_the_plain_run_route_finishes_on_a_base_exception_too(self):
        # 非流式那条路（POST /api/run）是同一条规矩
        self.start(ExplodingAgent(Boom("线程要没了")))
        session_id = self.sessions.create().id

        with patch.object(ApiServer, "handle_error", lambda *args: None):
            self.raw_post("/api/run", {"task": self.TASK, "session_id": session_id})
            self.wait_finished()

        self.assert_nothing_active(session_id)
