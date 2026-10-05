"""HTTP 层上的会话：session_id 的发放与复用、过期、上限、删除、并发隔离。

起真服务、走真 HTTP（沿用 test_api.py 的 ApiCase）；这里钉的是接口层的
会话语义——SessionManager 自己的行为（TTL 边界、LRU 细节）在
test_sessions.py，安全边界在 test_api_security.py。
"""

import json
import threading
import time

from agent import Agent
from tests.test_agent import FakeLLMClient
from tests.test_api import ApiCase, CountingAgent
from tests.test_api_stream import StreamCase, StubAgent
from tests.test_sessions import FakeClock

JSON_HEADER = {"Content-Type": "application/json"}


class SessionIdTest(ApiCase):
    def test_a_request_without_an_id_gets_one(self):
        self.start(CountingAgent())

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 200)
        self.assertIsInstance(payload["session_id"], str)
        self.assertTrue(payload["session_id"])

    def test_two_requests_without_an_id_get_two_sessions(self):
        created: list[CountingAgent] = []

        def make_agent():
            created.append(CountingAgent())
            return created[-1]

        self.start(factory=make_agent)

        first = self.run_task("任务甲")
        second = self.run_task("任务乙")

        self.assertNotEqual(first["session_id"], second["session_id"])
        # 每个新会话一只自己的 Agent：任务没有落到别人头上
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        self.assertEqual(created[0].tasks, ["任务甲"])
        self.assertEqual(created[1].tasks, ["任务乙"])

    def test_a_known_id_continues_the_same_conversation(self):
        client = FakeLLMClient(["回答甲。", "回答乙。"])
        created: list[Agent] = []

        def make_agent():
            created.append(Agent(client=client))
            return created[-1]

        self.start(factory=make_agent)

        first = self.run_task("任务甲")
        second = self.run_task("任务乙", session_id=first["session_id"])

        self.assertEqual(second["session_id"], first["session_id"])
        self.assertEqual(len(created), 1)  # 没另建一只
        # 第二轮发给模型的消息里带着第一轮的内容：接的是同一段历史
        second_call = [m["content"] for m in client.calls[1]]
        self.assertIn("任务甲", second_call)

    def test_an_unknown_id_is_404_and_creates_nothing(self):
        created: list[CountingAgent] = []

        def make_agent():
            created.append(CountingAgent())
            return created[-1]

        self.start(factory=make_agent)

        status, payload = self.post_task("任务甲", session_id="没有这个")

        self.assertEqual(status, 404)
        self.assertFalse(payload["success"])
        self.assertIn("没有这个", payload["answer"])
        self.assertEqual(created, [])  # 没有顶着客户端给的 id 悄悄新建
        self.assertEqual(len(self.sessions), 0)

    def test_a_session_id_that_is_not_a_string_is_400(self):
        self.start(CountingAgent())

        for value in (123, ["甲"], {"id": "甲"}, True):
            with self.subTest(value=value):
                body = json.dumps(
                    {"task": "任务甲", "session_id": value}
                ).encode("utf-8")
                status, payload = self.call_json(
                    "POST", "/api/run", body, JSON_HEADER
                )

                self.assertEqual(status, 400)
                self.assertIn("字符串", payload["answer"])

    def test_an_empty_session_id_is_400(self):
        self.start(CountingAgent())
        body = json.dumps({"task": "任务甲", "session_id": "   "}).encode("utf-8")

        status, payload = self.call_json("POST", "/api/run", body, JSON_HEADER)

        self.assertEqual(status, 400)
        self.assertIn("不能是空的", payload["answer"])


class SessionExpiryTest(ApiCase):
    def test_an_expired_session_is_404(self):
        clock = FakeClock()
        self.start(CountingAgent(), ttl=100.0, clock=clock)

        first = self.run_task("任务甲")
        clock.advance(101)

        status, payload = self.post_task("任务乙", session_id=first["session_id"])

        self.assertEqual(status, 404)
        self.assertIn(first["session_id"], payload["answer"])

    def test_a_session_used_in_time_stays(self):
        clock = FakeClock()
        self.start(CountingAgent(), ttl=100.0, clock=clock)

        first = self.run_task("任务甲")
        clock.advance(90)
        self.run_task("任务乙", session_id=first["session_id"])
        clock.advance(90)  # 距创建 180 秒，但距上次使用才 90 秒

        third = self.run_task("任务丙", session_id=first["session_id"])

        self.assertEqual(third["session_id"], first["session_id"])


class LongRunTest(StreamCase):
    """跑得比 ttl 还长的一轮：跑完不许自己算过期（13 阶段回归）。

    继承 StreamCase（它就是 ApiCase 加拆帧那几个方法）：两条路由都要测。
    """

    def long_agent(self, clock: FakeClock, seconds: float):
        """跑的过程中把假时钟拨过去——「这一轮花的时间」就这么来的。"""

        class LongAgent:
            def run(self, task, cancellation=None):
                clock.advance(seconds)
                return "跑完了"

        return LongAgent()

    def test_a_run_that_outlives_the_ttl_keeps_its_session(self):
        clock = FakeClock()
        self.start(self.long_agent(clock, 101), ttl=100.0, clock=clock)

        first = self.run_task("任务甲")
        # 跑完那一刻距「上次使用」已经 101 秒了。服务端跑完补记一次，
        # 所以这一刻是它刚被用过——接着追问必须还是同一个会话
        second = self.run_task("任务乙", session_id=first["session_id"])

        self.assertEqual(second["session_id"], first["session_id"])

    def test_a_long_stream_keeps_its_session_too(self):
        clock = FakeClock()

        class LongStreamAgent:
            def run(self, task, **kwargs):
                clock.advance(101)
                return "跑完了"

        self.start(LongStreamAgent(), ttl=100.0, clock=clock)
        session_id = self.sessions.create().id

        # 流式那条路（_run_stream）走的是同一个 touch
        self.assertEqual(
            self.stream_events("任务甲", session_id=session_id)[-1][0], "completed"
        )
        status, payload = self.post_task("任务乙", session_id=session_id)

        self.assertEqual(status, 200)
        self.assertEqual(payload["session_id"], session_id)


class MaxSessionsTest(ApiCase):
    def test_a_new_session_evicts_the_least_recently_used(self):
        self.start(CountingAgent(), max_sessions=1)

        first = self.run_task("任务甲")
        second = self.run_task("任务乙")  # 超量：丢最久没用的（第一个）

        self.assertNotEqual(second["session_id"], first["session_id"])
        self.assertEqual(len(self.sessions), 1)
        status, _ = self.post_task("任务丙", session_id=first["session_id"])
        self.assertEqual(status, 404)


class DeleteSessionTest(ApiCase):
    def setUp(self):
        self.start(CountingAgent())

    def delete(self, payload: dict, headers: dict | None = None):
        body = json.dumps(payload).encode("utf-8")
        return self.call_json(
            "DELETE", "/api/sessions", body, headers or JSON_HEADER
        )

    def test_delete_removes_the_session(self):
        session_id = self.run_task("任务甲")["session_id"]

        status, payload = self.delete({"session_id": session_id})

        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {"success": True, "session_id": session_id, "deleted": True},
        )
        # 再拿它跑任务：这个会话没了
        status, _ = self.post_task("任务乙", session_id=session_id)
        self.assertEqual(status, 404)
        self.assertEqual(len(self.sessions), 0)

    def test_deleting_an_unknown_session_is_404(self):
        status, payload = self.delete({"session_id": "没有这个"})

        self.assertEqual(status, 404)
        self.assertFalse(payload["success"])

    def test_a_missing_session_id_is_400(self):
        status, payload = self.delete({})

        self.assertEqual(status, 400)
        self.assertIn("session_id", payload["answer"])

    def test_deleting_one_session_leaves_the_others(self):
        first = self.run_task("任务甲")
        second = self.run_task("任务乙")

        self.delete({"session_id": first["session_id"]})

        status, payload = self.post_task("任务丙", session_id=second["session_id"])
        self.assertEqual(status, 200)
        self.assertEqual(payload["session_id"], second["session_id"])

    def test_get_is_not_allowed(self):
        status, headers, _ = self.call("GET", "/api/sessions")

        self.assertEqual(status, 405)
        self.assertEqual(headers.get("Allow"), "DELETE")


class StreamSessionTest(StreamCase):
    """流式接口上的 session_id：发放、续用、未知 id 走普通 JSON 404。"""

    def test_a_stream_carries_the_id_and_continues_the_session(self):
        created: list[StubAgent] = []

        def make_agent():
            created.append(StubAgent())
            return created[-1]

        self.start(factory=make_agent)

        first = self.stream_events("任务甲")
        session_id = first[0][1]["session_id"]
        second = self.stream_events("任务乙", session_id=session_id)

        self.assertEqual(len(created), 1)  # 两条流一个会话：没另建
        self.assertEqual(created[0].tasks, ["任务甲", "任务乙"])
        self.assertEqual(second[-1][1]["session_id"], session_id)

    def test_an_unknown_id_gets_plain_json_404(self):
        self.start(StubAgent())

        status, headers, raw = self.stream("任务甲", session_id="没有这个")

        self.assertEqual(status, 404)
        self.assertEqual(
            headers.get("Content-Type"), "application/json; charset=utf-8"
        )
        self.assertFalse(json.loads(raw.decode("utf-8"))["success"])


class SessionConcurrencyTest(ApiCase):
    """不同会话各跑各的：没有一把全局锁把它们串起来。"""

    def test_two_sessions_run_at_the_same_time(self):
        barrier = threading.Barrier(2)
        inside: list[str] = []
        peak: list[int] = []

        class SlowAgent:
            def run(self, task, cancellation=None):
                inside.append(task)
                peak.append(len(inside))
                # 两个会话都跑进来了才一起往下走——被全局锁串起来的话，
                # 这里会等到超时
                barrier.wait(timeout=5)
                time.sleep(0.05)
                inside.pop()
                return task

        self.start(SlowAgent())
        session_ids = [self.sessions.create().id for _ in range(2)]
        results: list[tuple[int, dict]] = []

        def send(task: str, session_id: str) -> None:
            results.append(self.post_task(task, session_id))

        workers = [
            threading.Thread(target=send, args=("任务甲", session_ids[0]), daemon=True),
            threading.Thread(target=send, args=("任务乙", session_ids[1]), daemon=True),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)

        self.assertEqual(sorted(status for status, _ in results), [200, 200])
        self.assertEqual(max(peak), 2)  # 真同时跑
