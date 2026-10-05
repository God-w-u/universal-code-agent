"""会话管理测试：隔离、过期、上限、删除——不起 HTTP，直接调 SessionManager。

Agent 用真 Agent + 假客户端（对话历史看得见是隔离的关键）或记账用的假
Agent；时间用假时钟（FakeClock），「过多久」不用真等。

HTTP 上的会话语义在 test_api_sessions.py；这里钉的是这一层自己的行为：
session_id -> Session 的表怎么建、怎么清、怎么不串话。
"""

import threading
import unittest

from agent import Agent
from api.sessions import DEFAULT_MAX_SESSIONS, DEFAULT_TTL, Session, SessionManager
from tests.test_agent import FakeLLMClient
from tools import CalculatorTool, ToolRegistry


class FakeClock:
    """假时钟：手动拨快，测 TTL 不用真睡。"""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingAgent:
    """假 Agent：只记 task，回答固定（会话测试不关心 Agent 内部怎么跑）。"""

    def __init__(self, answer: str = "好的") -> None:
        self.answer = answer
        self.tasks: list[str] = []

    def run(self, task: str) -> str:
        self.tasks.append(task)
        return self.answer


class SessionTest(unittest.TestCase):
    """Session 本身：id、Agent、时间、自己那把锁。"""

    def test_a_session_carries_its_own_agent_and_times(self):
        agent = RecordingAgent()
        session = Session(id="甲", agent=agent, created_at=1.0, last_used=2.0)

        self.assertEqual(session.id, "甲")
        self.assertIs(session.agent, agent)
        self.assertEqual(session.created_at, 1.0)
        self.assertEqual(session.last_used, 2.0)
        self.assertFalse(session.lock.locked())

    def test_every_session_gets_its_own_lock(self):
        first = Session(id="甲", agent=RecordingAgent(), created_at=0, last_used=0)
        second = Session(id="乙", agent=RecordingAgent(), created_at=0, last_used=0)

        self.assertIsNot(first.lock, second.lock)


class CreateTest(unittest.TestCase):
    def test_create_makes_a_new_agent_every_time(self):
        created: list[RecordingAgent] = []
        manager = SessionManager(lambda: created.append(RecordingAgent()) or created[-1])

        first = manager.create()
        second = manager.create()

        self.assertEqual(len(created), 2)  # 每会话一只新 Agent
        self.assertIsNot(first.agent, second.agent)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(manager), 2)

    def test_a_new_session_starts_touched(self):
        clock = FakeClock(100.0)
        manager = SessionManager(RecordingAgent, clock=clock)

        session = manager.create()

        self.assertEqual(session.created_at, 100.0)
        self.assertEqual(session.last_used, 100.0)


class IsolationTest(unittest.TestCase):
    """A、B 两个会话：对话历史绝不互相看见，共享件还是共享件。"""

    def setUp(self):
        self.client = FakeLLMClient(["回答甲。", "回答乙。"])
        self.registry = ToolRegistry([CalculatorTool()])
        self.manager = SessionManager(
            lambda: Agent(client=self.client, tools=self.registry)
        )

    def test_two_sessions_do_not_share_messages(self):
        a = self.manager.create()
        b = self.manager.create()

        a.agent.run("任务甲")
        b.agent.run("任务乙")

        self.assertEqual(a.agent.messages[1]["content"], "任务甲")
        self.assertEqual(a.agent.messages[2]["content"], "回答甲。")
        self.assertEqual(b.agent.messages[1]["content"], "任务乙")
        self.assertEqual(b.agent.messages[2]["content"], "回答乙。")
        # A 的历史里没有 B 的字，反过来也一样
        self.assertNotIn("任务乙", [m["content"] for m in a.agent.messages])
        self.assertNotIn("任务甲", [m["content"] for m in b.agent.messages])

    def test_agents_share_the_resources_but_not_the_history(self):
        a = self.manager.create()
        b = self.manager.create()

        self.assertIs(a.agent.tools, self.registry)  # 工具注册表共享
        self.assertIs(b.agent.tools, self.registry)
        self.assertIs(a.agent.client, b.agent.client)  # 模型客户端共享
        self.assertIsNot(a.agent.messages, b.agent.messages)  # 历史不共享


class GetTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.manager = SessionManager(RecordingAgent, clock=self.clock)

    def test_get_returns_the_same_session_and_touches_it(self):
        session = self.manager.create()
        self.clock.advance(10)

        found = self.manager.get(session.id)

        self.assertIs(found, session)
        self.assertEqual(session.last_used, 10.0)  # 刚被用过：重新计时

    def test_get_an_unknown_id_is_none(self):
        self.assertIsNone(self.manager.get("没有这个"))

    def test_acquire_without_an_id_creates(self):
        session = self.manager.acquire()

        self.assertEqual(len(self.manager), 1)
        self.assertIs(self.manager.acquire(session.id), session)

    def test_acquire_with_an_unknown_id_is_none_and_creates_nothing(self):
        # 认不出来的 id 绝不顶着它新建——session_id 一律服务端发
        self.assertIsNone(self.manager.acquire("没有这个"))
        self.assertEqual(len(self.manager), 0)


class TtlTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.manager = SessionManager(RecordingAgent, ttl=100.0, clock=self.clock)

    def test_expired_sessions_show_up_as_none(self):
        session = self.manager.create()
        self.clock.advance(101)

        self.assertIsNone(self.manager.get(session.id))
        self.assertEqual(len(self.manager), 0)

    def test_using_a_session_keeps_it_alive(self):
        session = self.manager.create()
        self.clock.advance(90)
        self.manager.get(session.id)  # 用了一下：重新计时
        self.clock.advance(90)

        # 距创建 180 秒了，但距上次使用才 90 秒：还没过期
        self.assertIs(self.manager.get(session.id), session)

    def test_the_boundary_is_strictly_greater(self):
        # 正好等于 ttl 不算过期（> ttl 才清），边界上的会话不会忽有忽无
        session = self.manager.create()
        self.clock.advance(100.0)

        self.assertIsNotNone(self.manager.get(session.id))

    def test_purge_expired_reports_what_it_cleaned(self):
        a = self.manager.create()
        self.clock.advance(50)
        b = self.manager.create()
        self.clock.advance(60)  # a 已 110 秒没人用，b 才 60 秒

        cleaned = self.manager.purge_expired()

        self.assertEqual(cleaned, [a.id])
        self.assertEqual(len(self.manager), 1)
        self.assertIsNotNone(self.manager.get(b.id))

    def test_creating_also_cleans_the_expired(self):
        self.manager.create()
        self.clock.advance(101)

        self.manager.create()

        self.assertEqual(len(self.manager), 1)  # 旧的那个顺手清了

    def test_a_running_session_is_not_purged(self):
        # 锁被握着 = 正在跑任务。把一条正在跑的会话判过期摘掉，跑的那一轮
        # 手里还攥着对象、会照常跑完，可客户端再也拿不到它（下一个请求
        # 404），那段对话等于凭空消失——而且它其实刚被用过
        session = self.manager.create()
        session.lock.acquire()
        self.addCleanup(session.lock.release)
        self.clock.advance(101)

        cleaned = self.manager.purge_expired()

        self.assertEqual(cleaned, [])
        self.assertEqual(len(self.manager), 1)
        self.assertIs(self.manager.get(session.id), session)

    def test_a_finished_run_that_outlived_the_ttl_is_kept_by_touch(self):
        # 跑一轮可能比 ttl 还长。跑完 touch 一次 =「刚用过」，客户端接着
        # 追问不该先被清掉再拿到 404
        session = self.manager.create()
        session.lock.acquire()
        self.clock.advance(101)  # 早过期了，但锁还握着
        self.manager.touch(session)  # 跑完了：服务端那条路会调这个
        session.lock.release()

        self.clock.advance(90)  # 距 touch 才 90 秒

        self.assertIs(self.manager.get(session.id), session)

    def test_purge_returns_after_a_running_session_lets_go(self):
        # 锁一松开，过期就是过期——「不清」只保护正在跑的那一小会儿
        session = self.manager.create()
        session.lock.acquire()
        self.clock.advance(101)
        self.assertEqual(self.manager.purge_expired(), [])

        session.lock.release()

        self.assertEqual(self.manager.purge_expired(), [session.id])


class TouchTest(unittest.TestCase):
    """touch()：跑完一轮记一次「刚用过」。"""

    def setUp(self):
        self.clock = FakeClock(100.0)
        self.manager = SessionManager(RecordingAgent, clock=self.clock)

    def test_touch_moves_last_used_forward(self):
        session = self.manager.create()
        self.clock.advance(30)

        self.manager.touch(session)

        self.assertEqual(session.last_used, 130.0)

    def test_touch_of_a_deleted_session_does_not_bring_it_back(self):
        session = self.manager.create()
        self.manager.delete(session.id)

        self.manager.touch(session)

        self.assertIsNone(self.manager.get(session.id))
        self.assertEqual(len(self.manager), 0)  # 没有又被塞回表里

    def test_touch_of_a_foreign_session_is_ignored(self):
        # 同一个 id 已经换了另一只（删了又建）：手里这只旧对象不该改到新那份
        session = self.manager.create()
        self.manager.delete(session.id)
        twin = Session(
            id=session.id, agent=RecordingAgent(), created_at=0.0, last_used=0.0
        )
        session.last_used = 55.0  # 摆一个能看出来的值

        self.manager.touch(session)

        self.assertEqual(session.last_used, 55.0)  # 没动
        self.assertIsNone(self.manager.get(twin.id))  # 表里本来就没有它


class MaxSessionsTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.manager = SessionManager(
            RecordingAgent, max_sessions=2, clock=self.clock
        )

    def test_the_least_recently_used_session_is_dropped(self):
        a = self.manager.create()
        self.clock.advance(1)
        b = self.manager.create()
        self.clock.advance(1)
        self.manager.get(a.id)  # A 刚被用过，B 最久没用
        self.clock.advance(1)

        c = self.manager.create()

        self.assertEqual(len(self.manager), 2)
        self.assertIs(self.manager.get(a.id), a)  # 最近用过的留着
        self.assertIsNone(self.manager.get(b.id))  # 最久没用的被丢了
        self.assertIsNotNone(self.manager.get(c.id))

    def test_a_busy_session_is_not_dropped(self):
        # A 手里握着锁（正在跑任务）：超量时先丢空闲的，不动它
        a = self.manager.create()
        b = self.manager.create()
        a.lock.acquire()
        try:
            self.manager.create()
        finally:
            a.lock.release()

        self.assertIs(self.manager.get(a.id), a)
        self.assertIsNone(self.manager.get(b.id))

    def test_when_everyone_is_busy_the_oldest_still_goes(self):
        # 没有空闲的可丢：只能还是丢最久的（它跑完自己会看到 404，
        # 但总比表无限涨强）
        a = self.manager.create()
        self.clock.advance(1)
        b = self.manager.create()
        a.lock.acquire()
        b.lock.acquire()
        try:
            c = self.manager.create()
        finally:
            a.lock.release()
            b.lock.release()

        self.assertEqual(len(self.manager), 2)
        self.assertIsNone(self.manager.get(a.id))
        self.assertIsNotNone(self.manager.get(c.id))


class DeleteTest(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager(RecordingAgent)

    def test_delete_removes_only_that_session(self):
        a = self.manager.create()
        b = self.manager.create()

        self.assertTrue(self.manager.delete(a.id))
        self.assertFalse(self.manager.delete(a.id))  # 再删一次：没有了
        self.assertIsNone(self.manager.get(a.id))
        self.assertIsNotNone(self.manager.get(b.id))

    def test_delete_does_not_touch_shared_resources(self):
        # 共享的注册表删完还在，新会话拿到的还是它
        registry = ToolRegistry([CalculatorTool()])
        client = FakeLLMClient(["好了。"])
        manager = SessionManager(lambda: Agent(client=client, tools=registry))

        a = manager.create()
        manager.delete(a.id)
        b = manager.create()

        self.assertIs(b.agent.tools, registry)


class ValidationTest(unittest.TestCase):
    def test_a_non_positive_ttl_is_rejected(self):
        for ttl in (0, 0.0, -1.0):
            with self.subTest(ttl=ttl):
                with self.assertRaises(ValueError):
                    SessionManager(RecordingAgent, ttl=ttl)

    def test_max_sessions_must_be_at_least_one(self):
        for max_sessions in (0, -1):
            with self.subTest(max_sessions=max_sessions):
                with self.assertRaises(ValueError):
                    SessionManager(RecordingAgent, max_sessions=max_sessions)

    def test_the_defaults_are_sane(self):
        manager = SessionManager(RecordingAgent)

        self.assertEqual(manager.ttl, DEFAULT_TTL)
        self.assertEqual(manager.max_sessions, DEFAULT_MAX_SESSIONS)
        self.assertGreater(DEFAULT_TTL, 0)
        self.assertGreaterEqual(DEFAULT_MAX_SESSIONS, 1)


class ConcurrencyTest(unittest.TestCase):
    def test_creating_from_many_threads_keeps_every_session(self):
        manager = SessionManager(RecordingAgent, max_sessions=200)
        created: list[str] = []
        bookkeeping = threading.Lock()

        def worker() -> None:
            session = manager.create()
            with bookkeeping:
                created.append(session.id)

        workers = [threading.Thread(target=worker) for _ in range(50)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(5)

        self.assertEqual(len(created), 50)
        self.assertEqual(len(set(created)), 50)  # id 不重
        self.assertEqual(len(manager), 50)
