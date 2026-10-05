"""HTTP API 测试：四条路由、错误转换、会话复用、入口。

起的是真服务（127.0.0.1 上挑一个空闲端口），请求走真 socket、真 HTTP——
只是 Agent 换成测试替身：正常路径用 Agent + FakeLLMClient，别的用假 Agent。
全程不打 DeepSeek，也不依赖网络。

会话本身的行为（隔离、过期、上限）在 test_sessions.py；HTTP 上的会话语义
在 test_api_sessions.py；Origin / Token / Content-Type 三道检查在
test_api_security.py。这里钉的是基础盘：参数怎么解析、错误变成什么状态码、
一个会话怎么复用同一只 Agent。

ApiCase.start 默认给一只固定的 Agent（每次新建的会话都拿到它），只关心
HTTP 的用例照旧；要验「每个会话一只新 Agent」的用 factory=。
"""

import contextlib
import http.client
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent, AgentError
from api import create_server
from api.__main__ import main as api_entry
from api.sessions import SessionManager
from config import ApiConfig, ConfigError
from llm import LLMError
from tests.test_agent import FakeLLMClient, tool_call_reply
from tools import CalculatorTool, CallGraphTool, ToolRegistry


class CountingAgent:
    """假 Agent：不真跑任务，只记下收到的 task，回答固定。

    run 收得下可选参数（接口层会传 cancellation，见 13.4），只是不用它。
    """

    def __init__(self, answer: str = "好的") -> None:
        self.answer = answer
        self.tasks: list[str] = []

    def run(self, task: str, cancellation=None) -> str:
        self.tasks.append(task)
        return self.answer


class ApiCase(unittest.TestCase):
    """起一个真服务（port=0 让系统挑端口），请求用 http.client 发。

    agent=：固定一只，每次新建的会话都拿到它——只测 HTTP 层的用例用这个；
    factory=：不带参数、每调一次造一只新 Agent 的可调用对象——测会话隔离
    的用例用这个。两者给一个就行。
    """

    def start(
        self,
        agent=None,
        *,
        factory=None,
        ttl=3600.0,
        max_sessions=32,
        cors_origins=(),
        token=None,
        clock=None,
        run_timeout=0.0,
    ) -> None:
        if factory is None:
            factory = lambda: agent  # noqa: E731 —— 固定一只，写成 def 反而绕
        sessions = SessionManager(
            factory,
            ttl=ttl,
            max_sessions=max_sessions,
            clock=time.monotonic if clock is None else clock,
        )
        server = create_server(
            sessions,
            port=0,
            cors_origins=cors_origins,
            token=token,
            run_timeout=run_timeout,
        )
        # addCleanup 是后进先出：这样注册下来，先 shutdown（等 serve_forever
        # 退出来）再 close（关 socket）
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        # poll_interval 调小：serve_forever 每轮都查一次「该停了吗」，默认 0.5 秒
        # 是给长跑的服务省 CPU 的；测试里每个用例都起停一次，等不起
        threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
        ).start()

        self.server = server
        self.sessions = server.sessions
        self.port = server.server_address[1]

    def call(
        self,
        method: str,
        path: str,
        body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """发一次请求，返回 (状态码, 响应头, 原始 body)。"""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def call_json(
        self,
        method: str,
        path: str,
        body: bytes | str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict]:
        """发一次请求，顺手验「响应一定是 JSON」，返回 (状态码, 解析结果)。

        每个用例都从这儿走，等于每个响应都验一遍格式。
        """
        status, response_headers, raw = self.call(method, path, body, headers)
        self.assertEqual(
            response_headers.get("Content-Type"),
            "application/json; charset=utf-8",
        )
        return status, json.loads(raw.decode("utf-8"))

    def post_task(self, task: str, session_id: str | None = None) -> tuple[int, dict]:
        """发一个 /api/run 请求（body 一律 UTF-8 编码；session_id 可省）。"""
        payload: dict[str, object] = {"task": task}
        if session_id is not None:
            payload["session_id"] = session_id
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.call_json(
            "POST", "/api/run", body, {"Content-Type": "application/json"}
        )

    def run_task(self, task: str, session_id: str | None = None) -> dict:
        """发一个正常任务，要求 200 + success，返回响应体。"""
        status, payload = self.post_task(task, session_id)
        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        return payload


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


class HealthTest(ApiCase):
    def setUp(self):
        self.agent = CountingAgent()
        self.start(self.agent)

    def test_health_is_ok(self):
        status, payload = self.call_json("GET", "/api/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_health_does_not_touch_the_agent(self):
        self.call_json("GET", "/api/health")

        self.assertEqual(self.agent.tasks, [])

    def test_a_trailing_slash_is_the_same_path(self):
        status, payload = self.call_json("GET", "/api/health/")

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})


# ---------------------------------------------------------------------------
# 正常请求
# ---------------------------------------------------------------------------


class RunTest(ApiCase):
    """正常路径用真 Agent + 脚本化的假客户端，不联网。"""

    def setUp(self):
        self.client = FakeLLMClient(["任务已完成。"])
        self.agent = Agent(client=self.client)
        self.start(self.agent)

    def test_a_task_gets_an_answer_and_a_session_id(self):
        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 200)
        self.assertEqual(payload["success"], True)
        self.assertEqual(payload["answer"], "任务已完成。")
        # session_id 由服务端发：带着它，下一次请求才接得上这段对话
        self.assertIsInstance(payload["session_id"], str)
        self.assertTrue(payload["session_id"])

    def test_the_task_really_went_through_the_agent(self):
        self.run_task("任务甲")

        # 回答进了会话历史：走的是 Agent.run 那条真路，不是 HTTP 层自己编的
        self.assertEqual(self.agent.messages[1]["content"], "任务甲")
        self.assertEqual(self.agent.messages[2]["content"], "任务已完成。")

    def test_a_second_task_in_the_same_session_stacks_on_the_history(self):
        first = self.run_task("任务甲")
        second = self.run_task("任务乙", session_id=first["session_id"])

        # 同一个 session_id 接着用：两个任务在同一条会话历史里
        self.assertEqual(second["session_id"], first["session_id"])
        self.assertEqual(self.agent.messages[1]["content"], "任务甲")
        self.assertEqual(self.agent.messages[3]["content"], "任务乙")
        self.assertEqual(len(self.client.calls), 2)

    def test_each_session_gets_its_own_id(self):
        first = self.run_task("任务甲")
        second = self.run_task("任务乙")

        self.assertNotEqual(first["session_id"], second["session_id"])

    def test_a_connection_can_serve_two_requests(self):
        # HTTP/1.1 长连接：Content-Length 写对了才敢复用一条连接
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)

        conn.request("GET", "/api/health")
        first = conn.getresponse()
        first.read()
        conn.request("GET", "/api/health")
        second = conn.getresponse()
        body = second.read()

        self.assertEqual((first.status, second.status), (200, 200))
        self.assertEqual(json.loads(body.decode("utf-8")), {"status": "ok"})


# ---------------------------------------------------------------------------
# 参数错误：一律 400，绝不到 Agent 那儿
# ---------------------------------------------------------------------------


class BadRequestTest(ApiCase):
    def setUp(self):
        self.agent = CountingAgent()
        self.start(self.agent)

    def test_a_missing_task(self):
        status, payload = self.call_json(
            "POST", "/api/run", b'{"other": 1}', {"Content-Type": "application/json"}
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["success"])
        self.assertIn("task", payload["answer"])

    def test_an_empty_task(self):
        status, payload = self.post_task("")

        self.assertEqual(status, 400)
        self.assertIn("空", payload["answer"])

    def test_a_whitespace_task(self):
        status, payload = self.post_task("   ")

        self.assertEqual(status, 400)
        self.assertIn("空", payload["answer"])

    def test_a_task_that_is_not_a_string(self):
        status, payload = self.call_json(
            "POST", "/api/run", b'{"task": 123}', {"Content-Type": "application/json"}
        )

        self.assertEqual(status, 400)
        self.assertIn("字符串", payload["answer"])

    def test_a_body_that_is_not_an_object(self):
        status, payload = self.call_json(
            "POST", "/api/run", b"[]", {"Content-Type": "application/json"}
        )

        self.assertEqual(status, 400)
        self.assertIn("JSON 对象", payload["answer"])

    def test_an_empty_body(self):
        status, payload = self.call_json(
            "POST", "/api/run", b"", {"Content-Type": "application/json"}
        )

        self.assertEqual(status, 400)
        self.assertFalse(payload["success"])

    def test_a_body_that_is_not_json(self):
        status, payload = self.call_json(
            "POST",
            "/api/run",
            "不是 JSON".encode("utf-8"),
            {"Content-Type": "application/json"},
        )

        self.assertEqual(status, 400)
        self.assertIn("JSON", payload["answer"])

    def test_broken_json(self):
        status, payload = self.call_json(
            "POST", "/api/run", b'{"task": ', {"Content-Type": "application/json"}
        )

        self.assertEqual(status, 400)
        self.assertIn("JSON", payload["answer"])

    def test_a_body_that_is_too_big(self):
        # 只发头、不真发那么大的 body：Content-Length 就够服务器判断了，
        # 也免得测试和服务器在 1 MiB 上抢 socket 缓冲区
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(
                b"POST /api/run HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 2000000\r\n"
                b"\r\n"
            )
            raw = b""
            while True:  # 服务器回完就断开（Connection: close），读到 EOF 为止
                chunk = sock.recv(4096)
                if not chunk:
                    break
                raw += chunk

        head, _, body = raw.partition(b"\r\n\r\n")
        self.assertIn(b" 413 ", head)
        self.assertIn(b"Connection: close", head)
        self.assertFalse(json.loads(body.decode("utf-8"))["success"])

    def test_a_bad_request_never_reaches_the_agent(self):
        for body in (b"", b"[]", b'{"task": ""}', b"not json"):
            self.call_json(
                "POST", "/api/run", body, {"Content-Type": "application/json"}
            )

        self.assertEqual(self.agent.tasks, [])


# ---------------------------------------------------------------------------
# 路由：404 / 405，出口都是 JSON
# ---------------------------------------------------------------------------


class RoutingTest(ApiCase):
    def setUp(self):
        self.agent = CountingAgent()
        self.start(self.agent)

    def test_an_unknown_path_is_404(self):
        status, payload = self.call_json("GET", "/api/nowhere")

        self.assertEqual(status, 404)
        self.assertFalse(payload["success"])

    def test_a_wrong_method_is_405(self):
        # 路径是 /api/run，但它是 POST-only
        status, headers, raw = self.call("GET", "/api/run")

        self.assertEqual(status, 405)
        self.assertEqual(headers.get("Allow"), "POST")
        self.assertIn("POST", json.loads(raw.decode("utf-8"))["answer"])

    def test_health_is_get_only(self):
        status, headers, _ = self.call("POST", "/api/health", b"{}")

        self.assertEqual(status, 405)
        self.assertEqual(headers.get("Allow"), "GET")

    def test_every_response_is_json(self):
        cases = [
            ("GET", "/api/health", None),
            ("GET", "/api/run", None),
            ("GET", "/api/nowhere", None),
            ("POST", "/api/health", b"{}"),
            ("POST", "/api/run", b"not json"),
            ("POST", "/api/nowhere", b"{}"),
            ("PUT", "/api/run", b"{}"),
        ]
        for method, path, body in cases:
            with self.subTest(method=method, path=path):
                # call_json 自己会验 Content-Type 和「能解析成 JSON」
                status, payload = self.call_json(method, path, body)

                self.assertIsInstance(payload, dict)
                if status >= 400:
                    self.assertFalse(payload["success"])


# ---------------------------------------------------------------------------
# Agent 异常：状态码分开，服务不崩
# ---------------------------------------------------------------------------


class FailingAgent:
    """一跑就抛指定异常的假 Agent。"""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def run(self, task: str, cancellation=None) -> str:
        raise self.error


class AgentFailureTest(ApiCase):
    def test_a_model_failure_is_502(self):
        # 真 Agent + 假客户端：模型这层失败走的是 Agent.run 抛 LLMError 那条真路
        self.start(Agent(client=FakeLLMClient([LLMError("模型连不上")])))

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 502)
        self.assertFalse(payload["success"])
        self.assertIn("模型连不上", payload["answer"])
        # 失败响应也带 session_id：客户端知道这段对话还在
        self.assertIsInstance(payload["session_id"], str)

    def test_an_agent_failure_is_500(self):
        # 模型一直要求调工具，撞上 max_steps：Agent 抛 AgentError
        client = FakeLLMClient([tool_call_reply(("calculator", '{"expression": "1+1"}'))])
        agent = Agent(client=client, tools=ToolRegistry([CalculatorTool()]), max_steps=2)
        self.start(agent)

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 500)
        self.assertFalse(payload["success"])
        self.assertIn("Agent", payload["answer"])

    def test_an_unexpected_error_is_500(self):
        self.start(FailingAgent(RuntimeError("意外")))

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 500)
        self.assertEqual(payload["answer"], "RuntimeError: 意外")

    def test_the_service_keeps_serving_after_a_failure(self):
        # 假 Agent 第一次抛、第二次正常：服务不能因为一次失败就没了
        class SometimesFailing:
            def __init__(self):
                self.calls = 0

            def run(self, task, cancellation=None):
                self.calls += 1
                if self.calls == 1:
                    raise AgentError("第一次炸了")
                return "第二次好了"

        self.start(SometimesFailing())

        first = self.post_task("任务甲")
        second = self.post_task("任务乙")
        health = self.call_json("GET", "/api/health")

        self.assertEqual(first[0], 500)
        self.assertEqual(second[0], 200)
        self.assertTrue(second[1]["success"])
        self.assertEqual(second[1]["answer"], "第二次好了")
        self.assertEqual(health[0], 200)


# ---------------------------------------------------------------------------
# 会话复用：同一会话一只 Agent，索引不重建
# ---------------------------------------------------------------------------


class ReuseTest(ApiCase):
    def test_one_session_reuses_the_same_agent(self):
        agent = CountingAgent()
        self.start(agent)

        first = self.run_task("任务甲")
        session_id = first["session_id"]
        self.run_task("任务乙", session_id=session_id)
        self.run_task("任务丙", session_id=session_id)

        # 三次请求都落在同一个会话的同一只 Agent 上：没有哪次偷偷另建一只
        self.assertIs(self.sessions.get(session_id).agent, agent)
        self.assertEqual(agent.tasks, ["任务甲", "任务乙", "任务丙"])

    def test_the_call_graph_index_is_built_once_for_many_sessions(self):
        # 真的工具、真的索引：两个新会话各查一次 call_graph——两只 Agent，
        # 但注册表（连着索引缓存）是共享的，建图只该发生一次
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve()
        (root / "m.py").write_text(
            "def helper():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def caller():\n"
            "    return helper()\n",
            encoding="utf-8",
            newline="",
        )

        tool = CallGraphTool(root=root)
        registry = ToolRegistry([tool])
        client = FakeLLMClient(
            [
                tool_call_reply(
                    ("call_graph", '{"function": "helper", "direction": "callers"}')
                ),
                "第一次查完了。",
                tool_call_reply(
                    ("call_graph", '{"function": "helper", "direction": "callers"}')
                ),
                "第二次查完了。",
            ]
        )
        created: list[Agent] = []

        def make_agent() -> Agent:
            created.append(Agent(client=client, tools=registry))
            return created[-1]

        self.start(factory=make_agent)

        with patch.object(tool.index, "build", wraps=tool.index.build) as build:
            first = self.run_task("任务甲")  # 不带 session_id：新建会话
            second = self.run_task("任务乙")  # 再新建一个，不重扫项目

        self.assertEqual(first["answer"], "第一次查完了。")
        self.assertEqual(second["answer"], "第二次查完了。")
        # 两个新会话 = 两只新 Agent（会话状态不共享）
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        # 工具真跑了：回灌给模型的观察里有 caller 这个调用者
        outputs = [m["content"] for m in created[0].messages if m["role"] == "tool"]
        self.assertTrue(any("caller" in text for text in outputs))
        # 但图只建了一次（共享的是工具注册表）
        self.assertEqual(build.call_count, 1)


# ---------------------------------------------------------------------------
# 并发：同一会话排队跑，健康检查不等
# ---------------------------------------------------------------------------


class ConcurrencyTest(ApiCase):
    def test_health_answers_while_a_task_is_running(self):
        started = threading.Event()
        release = threading.Event()

        class SlowAgent:
            def run(self, task, cancellation=None):
                started.set()
                release.wait(5)
                return "慢回答"

        self.start(SlowAgent())
        worker = threading.Thread(target=self.post_task, args=("任务甲",), daemon=True)
        worker.start()
        self.assertTrue(started.wait(5), "任务没开始跑")

        # 会话忙着的时候健康检查照样答：它不碰任何会话锁
        status, payload = self.call_json("GET", "/api/health")
        release.set()
        worker.join(5)

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_two_tasks_in_one_session_do_not_run_at_the_same_time(self):
        inside = []
        peak = []

        class SlowAgent:
            def run(self, task, cancellation=None):
                inside.append(task)
                peak.append(len(inside))
                time.sleep(0.05)
                inside.pop()
                return task

        self.start(SlowAgent())
        session_id = self.run_task("任务甲")["session_id"]
        workers = [
            threading.Thread(
                target=self.post_task, args=(f"任务{index}", session_id), daemon=True
            )
            for index in (1, 2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(5)

        # 共享的是会话历史，同一会话的两个任务同时跑会互相踩：峰值只能是 1
        self.assertEqual(max(peak), 1)


# ---------------------------------------------------------------------------
# 入口：python -m api
# ---------------------------------------------------------------------------


class EntryPointTest(unittest.TestCase):
    def test_the_factory_is_built_once_and_sessions_come_from_it(self):
        built = []
        served = []

        class FakeFactory:
            def new_agent(self):
                built.append(CountingAgent())
                return built[-1]

        class FakeServer:
            def __init__(self, sessions):
                self.sessions = sessions
                self.server_address = ("127.0.0.1", 8000)

            def serve_forever(self):
                served.append("serving")

            def server_close(self):
                served.append("closed")

        servers = []

        def fake_create_server(sessions, host=None, port=None, **kwargs):
            servers.append(FakeServer(sessions))
            return servers[-1]

        with patch("api.__main__.build_agent_factory", FakeFactory):
            with patch("api.__main__.load_api_config", return_value=ApiConfig()):
                with patch("api.__main__.create_server", fake_create_server):
                    # basicConfig 是全局的：真调一次，后面所有用例的请求日志都会
                    # 跟着打开——入口测试不负责验日志配置，别动整个进程
                    with patch("logging.basicConfig"):
                        with contextlib.redirect_stdout(io.StringIO()):
                            code = api_entry(["--port", "0"])

        self.assertEqual(code, 0)
        self.assertEqual(served, ["serving", "closed"])
        # 起服务时不预先造 Agent：第一个会话来了才造（懒着来）
        self.assertEqual(built, [])
        session = servers[0].sessions.create()
        self.assertIs(session.agent, built[0])

    def test_a_bad_llm_config_stops_before_serving(self):
        with patch("api.__main__.build_agent_factory", side_effect=ConfigError("配置不对")):
            with patch("api.__main__.load_api_config", return_value=ApiConfig()):
                with patch("api.__main__.create_server") as create:
                    with contextlib.redirect_stderr(io.StringIO()) as stderr:
                        code = api_entry([])

        self.assertEqual(code, 1)
        create.assert_not_called()
        self.assertIn("配置不对", stderr.getvalue())

    def test_a_bad_api_config_stops_before_serving(self):
        with patch("api.__main__.build_agent_factory", return_value=object()):
            with patch(
                "api.__main__.load_api_config",
                side_effect=ConfigError("会话参数不对"),
            ):
                with patch("api.__main__.create_server") as create:
                    with contextlib.redirect_stderr(io.StringIO()) as stderr:
                        code = api_entry([])

        self.assertEqual(code, 1)
        create.assert_not_called()
        self.assertIn("会话参数不对", stderr.getvalue())
