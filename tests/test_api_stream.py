"""SSE 流式接口测试：帧格式、事件顺序、边跑边推、异常收场、和 /api/run 共处。

起服务、发请求都沿用 test_api.py（真 socket、真 HTTP）；Agent 全用替身
（FakeLLMClient 或假 Agent），不联网、不打 DeepSeek。

SSE 相关的单元件（帧怎么拼、Trace 回调怎么翻译）在 api/sse.py 与
test_trace.py；这里钉的是端到端的那条流。13.3 起流式接口也认 session_id
（可省）：start / completed / error 三个事件里都带着它。
"""

import http.client
import json
import threading

from agent import Agent, AgentError
from llm import LLMError
from tests.test_agent import FakeLLMClient, tool_call_reply
from tests.test_api import ApiCase
from tools import CalculatorTool, Tool, ToolRegistry


class EchoTool(Tool):
    """回显工具：返回正文故意又长又好认，用来验证「正文不进流」。"""

    name = "echo"
    description = "回显"
    parameters = {
        "text": {"type": "string", "description": "要回显的文本", "required": True},
    }

    BODY = "这里是工具的完整返回正文" * 20

    def execute(self, text: str) -> str:
        return f"{self.BODY}（{text}）"


class SlowTool(Tool):
    """卡住不返回的慢工具：验证事件是边跑边推，不是跑完一起给。"""

    name = "slow"
    description = "卡住一会儿"
    parameters = {}

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def execute(self) -> str:
        self.started.set()
        self.release.wait(5)
        return "慢结果"


class StubAgent:
    """假 Agent：run 收得下观察者和取消参数（流式接口会传），只记下 task。

    四个观察者（工具两个 / 模型两个，13.5 起接口层会传）都收下但不用。
    """

    def __init__(self, answer: str = "好的") -> None:
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
        return self.answer


class StreamCase(ApiCase):
    """流式请求的公共部分：发请求、拆帧。"""

    def stream(self, task=None, body=None, headers=None, session_id=None):
        if body is None:
            payload = {"task": task}
            if session_id is not None:
                payload["session_id"] = session_id
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.call(
            "POST",
            "/api/run/stream",
            body,
            headers or {"Content-Type": "application/json"},
        )

    def parse(self, raw: bytes):
        """拆成 [(event, data), ...]，顺手验每帧的写法。

        规格里「每个事件都是合法 JSON data」这条就在这儿落地：data 那行
        直接 json.loads，解析不了这个用例就红。也验了「一帧两行 + 空行」。
        """
        events = []
        for block in raw.decode("utf-8").split("\n\n"):
            if not block:
                continue
            lines = block.split("\n")
            self.assertEqual(len(lines), 2, block)
            self.assertTrue(lines[0].startswith("event: "), block)
            self.assertTrue(lines[1].startswith("data: "), block)
            events.append(
                (lines[0][len("event: ") :], json.loads(lines[1][len("data: ") :]))
            )
        return events

    def stream_events(self, task: str, session_id=None):
        """发一个流式请求，要求 200 + SSE 头，返回拆好的事件列表。"""
        status, headers, raw = self.stream(task, session_id=session_id)
        self.assertEqual(status, 200)
        self.assertEqual(
            headers.get("Content-Type"), "text/event-stream; charset=utf-8"
        )
        return self.parse(raw)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


class StreamFormatTest(StreamCase):
    """真 Agent + 一次工具调用：start → tool_start → tool_result → completed。"""

    def setUp(self):
        client = FakeLLMClient(
            [tool_call_reply(("echo", '{"text": "任务甲"}')), "任务完成了。"]
        )
        self.start(Agent(client=client, tools=ToolRegistry([EchoTool()])))

    def test_the_response_is_an_event_stream(self):
        status, headers, _ = self.stream("任务甲")

        self.assertEqual(status, 200)
        self.assertEqual(
            headers.get("Content-Type"), "text/event-stream; charset=utf-8"
        )
        # 没有 Content-Length，靠关连接收尾
        self.assertEqual(headers.get("Connection"), "close")

    def test_start_and_completed_wrap_the_run(self):
        events = self.stream_events("任务甲")

        # start 带任务原文和 session_id，completed 带最终回答和同一个 id
        name, data = events[0]
        self.assertEqual(name, "start")
        self.assertEqual(data["task"], "任务甲")
        session_id = data["session_id"]
        self.assertIsInstance(session_id, str)

        name, data = events[-1]
        self.assertEqual(name, "completed")
        self.assertTrue(data["success"])
        self.assertEqual(data["answer"], "任务完成了。")
        self.assertEqual(data["session_id"], session_id)

    def test_every_tool_call_gets_a_start_and_a_result(self):
        events = self.stream_events("任务甲")

        self.assertEqual(
            [name for name, _ in events],
            ["start", "tool_start", "tool_result", "completed"],
        )
        self.assertEqual(
            events[1][1], {"tool": "echo", "arguments": {"text": "任务甲"}}
        )
        result = events[2][1]
        self.assertEqual(result["tool"], "echo")
        self.assertTrue(result["success"])
        self.assertEqual(result["error"], "")
        self.assertGreaterEqual(result["duration_seconds"], 0)

    def test_the_tool_output_is_not_pushed_to_the_client(self):
        _, _, raw = self.stream("任务甲")

        text = raw.decode("utf-8")
        self.assertIn("echo", text)  # 工具调用本身看得到
        self.assertNotIn(EchoTool.BODY, text)  # 但返回正文一个字都没有
        # tool_result 里也没有放正文的键
        result = self.parse(raw)[2][1]
        self.assertNotIn("content", result)


class ArgumentSummaryTest(StreamCase):
    def test_a_long_argument_is_cut_short(self):
        # edit_file 的 content 可能是整份源码：参数值只推摘要
        long_text = "甲" * 500
        client = FakeLLMClient(
            [
                tool_call_reply(
                    ("echo", json.dumps({"text": long_text}, ensure_ascii=False))
                ),
                "完事。",
            ]
        )
        self.start(Agent(client=client, tools=ToolRegistry([EchoTool()])))

        events = self.stream_events("任务甲")

        argument = events[1][1]["arguments"]["text"]
        self.assertTrue(argument.endswith("…"))
        self.assertLessEqual(len(argument), 121)
        self.assertNotIn(long_text, argument)


# ---------------------------------------------------------------------------
# 异常收场：error 事件 + 正常关流
# ---------------------------------------------------------------------------


class StreamFailureTest(StreamCase):
    def test_an_agent_failure_ends_the_stream_with_an_error_event(self):
        # 模型一直要求调工具，撞上 max_steps：Agent 抛 AgentError
        client = FakeLLMClient(
            [tool_call_reply(("calculator", '{"expression": "1+1"}'))]
        )
        self.start(
            Agent(client=client, tools=ToolRegistry([CalculatorTool()]), max_steps=2)
        )

        events = self.stream_events("任务甲")

        names = [name for name, _ in events]
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-1], "error")  # 流是正常收尾的，不是断的
        self.assertNotIn("completed", names)
        self.assertIn("Agent", events[-1][1]["message"])
        # error 也带 session_id：和 start / completed 对齐，前端好对号入座
        self.assertEqual(events[-1][1]["session_id"], events[0][1]["session_id"])

    def test_a_model_failure_ends_the_stream_with_an_error_event(self):
        self.start(Agent(client=FakeLLMClient([LLMError("模型连不上")])))

        events = self.stream_events("任务甲")

        self.assertEqual(events[-1][0], "error")
        self.assertIn("模型连不上", events[-1][1]["message"])

    def test_the_server_keeps_serving_after_a_streamed_failure(self):
        class SometimesFailing(StubAgent):
            def run(
                self,
                task,
                on_tool_start=None,
                on_tool_end=None,
                cancellation=None,
                on_llm_start=None,
                on_llm_end=None,
            ):
                if not self.tasks:
                    self.tasks.append(task)
                    raise AgentError("第一次炸了")
                return super().run(
                    task,
                    on_tool_start,
                    on_tool_end,
                    cancellation,
                    on_llm_start,
                    on_llm_end,
                )

        self.start(SometimesFailing())

        first = self.stream_events("任务甲")
        second = self.stream_events("任务乙")
        health = self.call_json("GET", "/api/health")

        self.assertEqual(first[-1][0], "error")
        self.assertEqual(second[-1][0], "completed")
        self.assertTrue(second[-1][1]["success"])
        self.assertEqual(second[-1][1]["answer"], "好的")
        self.assertEqual(health, (200, {"status": "ok"}))


# ---------------------------------------------------------------------------
# 非法请求：还没开头，回普通 JSON
# ---------------------------------------------------------------------------


class StreamBadRequestTest(StreamCase):
    def setUp(self):
        self.agent = StubAgent()
        self.start(self.agent)

    def test_a_body_that_is_not_json_gets_plain_json(self):
        status, headers, raw = self.stream(body=b"not json")

        self.assertEqual(status, 400)
        self.assertEqual(
            headers.get("Content-Type"), "application/json; charset=utf-8"
        )
        self.assertFalse(json.loads(raw.decode("utf-8"))["success"])

    def test_a_missing_task_gets_plain_json(self):
        status, _, raw = self.stream(body=b"{}")

        self.assertEqual(status, 400)
        self.assertIn("task", json.loads(raw.decode("utf-8"))["answer"])

    def test_an_empty_task_gets_plain_json(self):
        status, _, _ = self.stream(body=json.dumps({"task": "  "}).encode("utf-8"))

        self.assertEqual(status, 400)

    def test_a_bad_request_never_reaches_the_agent(self):
        self.stream(body=b"not json")
        self.stream(body=b"{}")

        self.assertEqual(self.agent.tasks, [])

    def test_a_wrong_method_is_405(self):
        status, headers, _ = self.call("GET", "/api/run/stream")

        self.assertEqual(status, 405)
        self.assertEqual(headers.get("Allow"), "POST")


# ---------------------------------------------------------------------------
# 和 /api/run 共处：同一个会话、同一把锁
# ---------------------------------------------------------------------------


class StreamReuseTest(StreamCase):
    def test_both_endpoints_reuse_the_same_session(self):
        created: list[StubAgent] = []

        def make_agent():
            created.append(StubAgent())
            return created[-1]

        self.start(factory=make_agent)

        status, plain = self.post_task("任务甲")  # 13.1 的普通接口
        session_id = plain["session_id"]
        events = self.stream_events("任务乙", session_id=session_id)

        self.assertEqual(status, 200)
        self.assertTrue(plain["success"])
        self.assertEqual(plain["answer"], "好的")
        self.assertEqual(events[-1][0], "completed")
        self.assertEqual(events[-1][1]["answer"], "好的")
        self.assertEqual(events[-1][1]["session_id"], session_id)
        # 两条接口、一个会话：只造了一只 Agent，任务按顺序落在同一段历史里
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].tasks, ["任务甲", "任务乙"])

    def test_a_plain_run_after_a_stream_still_works(self):
        # 流式那条连接以 Connection: close 收尾，不影响下一个普通请求
        self.start(StubAgent())

        self.stream_events("任务甲")
        status, payload = self.post_task("任务乙")

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["answer"], "好的")


# ---------------------------------------------------------------------------
# 实时性：事件是推出来的，不是跑完一起给的
# ---------------------------------------------------------------------------


class StreamLivenessTest(StreamCase):
    def test_events_arrive_while_the_tool_is_still_running(self):
        started = threading.Event()
        release = threading.Event()
        client = FakeLLMClient([tool_call_reply(("slow", "{}")), "完事。"])
        self.start(Agent(client=client, tools=ToolRegistry([SlowTool(started, release)])))

        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        conn.request(
            "POST",
            "/api/run/stream",
            json.dumps({"task": "任务甲"}).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)

        # 工具还卡在 execute 里（release 没放），start / tool_start 两帧应该
        # 已经推到客户端了——服务端要是攒着一次性发，这里会一直读不到
        lines = [response.readline().decode("utf-8") for _ in range(4)]
        self.assertEqual(lines[0], "event: start\n")
        self.assertEqual(lines[2], "\n")
        self.assertEqual(lines[3], "event: tool_start\n")
        self.assertTrue(started.wait(5), "工具没开始跑")

        release.set()
        rest = response.read().decode("utf-8")
        self.assertIn("event: tool_result", rest)
        self.assertIn("event: completed", rest)

    def test_health_answers_while_a_stream_is_running(self):
        started = threading.Event()
        release = threading.Event()

        class SlowAgent:
            def run(
                self,
                task,
                on_tool_start=None,
                on_tool_end=None,
                cancellation=None,
                on_llm_start=None,
                on_llm_end=None,
            ):
                started.set()
                release.wait(5)
                return "慢回答"

        self.start(SlowAgent())
        results = []
        worker = threading.Thread(
            target=lambda: results.append(self.stream("任务甲")), daemon=True
        )
        worker.start()
        self.assertTrue(started.wait(5))

        # 任务在跑，健康检查不碰那把锁，照样秒回
        health = self.call_json("GET", "/api/health")
        release.set()
        worker.join(5)

        self.assertEqual(health, (200, {"status": "ok"}))
        self.assertEqual(results[0][0], 200)
