"""llm_start / llm_end 的端到端那条流（Phase 13.5）。

起真服务、走真 HTTP / 真 SSE（沿用 test_api.py 的 ApiCase 与
test_api_stream.py 的拆帧），Agent 是**真 Agent + 真 LLMClient**——只把
SDK 换成脚本化的假货（tests/test_llm_retry.py 的 ScriptedAPI），想让它
怎么失败、哪一跳成功都写在用例里，不联网。

钉四件事：

- llm_* 帧的形状：带 session_id + run_id，只有数字和名字，没有 prompt、
  没有回复正文、没有 key（键逐个钉死，多一个键就是多泄一样东西）；
- 重试在流里长什么样：一跳一对 llm_start / llm_end，completed 只有一次；
- 老事件一个不少也不变形：tool_start / tool_result 还是 13.2 那个形状；
- 退避期间整轮超时 / 客户端按停照样能收场（timed_out / cancelled），
  退避配得再长也拖不住它们。

重试本身的单元件（哪族错重试、退避怎么涨、统计怎么算）在
tests/test_llm_retry.py；这里只验它在 HTTP 上被看见的样子。
"""

import json
import threading
import unittest

from agent import Agent
from config import LLMConfig
from llm import LLMClient
from tests.test_api import ApiCase
from tests.test_api_runs import CancelRequests
from tests.test_api_stream import StreamCase
from tests.test_sessions import FakeClock
from tests.test_llm_retry import (
    FAKE_KEY,
    MODEL,
    ScriptedAPI,
    counts,
    rate_limited,
    reply,
    tool_reply,
)
from tools import CalculatorTool, ToolRegistry

MARKER_KEY = "sk-钥匙标记-不该进流"


class SignallingAPI(ScriptedAPI):
    """脚本化的假 SDK，外加一面小旗：第一跳一开始就举起来。

    取消用例靠它知道「服务端已经在退避里了」——旗子一举，客户端再按停，
    按的就是退避那一觉。
    """

    def __init__(self, signal, *outcomes, **kwargs):
        super().__init__(*outcomes, **kwargs)
        self.signal = signal

    def create(self, **kwargs):
        self.signal.set()
        return super().create(**kwargs)


def make_client(
    *outcomes, max_retries=2, backoff=0.0, real_time=False, signal=None, api_key=FAKE_KEY
):
    """造一只用假 SDK 的 LLMClient。

    real_time=False（默认）时 clock / sleep 都是假的：退避不真等，测试跑得快。
    退避要陪真时间玩的用例（整轮超时、客户端按停）才用 real_time=True——
    那边的等待是服务端的预算，必须走真时钟。
    """
    api = SignallingAPI(signal, *outcomes) if signal is not None else ScriptedAPI(*outcomes)
    config = LLMConfig(
        api_key=api_key, model=MODEL, max_retries=max_retries, retry_backoff=backoff
    )
    if real_time:
        client = LLMClient(config=config)
    else:
        # 退避不真等：FakeClock 既是时钟也是 sleep（睡多久走多久）
        clock = FakeClock()
        api.clock = clock
        client = LLMClient(config=config, sleep=clock.advance, clock=clock)
    client._client = api
    return client, api


def llm_frames(events):
    """只挑 llm_* 那两帧：验「不泄东西」时只看它们。"""
    return [data for name, data in events if name.startswith("llm_")]


# ---------------------------------------------------------------------------
# 正常路径：一帧 start、一帧 end，夹着原有事件
# ---------------------------------------------------------------------------


class LlmStreamEventsTest(StreamCase):
    def setUp(self):
        self.client, self.api = make_client(reply("任务完成了。", usage=counts()))
        self.start(Agent(client=self.client))

    def test_the_stream_carries_llm_start_and_llm_end(self):
        events = self.stream_events("任务甲")

        self.assertEqual(
            [name for name, _ in events], ["start", "llm_start", "llm_end", "completed"]
        )
        session_id = events[0][1]["session_id"]
        _, llm_start = events[1]
        self.assertEqual(llm_start["model"], MODEL)
        self.assertEqual(llm_start["attempt"], 1)
        self.assertEqual(llm_start["max_attempts"], 3)
        self.assertEqual(llm_start["session_id"], session_id)
        self.assertTrue(llm_start["run_id"])

        _, llm_end = events[2]
        self.assertEqual(llm_end["model"], MODEL)
        self.assertTrue(llm_end["success"])
        self.assertEqual(llm_end["attempt_count"], 1)
        self.assertEqual(
            llm_end["usage"],
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual(llm_end["error"], "")
        self.assertGreaterEqual(llm_end["duration_seconds"], 0)
        # 同一轮：两帧带的是同一个 session_id / run_id
        self.assertEqual(llm_end["session_id"], session_id)
        self.assertEqual(llm_end["run_id"], llm_start["run_id"])

        self.assertEqual(events[-1][1]["answer"], "任务完成了。")

    def test_usage_is_null_when_the_api_does_not_send_it(self):
        client, _ = make_client(reply("好了"))  # 这个返回没带 usage
        self.start(Agent(client=client))

        events = self.stream_events("任务甲")

        self.assertIsNone(events[2][1]["usage"])
        self.assertIn("usage", events[2][1])  # 键在，值是 null

    def test_the_frames_carry_nothing_but_numbers_and_names(self):
        events = self.stream_events("任务甲")

        # 形状钉死：多一个键就是多泄一样东西
        self.assertEqual(
            set(events[1][1]),
            {"model", "attempt", "max_attempts", "session_id", "run_id"},
        )
        self.assertEqual(
            set(events[2][1]),
            {
                "model",
                "success",
                "duration_seconds",
                "attempt_count",
                "usage",
                "error",
                "session_id",
                "run_id",
            },
        )

    def test_the_prompt_the_system_prompt_and_the_key_stay_out_of_the_frames(self):
        client, _ = make_client(reply("模型给的一整段回答"), api_key=MARKER_KEY)
        self.start(Agent(client=client, system_prompt="系统提示里的秘密标记"))

        events = self.stream_events("探针任务甲")

        text = json.dumps(llm_frames(events), ensure_ascii=False)
        self.assertNotIn(MARKER_KEY, text)
        self.assertNotIn("系统提示里的秘密标记", text)
        self.assertNotIn("探针任务甲", text)  # prompt 里的用户输入也不推
        self.assertNotIn("模型给的一整段回答", text)  # 回复正文也不推
        # 但回答本身在 completed 里（它和 POST /api/run 的返回是同一份东西）
        self.assertEqual(events[-1][1]["answer"], "模型给的一整段回答")

    def test_a_tool_round_trip_keeps_the_old_events_intact(self):
        client, _ = make_client(
            tool_reply("calculator", '{"expression": "1+1"}'), reply("算好了。")
        )
        self.start(Agent(client=client, tools=ToolRegistry([CalculatorTool()])))

        events = self.stream_events("任务甲")

        self.assertEqual(
            [name for name, _ in events],
            [
                "start",
                "llm_start",
                "llm_end",
                "tool_start",
                "tool_result",
                "llm_start",
                "llm_end",
                "completed",
            ],
        )
        # tool_start / tool_result 还是 13.2 那个形状（没有多出 session_id 之类）
        self.assertEqual(events[3][1], {"tool": "calculator", "arguments": {"expression": "1+1"}})
        result = events[4][1]
        self.assertEqual(result["tool"], "calculator")
        self.assertTrue(result["success"])
        self.assertEqual(result["error"], "")
        self.assertNotIn("session_id", result)
        # 两次模型调用：一次要工具、一次收尾。attempt_count 是**这一次调用**
        # 试了几跳（不是这一轮的第几次调用），所以两边都是 1
        self.assertEqual(
            [data["attempt_count"] for name, data in events if name == "llm_end"],
            [1, 1],
        )


# ---------------------------------------------------------------------------
# 重试在流里的样子
# ---------------------------------------------------------------------------


class LlmRetryStreamTest(StreamCase):
    def test_a_retry_shows_up_as_two_hops(self):
        client, api = make_client(rate_limited("限流了"), reply("任务完成了。"))
        self.start(Agent(client=client))

        events = self.stream_events("任务甲")

        self.assertEqual(
            [name for name, _ in events],
            ["start", "llm_start", "llm_end", "llm_start", "llm_end", "completed"],
        )
        first_end = events[2][1]
        self.assertFalse(first_end["success"])
        self.assertIn("限流了", first_end["error"])
        self.assertEqual(first_end["attempt_count"], 1)

        second_end = events[4][1]
        self.assertTrue(second_end["success"])
        self.assertEqual(second_end["attempt_count"], 2)
        self.assertEqual(second_end["duration_seconds"], 0)  # 假时钟没走

        self.assertEqual(events[-1][1]["answer"], "任务完成了。")  # 只收尾一次
        self.assertEqual(api.calls, 2)

    def test_a_final_failure_still_ends_with_an_error_event(self):
        client, api = make_client(rate_limited("一直限流"), max_retries=2)
        self.start(Agent(client=client))

        events = self.stream_events("任务甲")

        names = [name for name, _ in events]
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-1], "error")
        self.assertNotIn("completed", names)
        self.assertEqual(names.count("llm_start"), 3)  # 1 次原始 + 2 次重试
        self.assertEqual(names.count("llm_end"), 3)
        self.assertEqual(api.calls, 3)

        last_end = events[-2][1]
        self.assertFalse(last_end["success"])
        self.assertEqual(last_end["attempt_count"], 3)
        self.assertIn("一直限流", last_end["error"])
        # Agent 抛出来的是这一次调用的最终说明：试了几次说清楚了
        message = events[-1][1]["message"]
        self.assertIn("一直限流", message)
        self.assertIn("重试 2 次后仍失败", message)

    def test_a_long_error_is_cut_short_in_the_frame(self):
        client, _ = make_client(rate_limited("限" * 500))
        self.start(Agent(client=client))

        events = self.stream_events("任务甲")

        self.assertTrue(events[2][1]["error"].endswith("…"))
        self.assertLessEqual(len(events[2][1]["error"]), 121)


# ---------------------------------------------------------------------------
# 退避期间：整轮超时 / 客户端按停
# ---------------------------------------------------------------------------


class LlmBackoffInterruptionTest(StreamCase):
    """退避是真时间的一小段：服务端的超时和客户端的取消都插得进去。"""

    def test_a_run_timeout_during_the_backoff_ends_with_timed_out(self):
        # 退避配得又长又真（5 秒），整轮预算只有 0.15 秒
        client, api = make_client(
            rate_limited("限流了"), max_retries=5, backoff=5.0, real_time=True
        )
        self.start(Agent(client=client), run_timeout=0.15)

        events = self.stream_events("任务甲")

        names = [name for name, _ in events]
        self.assertEqual(names[-1], "timed_out")
        self.assertNotIn("completed", names)
        self.assertNotIn("error", names)  # 超时是收场方式，不是运行出错
        self.assertEqual(names.count("llm_end"), 1)  # 第一跳的失败已经被看见了
        self.assertFalse(events[2][1]["success"])
        self.assertIn("限流了", events[2][1]["error"])
        self.assertIn("上限", events[-1][1]["message"])
        self.assertEqual(api.calls, 1)  # 退避里就停了，没有第二跳


class LlmBackoffCancelTest(CancelRequests, StreamCase):
    def test_a_cancel_during_the_backoff_ends_with_a_cancelled_event(self):
        started = threading.Event()
        client, api = make_client(
            rate_limited(), max_retries=5, backoff=5.0, real_time=True, signal=started
        )
        self.start(Agent(client=client))
        session_id = self.sessions.create().id

        results = []
        worker = threading.Thread(
            target=lambda: results.append(
                self.stream("任务甲", session_id=session_id)
            ),
            daemon=True,
        )
        worker.start()
        self.assertTrue(started.wait(5), "模型调用没开始")

        # 第一跳已经 429，服务端正睡在退避里——这时候按停
        status, payload = self.cancel(session_id)
        worker.join(5)

        self.assertEqual(status, 200)
        self.assertTrue(payload["cancelled"])
        events = self.parse(results[0][2])
        names = [name for name, _ in events]
        self.assertEqual(names[0], "start")
        self.assertEqual(names[-1], "cancelled")
        self.assertNotIn("completed", names)
        self.assertEqual(names.count("llm_end"), 1)  # 只跑了一跳
        self.assertEqual(events[-1][1]["run_id"], payload["run_id"])
        self.assertEqual(api.calls, 1)  # 退避没睡完就走了，一次都没多试


# ---------------------------------------------------------------------------
# 普通接口：重试过也一样，响应还是 13.1 那个形状
# ---------------------------------------------------------------------------


class PlainRunResponseTest(ApiCase):
    def test_a_retried_run_keeps_the_plain_response_shape(self):
        client, api = make_client(rate_limited(), reply("任务完成了。"))
        self.start(Agent(client=client))

        status, payload = self.post_task("任务甲")

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["answer"], "任务完成了。")
        self.assertTrue(payload["run_id"])
        # 键就这四个：不因为重试多出 Trace / 统计 / usage 之类的东西
        self.assertEqual(set(payload), {"success", "answer", "session_id", "run_id"})
        self.assertEqual(api.calls, 2)  # 确实重试过一次


if __name__ == "__main__":
    unittest.main()
