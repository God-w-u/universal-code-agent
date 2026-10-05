"""LLM 重试、统计与配置（Phase 13.5）。

不联网：脚本化的假 SDK 客户端替掉 LLMClient._client——想让它抛什么错、
哪一跳成功，都写在用例里；sleep / clock 注入假的（test_sessions.FakeClock，
sleep 就是 clock.advance），**退避一秒都不真等**，指数的节奏照样看得出来
（看每一跳开始时假时钟走到哪儿）。

钉的是规格第 9 条那一串：429 / 5xx / 连接超时都要重试，参数错 / 认证失败 /
模型不存在一次都不多试；重试有上限、退避是指数的；重试后成功、重试完还是
失败；attempt_count / usage 有和无 / duration；llm_start / llm_end 观察者；
退避期间能被取消、也能被整轮超时打断；统计里不该有的东西（prompt 正文、
回复正文、API key）一个字都没有。

端到端那条流（SSE 上的 llm_* 帧）在 tests/test_api_llm.py。
"""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
)

from agent import Agent
from agent.cancel import CANCELLED, TIMEOUT, CancellationContext, RunCancelled
from config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    ConfigError,
    LLMConfig,
    load_llm_config,
)
from llm import LLMClient, LLMError, LLMUsage, call_llm
from llm.client import CANCEL_POLL_SECONDS
from tests.test_sessions import FakeClock

MODEL = "假模型"
# 假钥匙：只活在测试里，专门拿来验「它不进统计、更不进 Trace」
FAKE_KEY = "sk-假钥匙-不该出现在任何地方"
MESSAGES = [{"role": "user", "content": "任务甲"}]


# ---------------------------------------------------------------------------
# 假东西：假 SDK、假观察者、造错的助手
# ---------------------------------------------------------------------------


def _request() -> httpx.Request:
    """httpx 的请求对象：SDK 那几族异常要它才构造得出来。"""
    return httpx.Request("POST", "http://example.invalid/v1/chat/completions")


def _status_error(error_type, status: int, message: str):
    return error_type(
        message, response=httpx.Response(status, request=_request()), body=None
    )


def rate_limited(message="接口限流了"):
    return _status_error(RateLimitError, 429, message)


def server_error(message="服务端炸了"):
    return _status_error(InternalServerError, 500, message)


def bad_request(message="参数不对"):
    return _status_error(BadRequestError, 400, message)


def unauthorized(message="钥匙不对"):
    return _status_error(AuthenticationError, 401, message)


def model_missing(message="模型不存在"):
    return _status_error(NotFoundError, 404, message)


def connection_error():
    return APIConnectionError(request=_request())


def timed_out():
    return APITimeoutError(_request())


def reply(text="好了", usage=None):
    """一次成功的返回；usage 不给 = 接口没带用量。"""
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def counts(prompt=10, completion=5, total=15):
    """接口带的 usage：字段名跟着接口走。"""
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


def tool_reply(name="calculator", arguments='{"expression": "1+1"}'):
    message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="call_1", function=SimpleNamespace(name=name, arguments=arguments)
            )
        ],
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class ScriptedAPI:
    """假的 SDK 客户端：按脚本抛错 / 返回，并记下每次调用。

    脚本用完最后一项就一直重复（和 test_agent.FakeLLMClient 一个脾气），
    方便写「一直 429」这种场景。latency 是每次调用（假装）花掉的时间——
    会拨动假时钟，好让 duration 是个能算的数。
    """

    def __init__(self, *outcomes, latency=0.0):
        self.outcomes = list(outcomes) or [reply()]
        self.latency = latency
        self.calls = 0
        self.started_at = []  # 每次调用时假时钟的读数
        self.clock = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if self.clock is not None:
            self.started_at.append(self.clock())
            self.clock.advance(self.latency)
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingObserver:
    """假观察者：和 agent/trace.py 的 Trace 一样，有那两个字就够。"""

    def __init__(self):
        self.starts = []  # (model, attempt, max_attempts)
        self.ends = []  # LLMCallStats

    def start_llm_call(self, model, attempt, max_attempts):
        self.starts.append((model, attempt, max_attempts))

    def record_llm_call(self, stats):
        self.ends.append(stats)


class LLMCase(unittest.TestCase):
    """公共部分：造一只带假 SDK、假时钟的 LLMClient，退避不真等。"""

    def make_client(self, *outcomes, max_retries=2, backoff=0.5, latency=0.0):
        self.clock = FakeClock()
        self.api = ScriptedAPI(*outcomes, latency=latency)
        self.api.clock = self.clock
        config = LLMConfig(
            api_key=FAKE_KEY,
            model=MODEL,
            max_retries=max_retries,
            retry_backoff=backoff,
        )
        # FakeClock 是 callable、advance 收一个秒数，正好能当 sleep 用：
        # 睡多久假时钟走多久，测试里一秒都不用真等
        client = LLMClient(config=config, sleep=self.clock.advance, clock=self.clock)
        client._client = self.api
        return client


# ---------------------------------------------------------------------------
# 该重试的错：429 / 5xx / 连不上 / 超时
# ---------------------------------------------------------------------------


class RetryableErrorTest(LLMCase):
    def test_a_429_is_retried(self):
        client = self.make_client(rate_limited(), reply("好了"))

        self.assertEqual(client.chat(MESSAGES), "好了")
        self.assertEqual(self.api.calls, 2)

    def test_a_5xx_is_retried(self):
        client = self.make_client(server_error(), reply("好了"))

        self.assertEqual(client.chat(MESSAGES), "好了")
        self.assertEqual(self.api.calls, 2)

    def test_a_connection_error_is_retried(self):
        client = self.make_client(connection_error(), reply("好了"))

        self.assertEqual(client.chat(MESSAGES), "好了")
        self.assertEqual(self.api.calls, 2)

    def test_a_timeout_is_retried(self):
        client = self.make_client(timed_out(), reply("好了"))

        self.assertEqual(client.chat(MESSAGES), "好了")
        self.assertEqual(self.api.calls, 2)

    def test_a_429_twice_then_success(self):
        # 重试次数够的话，连着限流两次也还能答上来
        client = self.make_client(rate_limited(), rate_limited(), reply("好了"))

        answer, stats = client.call_with_stats(MESSAGES)

        self.assertEqual(answer, "好了")
        self.assertEqual(stats.attempt_count, 3)
        self.assertTrue(stats.success)


# ---------------------------------------------------------------------------
# 不该重试的错：重试多少次结果都一样
# ---------------------------------------------------------------------------


class NonRetryableErrorTest(LLMCase):
    def test_a_bad_request_is_not_retried(self):
        client = self.make_client(bad_request("参数错了"))

        with self.assertRaises(LLMError) as caught:
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 1)
        self.assertIn("参数错了", str(caught.exception))

    def test_an_authentication_error_is_not_retried(self):
        client = self.make_client(unauthorized("钥匙不对"))

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 1)

    def test_a_missing_model_is_not_retried(self):
        client = self.make_client(model_missing("模型不存在"))

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 1)

    def test_an_unknown_error_is_not_retried(self):
        # 认不出来的错按保守处理：宁愿少试一次，也不多试一次
        client = self.make_client(OpenAIError("说不清怎么了"))

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 1)


# ---------------------------------------------------------------------------
# 重试的上限
# ---------------------------------------------------------------------------


class RetryLimitTest(LLMCase):
    def test_it_gives_up_after_the_configured_retries(self):
        client = self.make_client(rate_limited("一直限流"), max_retries=2)

        with self.assertRaises(LLMError) as caught:
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 3)  # 1 次原始尝试 + 2 次重试
        message = str(caught.exception)
        self.assertIn("一直限流", message)  # 最后一次的错误原文还在
        self.assertIn("重试 2 次后仍失败", message)  # 也说清了试过几次

    def test_zero_retries_means_a_single_attempt(self):
        client = self.make_client(rate_limited(), max_retries=0)

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 1)

    def test_the_limit_is_configurable(self):
        client = self.make_client(rate_limited(), max_retries=4)

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertEqual(self.api.calls, 5)

    def test_a_single_attempt_does_not_talk_about_retries(self):
        client = self.make_client(rate_limited("限流了"), max_retries=0)

        with self.assertRaises(LLMError) as caught:
            client.chat(MESSAGES)

        self.assertNotIn("重试", str(caught.exception))

    def test_the_failure_keeps_the_last_error(self):
        # 最后一跳的错和第一次不一样时，抛的是最后那一个（不是第一次的）
        client = self.make_client(
            rate_limited("第一次限流"), server_error("最后一次服务端炸了")
        )

        with self.assertRaises(LLMError) as caught:
            client.chat(MESSAGES)

        message = str(caught.exception)
        self.assertIn("最后一次服务端炸了", message)
        self.assertNotIn("第一次限流", message)


# ---------------------------------------------------------------------------
# 退避
# ---------------------------------------------------------------------------


class BackoffTest(LLMCase):
    def test_the_wait_grows_exponentially(self):
        client = self.make_client(rate_limited(), max_retries=3, backoff=0.5)

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        # 每一跳开始时假时钟的读数：0 → 等 0.5 → 1.5 → 等 1.0 → 3.5，
        # 也就是每次重试等的是上一次的两倍（浮点累加，逐个近似比）
        self.assertEqual(len(self.api.started_at), 4)
        for actual, expected in zip(self.api.started_at, [0.0, 0.5, 1.5, 3.5]):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_a_zero_backoff_does_not_wait(self):
        client = self.make_client(rate_limited(), max_retries=2, backoff=0.0)

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertEqual(self.api.started_at, [0.0, 0.0, 0.0])

    def test_the_wait_is_sliced_into_short_pieces(self):
        # 「取消能不能及时生效」全看这一条：一次退避切成小片睡，没有哪一片
        # 长到让取消干等
        slept = []
        clock = FakeClock()
        api = ScriptedAPI(rate_limited())
        api.clock = clock
        config = LLMConfig(
            api_key=FAKE_KEY, model=MODEL, max_retries=1, retry_backoff=0.5
        )
        client = LLMClient(
            config=config,
            sleep=lambda seconds: (slept.append(seconds), clock.advance(seconds)),
            clock=clock,
        )
        client._client = api

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        self.assertTrue(slept)  # 确实等了
        self.assertAlmostEqual(sum(slept), 0.5, places=9)  # 一共 0.5 秒
        self.assertLessEqual(max(slept), CANCEL_POLL_SECONDS)  # 但没有一大觉


# ---------------------------------------------------------------------------
# 统计对象
# ---------------------------------------------------------------------------


class StatsTest(LLMCase):
    def test_a_successful_call_reports_success_and_duration(self):
        client = self.make_client(reply("好了"), latency=0.25)

        answer, stats = client.call_with_stats(MESSAGES)

        self.assertEqual(answer, "好了")
        self.assertTrue(stats.success)
        self.assertEqual(stats.model, MODEL)
        self.assertEqual(stats.attempt_count, 1)
        self.assertEqual(stats.max_attempts, 3)
        self.assertAlmostEqual(stats.duration_seconds, 0.25, places=9)
        self.assertAlmostEqual(stats.total_duration_seconds, 0.25, places=9)
        self.assertEqual(stats.error, "")

    def test_the_attempt_count_covers_every_hop(self):
        client = self.make_client(
            rate_limited(),
            rate_limited(),
            reply("好了"),
            latency=0.2,
            backoff=0.5,
            max_retries=2,
        )

        answer, stats = client.call_with_stats(MESSAGES)

        self.assertEqual(answer, "好了")
        self.assertEqual(stats.attempt_count, 3)
        self.assertEqual(stats.max_attempts, 3)
        # duration 只是最后一跳自己的耗时；total 把两次退避也算进去
        self.assertAlmostEqual(stats.duration_seconds, 0.2, places=9)
        self.assertAlmostEqual(
            stats.total_duration_seconds, 0.2 * 3 + 0.5 + 1.0, places=9
        )

    def test_usage_is_reported_when_the_api_sends_it(self):
        client = self.make_client(reply("好了", usage=counts(10, 5, 15)))

        _, stats = client.call_with_stats(MESSAGES)

        self.assertEqual(
            stats.usage, LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
        self.assertEqual(
            stats.to_dict()["usage"],
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    def test_usage_is_none_when_the_api_omits_it(self):
        client = self.make_client(reply("好了"))  # 这个返回没带 usage

        _, stats = client.call_with_stats(MESSAGES)

        self.assertIsNone(stats.usage)
        self.assertIsNone(stats.to_dict()["usage"])

    def test_usage_fields_the_api_did_not_send_stay_none(self):
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=None, total_tokens=None
        )
        client = self.make_client(reply("好了", usage=usage))

        _, stats = client.call_with_stats(MESSAGES)

        self.assertEqual(stats.usage.prompt_tokens, 10)
        self.assertIsNone(stats.usage.completion_tokens)  # 接口没给就不替它算

    def test_a_failed_call_reports_the_failure(self):
        observer = RecordingObserver()
        client = self.make_client(rate_limited("一直限流"), max_retries=2)

        with self.assertRaises(LLMError):
            client.call_with_stats(MESSAGES, observer=observer)

        self.assertEqual(len(observer.ends), 3)  # 每一跳都有一份统计
        last = observer.ends[-1]
        self.assertFalse(last.success)
        self.assertEqual(last.attempt_count, 3)
        self.assertIn("一直限流", last.error)
        self.assertIsNone(last.usage)

    def test_an_empty_choices_reply_is_a_failure(self):
        observer = RecordingObserver()
        client = self.make_client(SimpleNamespace(choices=[], usage=None))

        with self.assertRaises(LLMError):
            client.call_with_stats(MESSAGES, observer=observer)

        self.assertEqual(self.api.calls, 1)  # 空结果重试也没用：一次就抛
        self.assertFalse(observer.ends[-1].success)
        self.assertIn("空结果", observer.ends[-1].error)

    def test_the_stats_never_carry_the_prompt_the_answer_or_the_key(self):
        client = self.make_client(reply("模型给的一整段回答", usage=counts()))

        _, stats = client.call_with_stats(
            [{"role": "user", "content": "任务甲"}], tools=None, observer=None
        )

        # 键是固定的八个：没有藏 prompt / 正文 / key 的地方
        self.assertEqual(
            set(stats.to_dict()),
            {
                "model",
                "success",
                "duration_seconds",
                "total_duration_seconds",
                "attempt_count",
                "max_attempts",
                "usage",
                "error",
            },
        )
        text = json.dumps(stats.to_dict(), ensure_ascii=False)
        self.assertNotIn("任务甲", text)
        self.assertNotIn("模型给的一整段回答", text)
        self.assertNotIn(FAKE_KEY, text)

    def test_the_failure_stats_do_not_carry_the_key_either(self):
        observer = RecordingObserver()
        client = self.make_client(rate_limited(), max_retries=0)

        with self.assertRaises(LLMError):
            client.call_with_stats(MESSAGES, observer=observer)

        self.assertNotIn(FAKE_KEY, json.dumps(observer.ends[-1].to_dict()))

    def test_chat_with_tools_still_returns_the_assistant_message(self):
        client = self.make_client(tool_reply())

        message, stats = client.call_with_stats(
            MESSAGES, tools=[{"type": "function", "function": {"name": "calculator"}}]
        )

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "calculator")
        self.assertTrue(stats.success)


# ---------------------------------------------------------------------------
# 观察者（客户端这一层：一跳一对）
# ---------------------------------------------------------------------------


class LLMObserverTest(LLMCase):
    def test_every_attempt_is_announced(self):
        observer = RecordingObserver()
        client = self.make_client(rate_limited(), reply("好了"))

        client.call_with_stats(MESSAGES, observer=observer)

        self.assertEqual(observer.starts, [(MODEL, 1, 3), (MODEL, 2, 3)])
        self.assertEqual([stats.attempt_count for stats in observer.ends], [1, 2])
        self.assertEqual([stats.success for stats in observer.ends], [False, True])

    def test_a_single_attempt_announces_once(self):
        observer = RecordingObserver()
        client = self.make_client(reply("好了"))

        client.call_with_stats(MESSAGES, observer=observer)

        self.assertEqual(observer.starts, [(MODEL, 1, 3)])
        self.assertEqual(len(observer.ends), 1)

    def test_no_observer_is_fine(self):
        client = self.make_client(reply("好了"))

        answer, _ = client.call_with_stats(MESSAGES)

        self.assertEqual(answer, "好了")

    def test_an_exploding_observer_does_not_break_the_call(self):
        class AngryObserver:
            def start_llm_call(self, *args):
                raise RuntimeError("观察者炸了")

            def record_llm_call(self, *args):
                raise RuntimeError("观察者炸了")

        client = self.make_client(reply("好了"))

        answer, stats = client.call_with_stats(MESSAGES, observer=AngryObserver())

        self.assertEqual(answer, "好了")
        self.assertTrue(stats.success)


class CallLLMCompatTest(unittest.TestCase):
    """call_llm：有 call_with_stats 就走它，没有就退回老接口。"""

    class OldClient:
        """老式客户端：只有 chat / chat_with_tools（测试替身都长这样）。"""

        def __init__(self):
            self.calls = []

        def chat(self, messages):
            self.calls.append("chat")
            return "老回答"

        def chat_with_tools(self, messages, tools):
            self.calls.append("tools")
            return {"role": "assistant", "content": "老回答", "tool_calls": []}

    def test_an_old_client_is_called_the_old_way(self):
        client = self.OldClient()

        self.assertEqual(call_llm(client, MESSAGES), "老回答")
        self.assertEqual(
            call_llm(client, MESSAGES, tools=[{"type": "function"}]),
            {"role": "assistant", "content": "老回答", "tool_calls": []},
        )
        self.assertEqual(client.calls, ["chat", "tools"])

    def test_the_observer_and_cancellation_are_not_forced_on_old_clients(self):
        # 老客户端不认识 observer / cancellation，也不该被打扰
        client = self.OldClient()

        self.assertEqual(
            call_llm(client, MESSAGES, observer=object(), cancellation=object()),
            "老回答",
        )


# ---------------------------------------------------------------------------
# 退避期间的取消与整轮超时
# ---------------------------------------------------------------------------


class RetryCancellationTest(LLMCase):
    def test_an_already_cancelled_call_never_reaches_the_api(self):
        cancellation = CancellationContext()
        cancellation.cancel()
        client = self.make_client(reply("好了"))

        with self.assertRaises(RunCancelled):
            client.call_with_stats(MESSAGES, cancellation=cancellation)

        self.assertEqual(self.api.calls, 0)

    def test_a_cancel_during_the_backoff_stops_the_retries(self):
        cancellation = CancellationContext()
        clock = FakeClock()
        api = ScriptedAPI(rate_limited())
        api.clock = clock
        config = LLMConfig(
            api_key=FAKE_KEY, model=MODEL, max_retries=5, retry_backoff=10.0
        )

        def sleep(seconds):
            # 该等了……就在这时候，取消方在另一条线程上按了停
            clock.advance(seconds)
            cancellation.cancel()

        client = LLMClient(config=config, sleep=sleep, clock=clock)
        client._client = api

        with self.assertRaises(RunCancelled) as caught:
            client.call_with_stats(MESSAGES, cancellation=cancellation)

        self.assertEqual(caught.exception.reason, CANCELLED)
        self.assertEqual(api.calls, 1)  # 一次都没多试
        self.assertAlmostEqual(clock.now, CANCEL_POLL_SECONDS, places=9)

    def test_a_run_timeout_during_the_backoff_stops_the_retries(self):
        clock = FakeClock()
        cancellation = CancellationContext(timeout=0.3, clock=clock)
        api = ScriptedAPI(rate_limited())
        api.clock = clock
        config = LLMConfig(
            api_key=FAKE_KEY, model=MODEL, max_retries=5, retry_backoff=10.0
        )
        client = LLMClient(config=config, sleep=clock.advance, clock=clock)
        client._client = api

        with self.assertRaises(RunCancelled) as caught:
            client.call_with_stats(MESSAGES, cancellation=cancellation)

        self.assertEqual(caught.exception.reason, TIMEOUT)
        self.assertEqual(api.calls, 1)
        # 整轮预算到点就走：退避配得再长也不会把这一轮拖长
        self.assertAlmostEqual(clock.now, 0.3, places=9)

    def test_the_cancellation_is_not_swallowed_into_an_llm_error(self):
        # RunCancelled 是控制流，不是「模型调用失败」：原样穿出来，
        # 不会被重试逻辑接住、也不会变成 LLMError
        cancellation = CancellationContext()
        cancellation.cancel()
        client = self.make_client(rate_limited())

        with self.assertRaises(RunCancelled) as caught:
            client.call_with_stats(MESSAGES, cancellation=cancellation)

        self.assertNotIsInstance(caught.exception, LLMError)

    def test_no_cancellation_is_fine(self):
        client = self.make_client(rate_limited(), reply("好了"))

        answer, _ = client.call_with_stats(MESSAGES, cancellation=None)

        self.assertEqual(answer, "好了")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


class LLMRetryConfigTest(unittest.TestCase):
    BASE = {"OPENAI_API_KEY": "测试用的假钥匙", "OPENAI_MODEL": MODEL}

    def load(self, **extra):
        with patch.dict(os.environ, {**self.BASE, **extra}, clear=True):
            return load_llm_config()

    def test_the_defaults_are_conservative(self):
        config = self.load()

        self.assertEqual(config.max_retries, DEFAULT_MAX_RETRIES)
        self.assertEqual(config.retry_backoff, DEFAULT_RETRY_BACKOFF)

    def test_the_values_are_read_from_the_environment(self):
        config = self.load(LLM_MAX_RETRIES="5", LLM_RETRY_BACKOFF="0.25")

        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.retry_backoff, 0.25)

    def test_zero_is_allowed_for_both(self):
        config = self.load(LLM_MAX_RETRIES="0", LLM_RETRY_BACKOFF="0")

        self.assertEqual(config.max_retries, 0)
        self.assertEqual(config.retry_backoff, 0.0)

    def test_a_negative_retry_count_is_rejected(self):
        with self.assertRaises(ConfigError) as caught:
            self.load(LLM_MAX_RETRIES="-1")

        self.assertIn("LLM_MAX_RETRIES", str(caught.exception))

    def test_a_fractional_retry_count_is_rejected(self):
        with self.assertRaises(ConfigError):
            self.load(LLM_MAX_RETRIES="2.5")

    def test_a_negative_backoff_is_rejected(self):
        with self.assertRaises(ConfigError) as caught:
            self.load(LLM_RETRY_BACKOFF="-0.5")

        self.assertIn("LLM_RETRY_BACKOFF", str(caught.exception))

    def test_garbage_is_rejected(self):
        for key in ("LLM_MAX_RETRIES", "LLM_RETRY_BACKOFF"):
            with self.subTest(key=key):
                with self.assertRaises(ConfigError):
                    self.load(**{key: "一会儿"})


# ---------------------------------------------------------------------------
# Agent 那一层：观察者照传，重试不算新的 step
# ---------------------------------------------------------------------------


class AgentLlmObservationTest(LLMCase):
    def make_agent(self, *outcomes, **kwargs):
        return Agent(client=self.make_client(*outcomes, **kwargs))

    def test_the_agent_hands_llm_notifications_to_its_observers(self):
        agent = self.make_agent(reply("任务完成了。", usage=counts()))
        starts, ends = [], []

        answer = agent.run(
            "任务甲",
            on_llm_start=lambda *args: starts.append(args),
            on_llm_end=ends.append,
        )

        self.assertEqual(answer, "任务完成了。")
        self.assertEqual(starts, [(MODEL, 1, 3)])
        self.assertEqual(len(ends), 1)
        self.assertTrue(ends[0].success)
        self.assertEqual(ends[0].usage.total_tokens, 15)

    def test_the_llm_events_are_not_tool_events(self):
        agent = self.make_agent(reply("任务完成了。"))

        agent.run("任务甲")

        # 工具事件列表里只有真跑过的工具；模型调用另记一份
        self.assertEqual(agent.get_trace(), [])
        self.assertEqual(len(agent.trace.llm_calls), 1)
        self.assertEqual(agent.trace.llm_calls[0].model, MODEL)

    def test_a_retry_is_not_a_new_agent_step(self):
        agent = self.make_agent(rate_limited(), reply("任务完成了。"), backoff=0.0)
        ends = []

        answer = agent.run("任务甲", on_llm_end=ends.append)

        self.assertEqual(answer, "任务完成了。")
        self.assertEqual(len(ends), 2)  # 两跳（第一跳失败、第二跳成功）
        self.assertEqual(ends[-1].attempt_count, 2)  # 但都算同一次调用
        # 对话历史里还是规规矩矩的三条：重试不留痕
        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "user", "assistant"],
        )
        self.assertEqual(len(agent.get_trace()), 0)

    def test_a_final_failure_keeps_the_clear_error(self):
        agent = self.make_agent(rate_limited("一直限流"), max_retries=1, backoff=0.0)
        ends = []

        with self.assertRaises(LLMError) as caught:
            agent.run("任务甲", on_llm_end=ends.append)

        self.assertIn("一直限流", str(caught.exception))
        self.assertIn("重试 1 次后仍失败", str(caught.exception))
        self.assertFalse(ends[-1].success)

    def test_the_llm_observer_does_not_leak_into_the_next_run(self):
        agent = self.make_agent(reply("第一次。"), reply("第二次。"))
        ends = []

        agent.run("任务甲", on_llm_end=ends.append)
        self.assertEqual(len(ends), 1)

        agent.run("任务乙")  # 这一轮不带观察者

        self.assertEqual(len(ends), 1)  # 上一轮的回调不会再响


# ---------------------------------------------------------------------------
# SDK 自己的重试必须关掉（13 阶段回归）
# ---------------------------------------------------------------------------
#
# 上面所有用例都把 LLMClient._client 换成了假 SDK，所以「SDK 在它内部悄悄
# 重试」这件事它们一条都测不到——真 SDK 的默认重试发生在一次 create() 里
# 面，本模块的 attempt 计数、Trace 事件、取消检查点全看不见。这一节不打
# 假货：真的 openai SDK + 本地假 HTTP 服务，数**实际发出去几个请求**。


class _Counting429Handler(BaseHTTPRequestHandler):
    """只回 429 的假服务端：来一个请求记一笔。"""

    protocol_version = "HTTP/1.1"
    requests = 0

    def do_POST(self) -> None:  # noqa: N802 —— 基类的命名就是这样
        type(self).requests += 1
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = json.dumps(
            {"error": {"message": "限流了", "type": "rate_limit_error"}}
        ).encode("utf-8")
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass  # 测试里不往 stderr 打访问日志


class SDKRetryCountTest(unittest.TestCase):
    """`LLM_MAX_RETRIES=n` = 真的只发 n+1 个请求。"""

    def setUp(self):
        _Counting429Handler.requests = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Counting429Handler)
        server.daemon_threads = True
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        threading.Thread(
            target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
        ).start()
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"

    def count_requests(self, max_retries: int) -> int:
        # key 用纯 ASCII：httpx 编码请求头是按 latin-1 来的，上面那个
        # 带汉字的假钥匙到这一步会先炸 UnicodeEncodeError，测不到重试
        config = LLMConfig(
            api_key="sk-test",
            model=MODEL,
            base_url=self.base_url,
            max_retries=max_retries,
            retry_backoff=0.0,
        )
        client = LLMClient(config=config)

        with self.assertRaises(LLMError):
            client.chat(MESSAGES)

        return _Counting429Handler.requests

    def test_the_sdk_does_not_retry_behind_our_back(self):
        # 不关掉 SDK 的默认重试（DEFAULT_MAX_RETRIES = 2），这里会是 3
        self.assertEqual(self.count_requests(0), 1)

    def test_a_configured_retry_is_exactly_one_more_request(self):
        # 关不掉的话这里是 9（3 次尝试 × SDK 自己 3 次）
        self.assertEqual(self.count_requests(2), 3)

    def test_the_sdk_client_reports_no_retries_of_its_own(self):
        client = LLMClient(
            config=LLMConfig(api_key="sk-test", model=MODEL, base_url=self.base_url)
        )

        self.assertEqual(client._client.max_retries, 0)
