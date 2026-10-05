"""API 安全边界测试：Content-Type、CORS（含 OPTIONS 预检）、API Token、慢客户端。

纯 HTTP 层，起真服务走真 socket（沿用 test_api.py 的 ApiCase）。规格
（13.3）要求覆盖的每一项都在这里：缺少 / 错误 Content-Type；允许的
Origin 拿到 CORS 头、非允许的 403、不带 Origin 不受影响、默认（没配
origins）一律不放行、OPTIONS 预检的几种结局；Token 未配置 = 直接可用，
配置后缺失 / 错误 / 正确都验，health 保持开放，401 先于 415。13.10 的
慢客户端防护（请求发一半就不发了）在 SlowClientTest。

load_api_config 的单元测试（默认值、覆盖、非法值、`*` 被拒）在最下面。
"""

import json
import os
import socket
import unittest
from unittest.mock import patch

from api.server import ApiHandler
from config import (
    DEFAULT_MAX_SESSIONS,
    DEFAULT_RUN_TIMEOUT,
    DEFAULT_SESSION_TTL,
    ConfigError,
    load_api_config,
)
from tests.test_api import ApiCase, CountingAgent
from tests.test_api_stream import StubAgent

JSON_HEADER = {"Content-Type": "application/json"}
ORIGIN = "http://localhost:5173"


def task_body() -> bytes:
    return json.dumps({"task": "任务甲"}, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Content-Type：带体的方法必须声明 application/json
# ---------------------------------------------------------------------------


class ContentTypeTest(ApiCase):
    def setUp(self):
        self.agent = CountingAgent()
        self.start(self.agent)

    def test_a_post_without_content_type_is_415(self):
        status, payload = self.call_json("POST", "/api/run", task_body())

        self.assertEqual(status, 415)
        self.assertFalse(payload["success"])
        self.assertIn("application/json", payload["answer"])

    def test_a_wrong_content_type_is_415(self):
        status, payload = self.call_json(
            "POST", "/api/run", task_body(), {"Content-Type": "text/plain"}
        )

        self.assertEqual(status, 415)
        self.assertFalse(payload["success"])

    def test_a_charset_parameter_is_fine(self):
        status, payload = self.call_json(
            "POST",
            "/api/run",
            task_body(),
            {"Content-Type": "application/json; charset=utf-8"},
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])

    def test_the_stream_endpoint_checks_too(self):
        status, _ = self.call_json("POST", "/api/run/stream", task_body())

        self.assertEqual(status, 415)

    def test_delete_checks_too(self):
        body = json.dumps({"session_id": "甲"}).encode("utf-8")
        status, _ = self.call_json("DELETE", "/api/sessions", body)

        self.assertEqual(status, 415)

    def test_a_rejected_body_never_reaches_the_agent(self):
        self.call_json("POST", "/api/run", task_body())
        self.call_json(
            "POST", "/api/run", task_body(), {"Content-Type": "text/plain"}
        )

        self.assertEqual(self.agent.tasks, [])


# ---------------------------------------------------------------------------
# CORS：Origin 检查 + OPTIONS 预检
# ---------------------------------------------------------------------------


class CorsTest(ApiCase):
    def setUp(self):
        self.agent = CountingAgent()
        self.start(self.agent, cors_origins=(ORIGIN,))

    def post(self, origin=None, path="/api/run"):
        headers = dict(JSON_HEADER)
        if origin is not None:
            headers["Origin"] = origin
        return self.call("POST", path, task_body(), headers)

    def test_an_allowed_origin_gets_the_headers(self):
        status, headers, _ = self.post(ORIGIN)

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), ORIGIN)
        self.assertEqual(headers.get("Vary"), "Origin")

    def test_a_disallowed_origin_is_403(self):
        status, headers, raw = self.post("http://evil.example")

        self.assertEqual(status, 403)
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))
        self.assertIn("API_CORS_ORIGINS", json.loads(raw.decode("utf-8"))["answer"])

    def test_a_request_without_an_origin_is_not_blocked(self):
        # 命令行客户端不带 Origin：配不配 CORS 都照常可用
        status, _, _ = self.post()

        self.assertEqual(status, 200)

    def test_a_disallowed_origin_is_blocked_even_on_health(self):
        status, _, _ = self.call(
            "GET", "/api/health", headers={"Origin": "http://evil.example"}
        )

        self.assertEqual(status, 403)

    def test_a_blocked_request_never_reaches_the_agent(self):
        self.post("http://evil.example")

        self.assertEqual(self.agent.tasks, [])

    def test_a_preflight_passes_with_methods_and_headers(self):
        status, headers, _ = self.call(
            "OPTIONS",
            "/api/run",
            headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"},
        )

        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), ORIGIN)
        self.assertIn("POST", headers.get("Access-Control-Allow-Methods", ""))
        self.assertIn(
            "Authorization", headers.get("Access-Control-Allow-Headers", "")
        )
        self.assertIn("Content-Type", headers.get("Access-Control-Allow-Headers", ""))
        self.assertEqual(headers.get("Access-Control-Max-Age"), "600")

    def test_a_preflight_from_a_disallowed_origin_is_403(self):
        status, _, _ = self.call(
            "OPTIONS", "/api/run", headers={"Origin": "http://evil.example"}
        )

        self.assertEqual(status, 403)

    def test_options_without_an_origin_is_400(self):
        status, payload = self.call_json("OPTIONS", "/api/run")

        self.assertEqual(status, 400)
        self.assertFalse(payload["success"])

    def test_a_preflight_for_an_unknown_path_is_404(self):
        status, payload = self.call_json(
            "OPTIONS", "/api/nowhere", headers={"Origin": ORIGIN}
        )

        self.assertEqual(status, 404)
        self.assertFalse(payload["success"])

    def test_health_preflight_only_allows_get(self):
        status, headers, _ = self.call(
            "OPTIONS", "/api/health", headers={"Origin": ORIGIN}
        )

        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Access-Control-Allow-Methods"), "GET")


class CorsDefaultTest(ApiCase):
    """没配 API_CORS_ORIGINS：一个 Origin 都不放行（默认安全）。"""

    def test_any_origin_is_rejected_by_default(self):
        self.start(CountingAgent())

        status, _, _ = self.call(
            "GET", "/api/health", headers={"Origin": ORIGIN}
        )

        self.assertEqual(status, 403)

    def test_without_an_origin_everything_works(self):
        self.start(CountingAgent())

        self.assertEqual(self.post_task("任务甲")[0], 200)


class StreamCorsTest(ApiCase):
    def test_the_stream_response_carries_cors_headers(self):
        self.start(StubAgent(), cors_origins=(ORIGIN,))

        status, headers, raw = self.call(
            "POST",
            "/api/run/stream",
            task_body(),
            {**JSON_HEADER, "Origin": ORIGIN},
        )

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), ORIGIN)
        self.assertIn(b"event: completed", raw)


# ---------------------------------------------------------------------------
# API Token
# ---------------------------------------------------------------------------


class TokenTest(ApiCase):
    def setUp(self):
        self.agent = CountingAgent()
        self.start(self.agent, token="s3cret-token")

    def post(self, authorization=None):
        headers = dict(JSON_HEADER)
        if authorization is not None:
            headers["Authorization"] = authorization
        return self.call("POST", "/api/run", task_body(), headers)

    def test_a_missing_token_is_401(self):
        status, headers, raw = self.post()

        self.assertEqual(status, 401)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        self.assertIn("Token", json.loads(raw.decode("utf-8"))["answer"])
        self.assertEqual(self.agent.tasks, [])  # 没验过，任务压根没跑

    def test_a_wrong_token_is_401(self):
        status, _, _ = self.post("Bearer wrong-token")

        self.assertEqual(status, 401)

    def test_a_prefix_of_the_token_is_401(self):
        status, _, _ = self.post("Bearer s3cret-toke")

        self.assertEqual(status, 401)

    def test_the_right_token_gets_through(self):
        status, headers, raw = self.post("Bearer s3cret-token")

        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw.decode("utf-8"))["success"])
        # 通过了就不该再带 401 的挑战头
        self.assertIsNone(headers.get("WWW-Authenticate"))

    def test_the_scheme_is_case_insensitive(self):
        status, _, _ = self.post("bearer s3cret-token")

        self.assertEqual(status, 200)

    def test_another_scheme_is_401(self):
        status, _, _ = self.post("Basic s3cret-token")

        self.assertEqual(status, 401)

    def test_the_stream_endpoint_is_protected(self):
        status, _, _ = self.call("POST", "/api/run/stream", task_body(), JSON_HEADER)

        self.assertEqual(status, 401)

    def test_delete_is_protected(self):
        body = json.dumps({"session_id": "甲"}).encode("utf-8")
        status, _, _ = self.call("DELETE", "/api/sessions", body, JSON_HEADER)

        self.assertEqual(status, 401)

    def test_a_missing_token_is_answered_before_the_content_type(self):
        # 什么都没带：先答 401（更重要的边界先说），而不是 415
        status, _, _ = self.call("POST", "/api/run", task_body())

        self.assertEqual(status, 401)


class HealthStaysOpenTest(ApiCase):
    def test_health_needs_no_token(self):
        # 配了 Token 也一样：前端要能探活
        self.start(CountingAgent(), token="s3cret-token")

        status, payload = self.call_json("GET", "/api/health")

        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})


class NoTokenTest(ApiCase):
    def test_everything_works_without_a_token(self):
        # 没配 API_TOKEN = 本机开发：行为和 13.2 一字不差
        self.start(CountingAgent())

        self.assertEqual(self.post_task("任务甲")[0], 200)
        self.assertEqual(self.call_json("GET", "/api/health")[0], 200)


# ---------------------------------------------------------------------------
# 慢客户端：请求发一半就不发了，不许把服务端线程一直占着
# ---------------------------------------------------------------------------


class SlowClientTest(ApiCase):
    """连上来却不把请求发完的客户端，服务端要自己把连接收掉（13.10）。

    防护是 ApiHandler.timeout：socketserver 会给每条连接的 socket 设上它，
    读 / 写超时都变成 socket.timeout，而基类的 handle_one_request 本来就
    接住它并关连接。要紧的是它作用在**读请求行 / 请求头**这一步——那发生
    在 Token 校验和路由之前，配了口令也拦不住只发半截请求的客户端，只有
    超时能清掉它（ThreadingHTTPServer 每连接一个线程，不收就是线程一直
    被占着）。

    真等 60 秒不现实：用例把它**等比缩短**（真值 ÷ 120 = 半秒），测的
    仍然是生产上那一个值，不是另填一个数——真值要是没了，setUp 直接失败
    （见那里的断言），不会静默放行。
    流式那头「客户端中途断开」的回归在 test_api_runs.py 的 GhostRunTest。
    """

    def setUp(self):
        real = ApiHandler.timeout
        # None = stdlib 默认的「永不超时」，正是 13.10 要防的。这个类里
        # 每条用例都建立在「有这么一个超时」之上，没了就得红，不能跳过
        self.assertIsNotNone(real, "ApiHandler.timeout 没设：慢客户端能把线程占死")
        self.assertGreater(real, 0)
        self.assertLessEqual(real, 300, "超时大得没边等于没设")

        # 缩短到跑得起：真值 60 秒 → 半秒。留着等比关系，将来 60 改成别的
        # 数，这里跟着走
        self.short_timeout = real / 120
        self.wait = max(5.0, self.short_timeout * 4)  # 等服务端收连接的宽限

        self.agent = CountingAgent()
        # 带 Token 起服务：半截请求挡在读请求头那一步，连 401 都轮不到发
        self.start(self.agent, token="s3cret-token")
        for quiet in (
            patch.object(ApiHandler, "timeout", self.short_timeout),
            patch.object(ApiHandler, "log_error"),  # 超时那行日志别往 stderr 刷
        ):
            quiet.start()
            self.addCleanup(quiet.stop)

    def raw(self) -> socket.socket:
        conn = socket.create_connection(("127.0.0.1", self.port), timeout=self.wait)
        self.addCleanup(conn.close)
        return conn

    def assert_dropped(self, conn: socket.socket) -> None:
        """等服务端收掉这条连接：读到 EOF 或连接被重置都算收掉了。"""
        conn.settimeout(self.wait)
        try:
            data = conn.recv(1)
        except (ConnectionResetError, ConnectionAbortedError):
            return
        except TimeoutError:  # pragma: no cover - 只有防护没生效才会走到
            self.fail("服务端一直没关这条半截连接：慢客户端把线程占住了")
        self.assertEqual(data, b"", "连接还开着，服务端没有主动关掉它")

    def test_a_request_line_that_never_arrives_is_dropped(self):
        conn = self.raw()

        # 一个字节都不发：最纯粹的那种「连上来占着不放」
        self.assert_dropped(conn)
        self.assertEqual(self.agent.tasks, [])

    def test_headers_that_never_finish_are_dropped(self):
        conn = self.raw()
        conn.sendall(b"POST /api/run HTTP/1.1\r\nHost: 127.0.0.1\r\n")  # 没有收尾空行

        self.assert_dropped(conn)
        self.assertEqual(self.agent.tasks, [])

    def test_normal_requests_are_unaffected(self):
        # 超时压到半秒也照样收发正常：它管的是「等客户端的下一口」，
        # 不是「整个请求必须半秒内跑完」——一轮任务本来就可以跑几分钟
        status, payload = self.call_json(
            "POST",
            "/api/run",
            task_body(),
            {**JSON_HEADER, "Authorization": "Bearer s3cret-token"},
        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(self.agent.tasks, ["任务甲"])


# ---------------------------------------------------------------------------
# load_api_config：默认值、覆盖、非法值
# ---------------------------------------------------------------------------


class ApiConfigTest(unittest.TestCase):
    def test_the_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            api_config = load_api_config()

        self.assertEqual(api_config.session_ttl, DEFAULT_SESSION_TTL)
        self.assertEqual(api_config.max_sessions, DEFAULT_MAX_SESSIONS)
        self.assertEqual(api_config.cors_origins, ())
        self.assertIsNone(api_config.token)
        # 不设整轮超时 = 0 = 不限时：本地跑长任务很常见，默认别动手
        self.assertEqual(api_config.run_timeout, DEFAULT_RUN_TIMEOUT)

    def test_the_values_are_read_from_the_environment(self):
        with patch.dict(
            os.environ,
            {
                "API_SESSION_TTL": "60",
                "API_MAX_SESSIONS": "4",
                "API_CORS_ORIGINS": "http://localhost:5173",
                "API_TOKEN": "口令",
                "AGENT_RUN_TIMEOUT": "12.5",
            },
            clear=True,
        ):
            api_config = load_api_config()

        self.assertEqual(api_config.session_ttl, 60.0)
        self.assertEqual(api_config.max_sessions, 4)
        self.assertEqual(api_config.cors_origins, ("http://localhost:5173",))
        self.assertEqual(api_config.token, "口令")
        self.assertEqual(api_config.run_timeout, 12.5)

    def test_an_explicit_zero_run_timeout_means_no_limit(self):
        # 0 是合法值（= 不限时），和「没设」是一个意思；负数和乱写的才拒
        with patch.dict(os.environ, {"AGENT_RUN_TIMEOUT": "0"}, clear=True):
            api_config = load_api_config()

        self.assertEqual(api_config.run_timeout, 0.0)

    def test_a_bad_run_timeout_is_rejected(self):
        for value in ("-1", "一会儿"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"AGENT_RUN_TIMEOUT": value}, clear=True
                ):
                    with self.assertRaises(ConfigError):
                        load_api_config()

    def test_origins_are_trimmed_deduped_and_slashless(self):
        with patch.dict(
            os.environ,
            {
                "API_CORS_ORIGINS": (
                    " http://localhost:5173/ , http://localhost:5173 ,"
                    " http://127.0.0.1:5173 "
                )
            },
            clear=True,
        ):
            api_config = load_api_config()

        self.assertEqual(
            api_config.cors_origins,
            ("http://localhost:5173", "http://127.0.0.1:5173"),
        )

    def test_a_blank_token_means_none(self):
        with patch.dict(os.environ, {"API_TOKEN": "   "}, clear=True):
            api_config = load_api_config()

        self.assertIsNone(api_config.token)

    def test_a_non_numeric_ttl_is_rejected(self):
        with patch.dict(os.environ, {"API_SESSION_TTL": "一会儿"}, clear=True):
            with self.assertRaises(ConfigError):
                load_api_config()

    def test_a_non_positive_ttl_is_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"API_SESSION_TTL": value}, clear=True):
                    with self.assertRaises(ConfigError):
                        load_api_config()

    def test_a_bad_max_sessions_is_rejected(self):
        for value in ("0", "-1", "2.5", "很多"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"API_MAX_SESSIONS": value}, clear=True):
                    with self.assertRaises(ConfigError):
                        load_api_config()

    def test_a_star_origin_is_rejected(self):
        # 放行整个 Web 等于把本机服务交给任何网页：配置层直接拦下
        with patch.dict(os.environ, {"API_CORS_ORIGINS": "*"}, clear=True):
            with self.assertRaises(ConfigError):
                load_api_config()
