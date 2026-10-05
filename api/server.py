"""HTTP 服务：标准库 http.server 上的五条路由，把 Agent 包成 JSON 接口。

选型：项目里只有 openai / python-dotenv 两个依赖，没有 Web 框架；这一阶段
一共五条路由，为此引一个框架不划算（规格也不许新增不必要的依赖）。以后
接口多了想换框架，只动这个文件——外面看到的始终只有 create_server()。

**这一层只做四件事**：解析参数、调 Agent、序列化结果、转换错误。Agent /
工具 / RAG / Memory / Call Graph 的逻辑一行都不复制：Agent.run() 是唯一
入口，它返回的字符串原样进 answer。

    POST   /api/run         {"task": "...", "session_id": "..."（可省）}
                            → {"success": true, "answer": "...", "session_id": "...",
                               "run_id": "..."}
    POST   /api/run/stream  同上，但用 SSE 边跑边推事件（13.2，见 api/sse.py）
    POST   /api/runs/cancel {"session_id": "...", "run_id": "..."（可省）}
                            → 把正在跑的那一轮停下来（13.4）
    GET    /api/health      （无请求体）→ {"status": "ok"}
    DELETE /api/sessions    {"session_id": "..."} → 删掉这个会话

**会话（13.3）**：session_id 由服务端在响应里发；请求不带它就新建一个会话，
带了就接着用那本会话历史。认不出来的 id 回 404，不悄悄换一只。会话的创建 /
清理全在 api/sessions.py；这一层只做「取一个会话 → 拿它的锁跑 Agent.run →
把 session_id 写进响应」。同一个会话的两次 run 排队跑（会话锁），不同会话
各跑各的——服务级不再有一把把所有用户串起来的大锁。

**Run 与取消（13.4）**：每次 run 请求服务端发一个 run_id，并建一个独立的
取消开关（agent/cancel.py），登记进 api/runs.py。POST /api/runs/cancel 按
session_id（可加 run_id）把那一轮停下来——**协作式**：Agent 在检查点上发现
取消才停，不杀线程、不硬杀子进程；正在跑的单工具让它自然结束。被停下的
这一轮回 200 + success=false + status（cancelled / timed_out），不装扮成
成功，也不是 5xx：服务没出错，是这一轮被停了。cancel 请求只碰登记表和那
一轮的开关，**不取会话锁**，所以正在跑的那一轮卡不住它。

**访问规则（13.3）**，按下面的顺序检查：
- Origin：带 Origin 头的请求必须落在 API_CORS_ORIGINS 里，否则 403——挡的
  是浏览器跨源请求，和「本机别的网页对 localhost 发简单请求」（本服务的
  run_command / edit_file 是真会动系统的）。命令行客户端不带 Origin，不受
  影响；允许的 Origin 会拿到 CORS 响应头，OPTIONS 预检单独处理。
- Token：配了 API_TOKEN 之后，除 /api/health 外都要带
  `Authorization: Bearer <token>`，缺 / 错回 401。health 保持开放：前端要
  能探活，它也不碰 Agent、不泄露任何数据。没配 Token = 本机开发直接可用。
- Content-Type：POST / DELETE 必须声明 application/json，否则 415——顺带
  把「任何网页用简单请求就能打进来」那条路也堵死。

**错误怎么变成响应**：不重试、不吞、不改成别的错误类型，只给一个状态码和
「success=false + answer 是说明」的 JSON——400 参数不对（缺 task / 空 task /
不是 JSON / session_id 不是字符串 / 客户端自带 run_id）、401 缺 Token 或
Token 不对、403 Origin 不允许、404 没这个路径 / session 不存在或已过期 /
没有这样的 Run、405 方法不对、413 请求体太大、415 Content-Type 不是 JSON、
500 Agent 失败或意外、502 模型调用失败。**取消和整轮超时不是错误**：200 +
success=false + status（cancelled / timed_out），前端按 status 分。
**任何出口都是 JSON**，前端只看 success 和 answer。

**Agent 装一次**：会话里的 Agent 来自同一个工厂（main.AgentFactory）——
工具 / 索引 / Memory store 就是靠「所有会话共用一份」才只建一次，新建会话
不重扫项目（见 api/sessions.py）。
"""

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import urlsplit

from agent import AgentError
from agent.cancel import TIMEOUT as TIMEOUT_REASON
from agent.cancel import RunCancelled
from api.runs import CANCELLED, COMPLETED, FAILED, TIMED_OUT, Run, RunRegistry
from api.sessions import SessionManager
from api.sse import TraceStreamer, shorten
from llm import LLMError

_LOGGER = logging.getLogger("api")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# 请求体上限：task 是一句自然语言，1 MiB 已经宽得离谱；再大直接回 413，
# 既是保护（不替别人把内存吃满），也让「读不完就断开」这件事有明确边界
MAX_BODY_BYTES = 1 << 20


class ApiHandler(BaseHTTPRequestHandler):
    """四条路由的处理器：解析 → 取会话 → Agent.run() → JSON / SSE。"""

    protocol_version = "HTTP/1.1"
    server_version = "UniversalCodeAgentAPI/0.2"

    # 单个连接上「一个请求」最多等多久（秒）。socketserver 的
    # StreamRequestHandler.setup() 看到这个类属性就会给 socket 设上
    # settimeout()，于是读 / 写超时都变成 socket.timeout——而
    # BaseHTTPRequestHandler.handle_one_request 本来就捕获 socket.timeout
    # 并关掉连接，所以不用改任何业务代码。
    #
    # 为什么必须有：这个类属性不设就是 stdlib 默认的 None（永不超时），
    # 而 ThreadingHTTPServer 是每连接一个线程、线程数没有上限。于是
    # 「连上来发一半请求就不发了」的客户端能把线程一直占着——这一步
    # 在读请求行/请求头，**发生在 Token 校验之前**，配了 token 也拦不住。
    # 同理，SSE 那条流是 Agent 线程自己同步写 socket 的：客户端不读了，
    # 写就会一直卡着，会话锁松不掉，取消（协作式置点）也等不到检查点。
    #
    # 取值：一轮 LLM 调用自己可以跑很久（LLM_TIMEOUT 默认 60 秒），但
    # 那段时间服务端**不在这条 socket 上等**——它是等上游 HTTP。这个超时
    # 管的是「跟客户端之间一次读写」，所以按客户端该有的响应速度定：
    # 请求体早就在本地了，客户端没有理由隔几十秒才吐下一口。
    timeout = 60.0

    # 只声明认识的方法；不认识的（TRACE 之类）由基类回 501。认识的里面，
    # 路径对、方法不对的给 405，见 _dispatch
    # 这几个名字是基类定死的（N802 指驼峰命名），只能照抄；这里只声明认识
    # 的方法，不认识的（TRACE 之类）由基类回 501
    def do_GET(self) -> None:  # noqa: N802  # 基类的命名就是这样
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._preflight()

    # -- 访问规则与分发 ------------------------------------------------------

    def _dispatch(self, method: str) -> None:
        """按访问规则和路由表分发；挡掉就 401 / 403 / 415，找不到就 404 / 405。"""
        path = self._path()

        # 1) Origin：带了 Origin 的基本都是浏览器发的。不在允许列表里就 403
        #    ——既挡住「别的网页对 127.0.0.1 发简单请求」，也让「前端还没配
        #    CORS」在第一个请求就说清楚，而不是让浏览器控制台去猜
        if not self._origin_allowed():
            self._reject_origin()
            return

        view = _ROUTES.get((method, path))
        if view is None:
            # 请求体不读掉的话，HTTP/1.1 长连接里剩下的字节会被当成下一个
            # 请求——回错误之前先把它读干净
            self._read_body()
            allowed = sorted(m for (m, p) in _ROUTES if p == path)
            if allowed:
                self._send_json(
                    405,
                    {
                        "success": False,
                        "answer": f"{path} 不支持 {method}，请用 {' / '.join(allowed)}",
                    },
                    extra_headers={"Allow": ", ".join(allowed)},
                )
            else:
                self._send_json(
                    404,
                    {
                        "success": False,
                        "answer": f"没有 {path} 这个接口（只有 {_ROUTE_HINT}）",
                    },
                )
            return

        # 2) Token：配了才校验；health 在 _OPEN_ROUTES 里，保持可探活
        if (method, path) not in _OPEN_ROUTES and not self._authorized():
            self.close_connection = True  # 请求体没读，这条连接用完就断
            self._send_json(
                401,
                {
                    "success": False,
                    "answer": "这个接口需要 API Token：请在请求头带上 "
                    "Authorization: Bearer <token>（没配 API_TOKEN 的本地服务不需要）",
                },
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return

        # 3) Content-Type：带 JSON 体的方法必须声明 application/json
        if method in _JSON_BODY_METHODS and not self._json_content_type():
            self.close_connection = True
            received = self.headers.get("Content-Type") or "没有 Content-Type"
            self._send_json(
                415,
                {
                    "success": False,
                    "answer": f"请求体要是 JSON：Content-Type 请用 "
                    f"application/json（收到：{received}）",
                },
            )
            return

        try:
            view(self)
        except Exception:
            # Agent 自己的异常在 _run 里已经转成 JSON 了，能走到这儿的都是
            # HTTP 层自己的意外（比如客户端半路断开）——记一笔，补一个 500，
            # 绝不让一个请求把服务带崩
            _LOGGER.exception("%s %s 处理失败", method, path)
            try:
                self._send_json(
                    500, {"success": False, "answer": "服务器内部错误"}
                )
            except OSError:
                pass  # 连接已经断了，没什么可回的

    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _origin_allowed(self) -> bool:
        """没带 Origin（命令行、服务端调用）放行；带了就必须在允许列表里。"""
        origin = self._origin()
        return origin is None or origin in cast("ApiServer", self.server).cors_origins

    def _reject_origin(self) -> None:
        """Origin 不在允许列表：403，请求体不读、连接用完就断。"""
        self.close_connection = True
        self._send_json(
            403,
            {
                "success": False,
                "answer": f"Origin {self._origin()!r} 不在允许列表里：浏览器前端"
                f"要先把它的 Origin 配进 API_CORS_ORIGINS（逗号分隔）；"
                f"命令行客户端不带 Origin，不受影响",
            },
        )

    def _cors_headers(self) -> dict[str, str]:
        """允许的 Origin 才给 CORS 响应头（不配 = 浏览器前端读不到响应）。

        只回显请求里的那一个 Origin、不回 `*`：允许多个前端时，浏览器要的
        是「正好是你」的答复。
        """
        origin = self._origin()
        if origin and self._origin_allowed():
            return {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
        return {}

    def _authorized(self) -> bool:
        """Token 校验：没配就是本地开发，直接放行；配了必须对上。

        比对用 hmac.compare_digest（按字节），不用 ==：Token 也是密钥，
        比较不该随「前几个字符对不对」变快变慢。
        """
        expected = cast("ApiServer", self.server).token
        if not expected:
            return True
        scheme, _, value = self.headers.get("Authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(
            value.strip().encode("utf-8"), expected.encode("utf-8")
        )

    def _json_content_type(self) -> bool:
        """Content-Type 的主类型是不是 application/json（charset 等参数不管）。"""
        value = self.headers.get("Content-Type", "")
        return value.split(";", 1)[0].strip().lower() == "application/json"

    def _preflight(self) -> None:
        """OPTIONS：浏览器的 CORS 预检——只回「这个来源能不能用这些方法」。

        不算业务请求：不回 JSON 本体（正常是 204 空响应），也不碰会话。
        没有 Origin 的 OPTIONS 不是预检，直接说清楚；路径不存在就 404，
        让拼错的地址在预检这一步就暴露。
        """
        origin = self._origin()
        if origin is None:
            self._send_json(
                400,
                {
                    "success": False,
                    "answer": "OPTIONS 只用于浏览器的 CORS 预检：请带上 Origin 头",
                },
            )
            return
        if not self._origin_allowed():
            self._reject_origin()
            return

        path = self._path()
        methods = sorted(m for (m, p) in _ROUTES if p == path)
        if not methods:
            self._send_json(
                404,
                {
                    "success": False,
                    "answer": f"没有 {path} 这个接口（只有 {_ROUTE_HINT}）",
                },
            )
            return

        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", ", ".join(methods))
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization"
        )
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- 路由 ---------------------------------------------------------------

    def _health(self) -> None:
        """GET /api/health：进程还在服务就 200。

        不查 Agent 的状态、也不碰会话表：它回答的是「服务活着吗」，不牵扯
        某个任务成没成、有几个会话（任务的结果在 /api/run 的响应里）。
        也正因为如此，它是唯一不校验 Token 的接口——前端要能探活。
        """
        self._send_json(200, {"status": "ok"})

    def _run(self) -> None:
        """POST /api/run：跑一轮，等结果。

        不带 session_id 就新建一个会话，带了就接着用；响应里总是回
        session_id 和这一轮的 run_id，下一次请求带上 session_id 就能继续
        这段对话（run_id 是拿去找 /api/runs/cancel 的）。

        被取消 / 超时：200 + success=false + status=cancelled / timed_out
        ——这一轮被停了，不是服务出错。
        """
        task, session_id, error, status = self._read_task()
        if error is not None:
            self._send_json(status, {"success": False, "answer": error})
            return
        # _read_task 的约定（见它的文档）：error 为 None 时 task 一定有值
        assert task is not None

        server = cast("ApiServer", self.server)
        session = server.sessions.acquire(session_id)
        if session is None:
            # 没给 session_id 时 acquire 会新建一个，不会落空——走到这儿说明
            # 给了一个认不出来的 id
            assert session_id is not None
            self._send_json(
                404, {"success": False, "answer": _unknown_session(session_id)}
            )
            return

        # 登记这一条 Run + 它自己的取消开关（配了 AGENT_RUN_TIMEOUT 的话，
        # 开关上带着整轮预算的到点时刻）。开跑之前就登记：排队等会话锁的
        # 这会儿，取消请求已经找得到它了
        run = server.runs.create(session.id, timeout=server.run_timeout or None)
        server.runs.begin(run)

        try:
            # 同一个会话串行（messages 是共享的会话历史）；不同会话各跑各的
            with session.lock:
                try:
                    answer = session.agent.run(task, cancellation=run.cancellation)
                finally:
                    # 跑完记一次「刚用过」。不记的话，一个跑得比
                    # API_SESSION_TTL 还长的任务，跑完那一刻自己就已经
                    # 算过期了——客户端接着追问会先被清掉再拿到 404。
                    server.sessions.touch(session)
        except RunCancelled as exc:
            # 协作式停下：Agent 自己在检查点停的，消息已回滚。不是失败，
            # 也更不是 5xx——给 200 + 明确的 status
            final = _stopped_status(exc.reason)
            server.runs.finish(run, final)
            self._send_json(200, _stopped_result(final, run))
        except LLMError as exc:
            # 502：请求本身没毛病，是上游模型那边没成
            server.runs.finish(run, FAILED)
            self._send_json(502, _run_error(f"调用模型失败：{exc}", run))
        except AgentError as exc:
            server.runs.finish(run, FAILED)
            self._send_json(500, _run_error(f"Agent 运行失败：{exc}", run))
        except Exception as exc:
            # 兜底：Agent 抛什么都不能让 HTTP 层崩。写法沿用 ToolExecutor 的
            # 「类型: 消息」，前端一眼能看出是什么错
            server.runs.finish(run, FAILED)
            self._send_json(500, _run_error(f"{type(exc).__name__}: {exc}", run))
        else:
            server.runs.finish(run, COMPLETED)
            self._send_json(
                200,
                {
                    "success": True,
                    "answer": answer,
                    "session_id": session.id,
                    "run_id": run.id,
                },
            )
        finally:
            # 兜底，同 _run_stream：上面每条出口都自己 finish 过了，这只防
            # except Exception 都接不住的 BaseException——登记的 Run 绝不
            # 允许活着离开这个函数
            if not run.finished:
                server.runs.finish(run, FAILED)

    def _run_stream(self) -> None:
        """POST /api/run/stream：用 SSE 把这一轮跑的过程边跑边推。

        事件顺序：start →（每次模型调用一组 llm_start / llm_end、每个工具一组
        tool_start / tool_result）→ completed；出错就用 error 收尾（不再发
        completed），被取消 / 超时则用 cancelled / timed_out 收尾——都是流里
        的一等事件，然后正常关闭流，响应本身仍然是 200。start / completed /
        error / cancelled / timed_out / llm_start / llm_end 里都带 session_id
        + run_id。llm_*（13.5）只有模型名、尝试次数、耗时、token 用量和错误
        摘要——prompt、回复正文、key 一个都不进流。

        参数不合法（缺 task、不是 JSON 之类）走普通 JSON 的 400；session_id
        认不出来走 404——都还没开头，没必要摆 SSE 的架势，前端也能和
        /api/run 用同一套判断。

        客户端断开**不会**自动取消这一轮（断线 ≠ 按停，分不清）；要停就
        POST /api/runs/cancel，那一轮的流最后仍会收到 cancelled / timed_out。
        """
        task, session_id, error, status = self._read_task()
        if error is not None:
            self._send_json(status, {"success": False, "answer": error})
            return
        # _read_task 的约定（见它的文档）：error 为 None 时 task 一定有值
        assert task is not None

        server = cast("ApiServer", self.server)
        session = server.sessions.acquire(session_id)
        if session is None:
            # 同 _run：没给 session_id 不会落空，落空就是给的 id 认不出来
            assert session_id is not None
            self._send_json(
                404, {"success": False, "answer": _unknown_session(session_id)}
            )
            return

        run = server.runs.create(session.id, timeout=server.run_timeout or None)
        server.runs.begin(run)

        # 把这一轮的身份交给流：llm_* 事件要带上它们（start / completed /
        # error 等由这里自己发，本来就带着）
        streamer = TraceStreamer(
            self._write_frame, session_id=session.id, run_id=run.id
        )
        try:
            # 开流和第一帧都放在 try 里面。它俩是这一轮里最早往客户端写字节
            # 的地方，而「客户端连上了、建流之前就断开」正是炸在这一步
            # （BrokenPipeError）。以前它们在 try 外面：异常直接从
            # _run_stream 抛出去，下面每个出口的 runs.finish() 一次都轮不到
            # ——_active 里留下一条永远 running 的幽灵。而 active_for() 找的
            # 是**最老**的那条，于是同一个会话往后每次「取消当前这一轮」都
            # 打在这条幽灵上，真正在跑的那一轮反而停不下来。
            self._begin_event_stream()
            streamer.send(
                "start", {"task": task, "session_id": session.id, "run_id": run.id}
            )

            # 和 /api/run 同一把「会话锁」：同一个会话的流式请求照样排队，
            # 绝不和另一个请求同时碰同一本会话历史；不同会话各拿各的锁
            with session.lock:
                try:
                    answer = session.agent.run(
                        task,
                        on_tool_start=streamer.on_tool_start,
                        on_tool_end=streamer.on_tool_end,
                        cancellation=run.cancellation,
                        on_llm_start=streamer.on_llm_start,
                        on_llm_end=streamer.on_llm_end,
                    )
                finally:
                    # 同 _run：跑得比 TTL 还长的一轮，跑完不能自己算过期
                    server.sessions.touch(session)
        except RunCancelled as exc:
            final = _stopped_status(exc.reason)
            server.runs.finish(run, final)
            streamer.send(final, _stream_stopped(final, run))
        except LLMError as exc:
            server.runs.finish(run, FAILED)
            streamer.send("error", _stream_error(f"调用模型失败：{exc}", run))
        except AgentError as exc:
            server.runs.finish(run, FAILED)
            streamer.send("error", _stream_error(f"Agent 运行失败：{exc}", run))
        except Exception as exc:
            server.runs.finish(run, FAILED)
            streamer.send(
                "error", _stream_error(f"{type(exc).__name__}: {exc}", run)
            )
        else:
            server.runs.finish(run, COMPLETED)
            # completed 带最终回答：它就是 /api/run 会返回的那一份，前端
            # 不额外发一次请求也能拿到结果
            streamer.send(
                "completed",
                {
                    "success": True,
                    "answer": answer,
                    "session_id": session.id,
                    "run_id": run.id,
                },
            )
        finally:
            # 上面每条出口都自己 finish 过了，正常走到这儿什么都不做。留着
            # 是给 except Exception 也接不住的那种（BaseException：Ctrl-C /
            # SystemExit）：登记的 Run 绝不允许活着离开这个函数。
            # 先看状态再补，避免把已经落定的终态又记一次（重复 finish 会白
            # 占一格 _finished 的名额，把别人挤出去）。
            if not run.finished:
                server.runs.finish(run, FAILED)

    def _delete_session(self) -> None:
        """DELETE /api/sessions：{"session_id": "..."} → 删掉这个会话。

        删的是会话那本账（它自己的 Agent / messages）；共享的工具、索引、
        Memory store 一个都不动。删不存在的 id 回 404——和「拿它跑任务」
        是同一种「这个会话没了」，客户端按同一句话处理。
        """
        data, error, status = self._read_json_object()
        if error is not None:
            self._send_json(status, {"success": False, "answer": error})
            return
        # _read_json_object 的约定（见它的文档）：error 为 None 时 data 是个对象
        assert data is not None

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            self._send_json(
                400,
                {
                    "success": False,
                    "answer": '要一个字符串的 session_id，例如 {"session_id": "abc123"}',
                },
            )
            return
        session_id = session_id.strip()

        if cast("ApiServer", self.server).sessions.delete(session_id):
            self._send_json(
                200, {"success": True, "session_id": session_id, "deleted": True}
            )
        else:
            self._send_json(
                404, {"success": False, "answer": _unknown_session(session_id)}
            )

    def _cancel_run(self) -> None:
        """POST /api/runs/cancel：把正在跑的那一轮停下来（协作式）。

        请求：{"session_id": "..."}（必填，取消这个会话当前的 Run），可加
        "run_id" 精确指定（只认服务端发过的那个）。响应带 run_id / status /
        cancelled（本次是不是真正按下了取消）。

        只碰 api/runs.py 的登记表和那一轮的取消开关，**不取会话锁**：正在
        跑的那一轮（还攥着会话锁）绝卡不住这个请求。取消是送出一个信号，
        不是当场叫停——Agent 在下一次检查点自己停，停之前 status 还是
        running，这是正常的。

        已结束的 Run 再 cancel 是**幂等**的：照样 200，只是 cancelled=false、
        status 是它当时的终态。跨会话的 Run 当作不存在（404），不泄露别人
        的 run_id。
        """
        data, error, status = self._read_json_object()
        if error is not None:
            self._send_json(status, {"success": False, "answer": error})
            return
        # _read_json_object 的约定（见它的文档）：error 为 None 时 data 是个对象
        assert data is not None

        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            self._send_json(
                400,
                {
                    "success": False,
                    "answer": '要一个字符串的 session_id，例如 {"session_id": "abc123"}'
                    '（可加 "run_id" 精确指定某一条 Run）',
                },
            )
            return
        session_id = session_id.strip()

        run_id = data.get("run_id")
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id.strip():
                self._send_json(
                    400,
                    {
                        "success": False,
                        "answer": f"run_id 要是非空字符串（或不给），收到 {run_id!r}",
                    },
                )
                return
            run_id = run_id.strip()

        runs = cast("ApiServer", self.server).runs
        if run_id is None:
            # 只给会话：取消它当前在跑的那一条（排最前面 = 真正在跑的那条）
            run = runs.active_for(session_id)
            if run is None:
                self._send_json(
                    404,
                    {"success": False, "answer": _no_active_run(session_id)},
                )
                return
            started = run.cancellation.cancel()
        else:
            found = runs.cancel(run_id, session_id)
            if found is None:
                self._send_json(
                    404,
                    {"success": False, "answer": _unknown_run(run_id, session_id)},
                )
                return
            run, started = found

        self._send_json(
            200,
            {
                "success": True,
                "session_id": session_id,
                "run_id": run.id,
                "status": run.status,
                "cancelled": started,
            },
        )

    # -- 请求 ---------------------------------------------------------------

    def _path(self) -> str:
        """请求路径：去掉查询串和末尾斜杠（/api/health/ 和 /api/health 一样）。"""
        return urlsplit(self.path).path.rstrip("/") or "/"

    def _read_body(self) -> tuple[bytes, str | None, int]:
        """把请求体读出来；成功给 (bytes, None, 0)，不合法给 (b"", 说明, 状态码)。

        Content-Length 缺失 / 不合法 / 超限 / 是 0，都算「读不出来」。这些
        情况不能保证请求体被读干净了，所以顺手把这条连接标记成用完就断：
        残留字节留在长连接里，下一次请求会从中间开始解析。
        """
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else -1
        except ValueError:
            size = -1

        if size < 0:
            error, status = '请求体不能为空，要 JSON：{"task": "..."}', 400
        elif size > MAX_BODY_BYTES:
            error, status = (
                f"请求体太大：{size} 字节，上限 {MAX_BODY_BYTES} 字节",
                413,
            )
        else:
            error, status = None, 0

        if error is not None:
            self.close_connection = True
            return b"", error, status
        return self.rfile.read(size), None, 0

    def _read_json_object(self) -> tuple[dict[str, Any] | None, str | None, int]:
        """读请求体并解析成一个 JSON 对象：成功 (dict, None, 0)。

        不合法给 (None, 说明, 状态码)——都是 400 那一类。两个带 JSON 体的
        接口（run / sessions）共用这一段，错误说法也就只有一份。
        """
        raw, error, status = self._read_body()
        if error is not None:
            return None, error, status

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"请求体不是合法 JSON：{exc}", 400

        if not isinstance(data, dict):
            return None, '请求体要是一个 JSON 对象，例如 {"task": "..."}', 400
        return data, None, 0

    def _read_task(self) -> tuple[str | None, str | None, str | None, int]:
        """取出并校验 task 和（可选的）session_id。

        成功给 (task, session_id 或 None, None, 0)，不合法给
        (None, None, 说明, 状态码)。校验全在 HTTP 层做完（缺 / 空 / 类型
        不对），绝不让它们走到 Agent.run() 去撞 ValueError——那是 500，
        而这里每一条都是 400。session_id 只做「是个非空字符串」的形式
        校验；它在不在、过没过期由 SessionManager 判断（认不出来回 404）。
        run_id（13.4）一律服务端生成，请求里带了就直接拒——客户端伪造不了
        一条 Run 的身份。
        """
        data, error, status = self._read_json_object()
        if error is not None:
            return None, None, error, status
        # _read_json_object 的约定（见它的文档）：error 为 None 时 data 是个对象
        assert data is not None

        task = data.get("task")
        if task is None:
            return None, None, '请求体里缺少 task 字段，例如 {"task": "..."}', 400
        if not isinstance(task, str):
            return None, None, f"task 必须是字符串，收到 {type(task).__name__}", 400
        if not task.strip():
            return None, None, "task 不能是空的", 400

        if data.get("run_id") is not None:
            return (
                None,
                None,
                "run_id 由服务端生成（在响应 / SSE 事件里发），请求里不要带",
                400,
            )

        session_id = data.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str):
                return (
                    None,
                    None,
                    f"session_id 必须是字符串，收到 {type(session_id).__name__}"
                    f"（省略它就新建一个会话）",
                    400,
                )
            session_id = session_id.strip()
            if not session_id:
                return None, None, "session_id 不能是空的（省略它就新建一个会话）", 400

        return task, session_id, None, 0

    # -- 响应 ---------------------------------------------------------------

    def _begin_event_stream(self) -> None:
        """SSE 的响应头。

        没有 Content-Length——流多长事先不知道，只能以「写完关连接」收尾，
        所以明写 Connection: close 并把 close_connection 立起来：不写的话
        HTTP/1.1 的客户端会一直等「后面还有没有」。允许的 Origin 在这里也
        要拿到 CORS 头，不然前端连流都读不到。
        """
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        for name, value in self._cors_headers().items():
            self.send_header(name, value)
        self.end_headers()

    def _write_frame(self, payload: bytes) -> None:
        """往响应里写一帧。wfile 不带缓冲，flush 一下更稳，也不贵。"""
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        """回一个 JSON 响应：UTF-8 编码，Content-Length 如实写。

        HTTP/1.1 的长连接靠 Content-Length 分段，写错了后面全乱。所有出口
        都走这里，也就保证了「响应一定是 JSON」；允许的 Origin 也在这儿
        统一拿到 CORS 头（403 那条路走不到这儿，因为它的 Origin 不允许）。
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            # 这次响应完就断（请求体没读干净 / 请求被挡了）：明说一句，
            # 别让对面猜自己发的字节有没有被收下
            self.send_header("Connection", "close")
        headers = {**self._cors_headers(), **(extra_headers or {})}
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """请求日志走 logging（INFO 级），不直接写 stderr。

        默认没人配置 logging 时 INFO 不输出，测试里干净；python -m api 里
        basicConfig 一开就有日志看。
        """
        _LOGGER.info("%s %s", self.address_string(), format % args)


def _unknown_session(session_id: str) -> str:
    """认不出来的 session_id 统一这一句：说清楚 + 怎么拿一个新的。"""
    return (
        f"session 不存在或已过期：{session_id}（session_id 由服务端在响应里发，"
        f"带上它继续这段对话；不传 session_id 就新建一个）"
    )


def _no_active_run(session_id: str) -> str:
    """这个会话没有在跑的 Run：说清楚（也覆盖 session 不认识的情况）。"""
    return (
        f"这个会话当前没有在跑的 Run：{session_id}（可能已经结束、session_id "
        f"不对或已过期；要精确指定某一条可以带上 run_id）"
    )


def _unknown_run(run_id: str, session_id: str) -> str:
    """找不到这条 Run：跨会话的也走这句——不泄露别人的 run_id。"""
    return (
        f"没有这样的 Run：{run_id}（run 不存在、不属于这个会话、"
        f"或结束太久已被清理）"
    )


# 被取消 / 超时的这一轮，answer 里说的话。两种分开说：一个有人按了停，
# 一个是预算用完了——前端 / 用户该看到的原因不一样
_STOPPED_MESSAGES = {
    CANCELLED: "这一轮已被取消：Agent 在下一个检查点停下了，"
    "已产生的对话已回滚，会话可以继续用",
    TIMED_OUT: "这一轮超过了整轮时间上限（AGENT_RUN_TIMEOUT）：已按超时停下，"
    "不是模型或工具本身报错",
}


def _stopped_status(reason: str) -> str:
    """RunCancelled 的原因 → Run 的终态；只有「超时」分家。"""
    return TIMED_OUT if reason == TIMEOUT_REASON else CANCELLED


def _stopped_result(status: str, run: Run) -> dict[str, Any]:
    """被取消 / 超时的响应体：200 + success=false + 明确的 status。"""
    return {
        "success": False,
        "answer": _STOPPED_MESSAGES[status],
        "session_id": run.session_id,
        "run_id": run.id,
        "status": status,
    }


def _run_error(message: str, run: Run) -> dict[str, Any]:
    """跑任务失败时的响应体：说明 + 会话 / Run 的 id + status=failed。

    调用方总会先把这一轮登记成 failed，这里只是把终态一起告诉客户端。
    """
    return {
        "success": False,
        "answer": message,
        "session_id": run.session_id,
        "run_id": run.id,
        "status": FAILED,
    }


def _stream_error(message: str, run: Run) -> dict[str, Any]:
    """error 事件的载荷：截断的说明 + 会话 id / run_id（和 start 对齐）。"""
    return {
        "message": shorten(message),
        "session_id": run.session_id,
        "run_id": run.id,
    }


def _stream_stopped(status: str, run: Run) -> dict[str, Any]:
    """cancelled / timed_out 事件的载荷：和 error 一个形状，名字换成终态。"""
    return {
        "message": shorten(_STOPPED_MESSAGES[status]),
        "session_id": run.session_id,
        "run_id": run.id,
    }


# 路由表：(方法, 路径) → 处理方法。加接口就是往这儿补一行——405 / 404 和
# Allow 头、OPTIONS 预检的放行方法都是从这张表推出来的，不用另写一份
_ROUTES = {
    ("GET", "/api/health"): ApiHandler._health,
    ("POST", "/api/run"): ApiHandler._run,
    ("POST", "/api/run/stream"): ApiHandler._run_stream,
    ("POST", "/api/runs/cancel"): ApiHandler._cancel_run,
    ("DELETE", "/api/sessions"): ApiHandler._delete_session,
}

# 不校验 Token 的接口：health 要能被前端探活，它也不碰 Agent、不泄露数据
_OPEN_ROUTES = frozenset({("GET", "/api/health")})

# 带 JSON 体的方法：必须先声明 Content-Type: application/json（见 _dispatch）
_JSON_BODY_METHODS = frozenset({"POST", "DELETE"})

# 404 的提示语从路由表推：以后加接口不用记得回来改文案
_ROUTE_HINT = "、".join(
    f"{method} {path}"
    for method, path in sorted(_ROUTES, key=lambda route: (route[1], route[0]))
)


class ApiServer(ThreadingHTTPServer):
    """装着会话表、Run 登记表和访问规则的服务。

    多线程是为了让不同会话（和健康检查、取消请求）互不排队；真正跑任务的
    串行化在每个 Session 自己的锁上（见 api/sessions.py），这里不再有一把
    全局的 agent_lock。daemon_threads：主线程一退，还挂着的请求线程不拦着
    进程结束。
    """

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        sessions: SessionManager,
        *,
        cors_origins: tuple[str, ...] = (),
        token: str | None = None,
        runs: RunRegistry | None = None,
        run_timeout: float = 0.0,
    ) -> None:
        super().__init__(address, ApiHandler)
        self.sessions = sessions
        # frozenset：请求进来只查「在不在」，顺序没有意义
        self.cors_origins = frozenset(cors_origins)
        self.token = token  # None = 不校验（本机开发）
        # run_id → Run（状态 + 取消开关，见 api/runs.py）。取消接口靠它
        # 找到目标，不碰会话锁；默认一条共享的，测试可以换一个
        self.runs = runs if runs is not None else RunRegistry()
        # 整轮超时（秒），0 = 不限时；到点由 Agent 在检查点抛 RunCancelled
        self.run_timeout = run_timeout


def create_server(
    sessions: SessionManager,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    cors_origins: tuple[str, ...] = (),
    token: str | None = None,
    run_timeout: float = 0.0,
) -> ApiServer:
    """建好服务（还没开始 serve_forever）：port=0 让系统挑一个空闲端口。

    sessions 由调用方装配（入口是 api/__main__.py，测试里现造）——里面
    那只工厂握着工具 / 索引 / Memory 这些共享件，服务这边只认它。
    cors_origins / token 是 13.3 的访问规则，run_timeout 是 13.4 的整轮
    超时；默认「不放任何 Origin、不校验 Token、不限时」——本机开发的行为
    和 13.2 一字不差。
    """
    return ApiServer(
        (host, port),
        sessions,
        cors_origins=cors_origins,
        token=token,
        run_timeout=run_timeout,
    )
