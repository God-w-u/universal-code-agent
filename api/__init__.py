"""HTTP API：把 Agent 包成一组最小的 JSON 接口，给以后的前端用。

这一层只当搬运工：解析请求 → 取会话 → 调 Agent.run() → 序列化结果 /
转换错误，Agent / 工具 / RAG / Memory / Call Graph 的逻辑一行都不复制。

- server：路由、请求解析、响应序列化、错误到状态码的映射，以及 Origin /
  Token / Content-Type 三道检查（13.3 的访问规则）；
- sessions：session_id → Session（一只独立 Agent），创建 / 复用 / 删除 /
  过期清理都在这层——不同会话的对话历史绝不互相看见（13.3）；
- runs：run_id → Run（状态 + 取消开关），取消接口靠它找到目标（13.4）；
- sse：SSE 帧的拼法 + 把 Trace 通知翻译成事件（13.2 的流式接口）；
- __main__：`python -m api` 启动，和 `python main.py` 的 CLI 互不影响。

接口：`POST /api/run`（一次给结果）、`POST /api/run/stream`（SSE 边跑边推）、
`POST /api/runs/cancel`（停掉正在跑的一轮）、`GET /api/health`、
`DELETE /api/sessions`；run 的两个接口都收可选 session_id，响应里总是回
session_id + run_id。

共享件（工具注册表、索引、模型客户端、Memory store）在启动时装配一次，
所有会话共用——新建会话不重扫项目；Conversation 的 messages 则每会话
一份，绝不共享。Run 的登记表也是一条共享的（服务上只有一张）。
"""

from api.runs import (
    CANCELLED,
    COMPLETED,
    CREATED,
    FAILED,
    FINISHED,
    RUNNING,
    TIMED_OUT,
    Run,
    RunRegistry,
)
from api.server import DEFAULT_HOST, DEFAULT_PORT, ApiServer, create_server
from api.sessions import DEFAULT_MAX_SESSIONS, DEFAULT_TTL, Session, SessionManager

__all__ = [
    "ApiServer",
    "create_server",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "Session",
    "SessionManager",
    "DEFAULT_TTL",
    "DEFAULT_MAX_SESSIONS",
    "Run",
    "RunRegistry",
    "CREATED",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMED_OUT",
    "FINISHED",
]
