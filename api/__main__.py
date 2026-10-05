"""HTTP API 的启动入口：`python -m api`。

单独一个入口，和 `python main.py`（命令行对话）互不影响：两边都是
「装配一次共享件，然后围着一只（这边是很多只）Agent 服务」，区别只是
这边把它挂在 HTTP 后面。配置也是同一套——模型、Memory、embedding 全走
main.build_agent_factory()，会话、安全边界与整轮超时（AGENT_RUN_TIMEOUT）
走 config.load_api_config()，.env 怎么配的这里就怎么用。

    python -m api                  # 127.0.0.1:8000（只服务本机）
    python -m api --port 9000
"""

import argparse
import logging
import sys
from typing import cast

from api.server import DEFAULT_HOST, DEFAULT_PORT, create_server
from api.sessions import SessionManager
from config import ApiConfig, ConfigError, load_api_config
from main import build_agent_factory


def security_summary(api_config: ApiConfig) -> str:
    """把会话 / CORS / Token / 整轮超时的当前状态打一行（**不打 Token 本身**）。"""
    origins = "、".join(api_config.cors_origins) or "无（浏览器前端接上来之前先配 API_CORS_ORIGINS）"
    token = (
        "已启用（受保护接口要带 Authorization: Bearer …）"
        if api_config.token
        else "未启用（本机开发）"
    )
    timeout = (
        f"{api_config.run_timeout:g} 秒" if api_config.run_timeout else "不限时"
    )
    return (
        f"会话：TTL {api_config.session_ttl:g} 秒，最多 {api_config.max_sessions} 个"
        f"  允许的 Origin：{origins}  API Token：{token}"
        f"  整轮超时（一次 Run 的总预算）：{timeout}"
    )


def main(argv: list[str] | None = None) -> int:
    """起服务并一直跑；Ctrl-C 停下。返回进程退出码。"""
    parser = argparse.ArgumentParser(
        prog="api", description="给 Universal Code Analysis Agent 起一个 HTTP 接口"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"监听地址（默认 {DEFAULT_HOST}，只服务本机；"
        f"绑到 0.0.0.0 等于对局域网开放，动手前先配 API_TOKEN）",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"监听端口（默认 {DEFAULT_PORT}）"
    )
    args = parser.parse_args(argv)

    # 装配失败（比如 .env 里模型配置不对、会话参数不是正数）在起服务之前
    # 就报出来：带着一份没配好的配置起服务，每个请求都会失败，不如现在
    # 就说清楚
    try:
        factory = build_agent_factory()
        api_config = load_api_config()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 1

    # 会话表：一个 session_id 一只 Agent；工具 / 索引 / Memory store 是
    # 工厂手里的那一份共享件，新建会话不重扫项目（见 api/sessions.py）
    sessions = SessionManager(
        factory.new_agent,
        ttl=api_config.session_ttl,
        max_sessions=api_config.max_sessions,
    )

    # 请求日志走 logging（见 api/server.py 的 log_message）：入口这里开一次
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    server = create_server(
        sessions,
        host=args.host,
        port=args.port,
        cors_origins=api_config.cors_origins,
        token=api_config.token,
        run_timeout=api_config.run_timeout,
    )
    # server_address 的类型是「str 或 bytes 的地址」两可（stdlib 的签名如此），
    # 这里绑的一定是 str host（create_server 拿的是 args.host），写明白省得
    # 让人以为会打出 b'127.0.0.1'
    host, port = cast("tuple[str, int]", server.server_address[:2])
    print(
        f"API 服务已启动：http://{host}:{port}"
        f"  （POST /api/run，POST /api/run/stream，POST /api/runs/cancel，"
        f"GET /api/health，DELETE /api/sessions；Ctrl-C 退出）"
    )
    print(security_summary(api_config))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
