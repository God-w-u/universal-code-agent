"""项目配置。

所有敏感信息（API Key 等）都从环境变量读取，不硬编码在代码里。
系统环境变量优先；若项目根目录存在 .env 文件，则作为补充来源。
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

# 下面两个读数字的函数：parse=int 就回 int（重试次数、会话条数），
# parse=float 就回 float（秒数）——调用方拿到的类型跟着 parse 走
_Number = TypeVar("_Number", int, float)

PROJECT_ROOT = Path(__file__).resolve().parent

# override=False：已存在的系统环境变量优先，不被 .env 覆盖。
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """配置缺失或不合法。"""


# 重试（13.5）：临时错误（429 / 5xx / 连接超时）最多再试几次、每跳之间等多久。
# 默认保守——最多 2 次重试（= 最多 3 次尝试）、基础退避 0.5 秒（指数：0.5、1.0）。
# 0 都是合法值：0 次重试 = 一次就抛，0 秒退避 = 不等（测试里常用）。用
# LLM_MAX_RETRIES / LLM_RETRY_BACKOFF 覆盖，见 .env.example。
LLM_MAX_RETRIES_ENV = "LLM_MAX_RETRIES"
LLM_RETRY_BACKOFF_ENV = "LLM_RETRY_BACKOFF"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 0.5


@dataclass(frozen=True)
class LLMConfig:
    """调用 LLM 所需的配置。"""

    api_key: str
    model: str
    base_url: str | None = None
    timeout: float = 60.0
    temperature: float = 0.0
    # 临时错误最多重试几次（0 = 一次就抛），每次重试前的基础退避秒数
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff: float = DEFAULT_RETRY_BACKOFF


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"缺少环境变量 {name}。请把 .env.example 复制为 .env 并填写，"
            f"或直接设置该环境变量后重试。"
        )
    return value


# search_code_chunks 的 embedding 模式（9.4）。off = 纯字面检索（默认，
# 行为和以前一字不差）；其余三个和 index.providers.PROVIDERS 的名字对齐
# ——下面有测试钉着两边一致，往 PROVIDERS 加新 provider 时别忘了这里。
SEARCH_EMBEDDING_MODES = ("off", "fake", "local", "multilingual")

# 读哪个环境变量（写进 .env 或直接设系统环境变量都行）
SEARCH_EMBEDDING_ENV = "SEARCH_CHUNKS_EMBEDDING"


def load_search_embedding() -> str:
    """search_code_chunks 用哪种 embedding：从环境变量读，默认 off。

    环境变量没设 / 是空的 → off（纯字面）。值不合法就抛 ConfigError——
    启动时立刻报错，而不是等第一次检索才发现拼错了。
    """
    value = os.getenv(SEARCH_EMBEDDING_ENV, "").strip().lower()
    if not value:
        return "off"
    if value not in SEARCH_EMBEDDING_MODES:
        raise ConfigError(
            f"{SEARCH_EMBEDDING_ENV} 只能是 "
            f"{'、'.join(SEARCH_EMBEDDING_MODES)} 之一，当前值：{value!r}"
        )
    return value


# Memory（11.2）：Agent 的历史记忆。默认**开**——历史攒着才有用，第一轮
# 就是空库，没有任何额外行为。要关就设 AGENT_MEMORY=off（或给 Agent 直接
# 传 memory=None）；数据文件位置用 AGENT_MEMORY_PATH 覆盖。
MEMORY_MODES = ("on", "off")
MEMORY_ENABLED_ENV = "AGENT_MEMORY"
MEMORY_PATH_ENV = "AGENT_MEMORY_PATH"

# Memory 的两个上限（11.3）：最多留几条、文件最多多少字节（超了丢最旧
# 的，见 memory/maintenance.py）。默认值和那边保持一致——往那边改默认
# 值别忘了这里，有测试钉着两处相等。用 AGENT_MEMORY_MAX_RECORDS /
# AGENT_MEMORY_MAX_BYTES 覆盖，只认正整数。
MEMORY_DEFAULT_MAX_RECORDS = 500
MEMORY_DEFAULT_MAX_BYTES = 2 * 1024 * 1024
MEMORY_MAX_RECORDS_ENV = "AGENT_MEMORY_MAX_RECORDS"
MEMORY_MAX_BYTES_ENV = "AGENT_MEMORY_MAX_BYTES"


@dataclass(frozen=True)
class MemoryConfig:
    """Memory 的开关、落盘位置和两个上限。

    path 为 None = 用 memory.store.default_memory_path() 的默认位置
    （~/.cache/universal-code-agent/memory.jsonl，项目之外）。
    max_records / max_bytes = 历史最多留几条、文件最多多少字节——超了
    就自动丢最旧的，永远保留最新（见 memory/maintenance.py）。
    """

    enabled: bool
    path: Path | None = None
    max_records: int = MEMORY_DEFAULT_MAX_RECORDS
    max_bytes: int = MEMORY_DEFAULT_MAX_BYTES


def _memory_limit(name: str, default: int) -> int:
    """读一个正整数上限：没设用默认；不是整数 / 小于 1 都抛 ConfigError。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(
            f"{name} 必须是整数（条数 / 字节数），当前值：{raw!r}"
        ) from None
    if value < 1:
        raise ConfigError(f"{name} 至少为 1，当前值：{value}")
    return value


def load_memory_config() -> MemoryConfig:
    """Memory 配置：从环境变量读，默认开启、默认路径、默认上限。

    环境变量没设 / 是空的 → 开；AGENT_MEMORY 只认 on / off，其它值抛
    ConfigError——启动时立刻报错，而不是等第一次检索才发现拼错了。
    两个上限必须是正整数，拼错 / 填成 0 同样在启动时抛 ConfigError。
    路径只做「非空即用」：文件不存在、目录建不出来这些都属于运行期降级
    （检索 / 写入失败不影响任务，见 memory/context.py），不在启动时拦。
    """
    mode = os.getenv(MEMORY_ENABLED_ENV, "").strip().lower()
    if not mode:
        enabled = True
    elif mode in MEMORY_MODES:
        enabled = mode == "on"
    else:
        raise ConfigError(
            f"{MEMORY_ENABLED_ENV} 只能是 {'、'.join(MEMORY_MODES)} 之一，"
            f"当前值：{mode!r}"
        )

    raw_path = os.getenv(MEMORY_PATH_ENV, "").strip()
    return MemoryConfig(
        enabled=enabled,
        path=Path(raw_path) if raw_path else None,
        max_records=_memory_limit(MEMORY_MAX_RECORDS_ENV, MEMORY_DEFAULT_MAX_RECORDS),
        max_bytes=_memory_limit(MEMORY_MAX_BYTES_ENV, MEMORY_DEFAULT_MAX_BYTES),
    )


# ── HTTP 服务（13.3）：会话生命周期 + 安全边界 + 整轮超时（13.4）────
#
# 会话：多久没人用就过期（秒）、最多同时留几个（超了丢最久没用那个）。
# 安全：哪些前端 Origin 允许跨源访问（默认一个都不放）；设了 API_TOKEN
# 之后受保护接口必须带 Authorization: Bearer <token>（不设 = 本机开发
# 直接用）。整轮超时：一次 Run 从受理到结束的总预算（秒），0 = 不限时。
# 五个都是选填，值不合法在启动时抛 ConfigError。
API_SESSION_TTL_ENV = "API_SESSION_TTL"
API_MAX_SESSIONS_ENV = "API_MAX_SESSIONS"
API_CORS_ORIGINS_ENV = "API_CORS_ORIGINS"
API_TOKEN_ENV = "API_TOKEN"
AGENT_RUN_TIMEOUT_ENV = "AGENT_RUN_TIMEOUT"

DEFAULT_SESSION_TTL = 3600.0  # 一小时没人用过就清掉
DEFAULT_MAX_SESSIONS = 32     # 本地服务够用；超了丢最久没用的那个
DEFAULT_RUN_TIMEOUT = 0.0     # 不限时：本地跑长任务很常见，默认别动手


@dataclass(frozen=True)
class ApiConfig:
    """HTTP 服务这一层的配置：会话生命周期 + 安全边界 + 整轮超时。"""

    session_ttl: float = DEFAULT_SESSION_TTL
    max_sessions: int = DEFAULT_MAX_SESSIONS
    cors_origins: tuple[str, ...] = ()
    token: str | None = None
    # 一次 Run 的总预算（秒，LLM + 工具 + 修复共享）；0 = 不限时
    run_timeout: float = DEFAULT_RUN_TIMEOUT


def _positive_number(
    name: str,
    default: _Number,
    *,
    what: str,
    parse: Callable[[str], _Number],
) -> _Number:
    """读一个大于 0 的数：没设用默认；不是数 / 不大于 0 都抛 ConfigError。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        raise ConfigError(f"{name} 必须是{what}，当前值：{raw!r}") from None
    if value <= 0:
        raise ConfigError(f"{name} 必须大于 0，当前值：{value}")
    return value


def _non_negative_number(
    name: str,
    default: _Number,
    *,
    what: str = "秒数（数字）",
    parse: Callable[[str], _Number],
) -> _Number:
    """读一个不小于 0 的数：没设用默认；不是数 / 是负数都抛 ConfigError。

    0 是合法值（整轮超时 = 不限时、重试次数 = 不重试、退避 = 不等待），
    所以不能复用 _positive_number。
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = parse(raw)
    except ValueError:
        raise ConfigError(f"{name} 必须是{what}，当前值：{raw!r}") from None
    if value < 0:
        raise ConfigError(
            f"{name} 不能是负数，当前值：{value:g}（0 是合法值）"
        )
    return value


def load_api_config() -> ApiConfig:
    """HTTP 服务的会话与安全配置：从环境变量读，全部有合理默认。

    - API_SESSION_TTL：会话过期秒数，默认 3600（正数）；
    - API_MAX_SESSIONS：会话上限，默认 32（正整数），超了丢最久没用的；
    - API_CORS_ORIGINS：允许的前端 Origin，逗号分隔，默认空（一个都不放
      行，浏览器前端接上来之前先配它）；**不接受 `*`**——放行整个 Web
      等于把本机服务交给任何网页，宁可让人一项项写清楚；
    - API_TOKEN：设了就必须带 `Authorization: Bearer <token>`；默认无，
      本机开发直接可用；
    - AGENT_RUN_TIMEOUT：一次 Run 的整轮预算（秒），默认 0 = 不限时。
      设了正数就到点按「运行超时」停下（协作式，见 agent/cancel.py）——
      和单个工具自己的 timeout 是两回事，那个照旧生效。
    """
    ttl = _positive_number(
        API_SESSION_TTL_ENV, DEFAULT_SESSION_TTL, what="数字", parse=float
    )
    sessions = _positive_number(
        API_MAX_SESSIONS_ENV, DEFAULT_MAX_SESSIONS, what="整数", parse=int
    )
    run_timeout = _non_negative_number(
        AGENT_RUN_TIMEOUT_ENV, DEFAULT_RUN_TIMEOUT, parse=float
    )

    raw_origins = os.getenv(API_CORS_ORIGINS_ENV, "").strip()
    origins = tuple(
        dict.fromkeys(
            item.strip().rstrip("/")
            for item in raw_origins.split(",")
            if item.strip()
        )
    )
    if "*" in origins:
        raise ConfigError(
            f"{API_CORS_ORIGINS_ENV} 不能用 *：请写明确的前端 Origin，"
            f"例如 http://localhost:5173"
        )

    return ApiConfig(
        session_ttl=ttl,
        max_sessions=sessions,
        cors_origins=origins,
        token=os.getenv(API_TOKEN_ENV, "").strip() or None,
        run_timeout=run_timeout,
    )


def load_llm_config() -> LLMConfig:
    """从环境变量构建 LLM 配置；缺失必需项时抛 ConfigError。

    - OPENAI_TIMEOUT：单次请求的超时秒数，默认 60（数字）；
    - LLM_MAX_RETRIES：临时错误（429 / 5xx / 连接超时）最多重试几次，
      默认 2（不小于 0 的整数，0 = 一次就抛）；
    - LLM_RETRY_BACKOFF：每次重试前的基础退避秒数，默认 0.5（不小于 0
      的数字），退避按指数涨：backoff、2×backoff……0 = 不等待。
    """
    timeout_raw = os.getenv("OPENAI_TIMEOUT", "60").strip()
    try:
        timeout = float(timeout_raw)
    except ValueError:
        raise ConfigError(
            f"OPENAI_TIMEOUT 必须是数字，当前值：{timeout_raw!r}"
        ) from None

    return LLMConfig(
        api_key=_require("OPENAI_API_KEY"),
        model=_require("OPENAI_MODEL"),
        # 留空表示使用 OpenAI 官方地址（由 SDK 决定）。
        base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
        timeout=timeout,
        max_retries=_non_negative_number(
            LLM_MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES, what="整数", parse=int
        ),
        retry_backoff=_non_negative_number(
            LLM_RETRY_BACKOFF_ENV, DEFAULT_RETRY_BACKOFF, parse=float
        ),
    )
