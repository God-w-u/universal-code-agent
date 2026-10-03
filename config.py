"""项目配置。

所有敏感信息（API Key 等）都从环境变量读取，不硬编码在代码里。
系统环境变量优先；若项目根目录存在 .env 文件，则作为补充来源。
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# override=False：已存在的系统环境变量优先，不被 .env 覆盖。
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """配置缺失或不合法。"""


@dataclass(frozen=True)
class LLMConfig:
    """调用 LLM 所需的配置。"""

    api_key: str
    model: str
    base_url: str | None = None
    timeout: float = 60.0
    temperature: float = 0.0


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


def load_llm_config() -> LLMConfig:
    """从环境变量构建 LLM 配置；缺失必需项时抛 ConfigError。"""
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
    )
