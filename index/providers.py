"""Embedding Provider 工厂：按名字拿一个 Embedder（9.0）。

调用方只给一个名字，拿到的对象一律满足 `Embedder` 接口——
Retriever / Evaluation 不认识名字，也不认识模型：

    embedder = create_embedder("fake")    # 假货：离线测试、演示（默认）
    embedder = create_embedder("local")   # 本地模型（可选依赖，见 local_embedding）
    retriever = Retriever(scorers=hybrid_scorers(embedder))

现在是三个 provider：

- `"fake"`：FakeEmbedder 加演示用小抄（8.6 那张表）。默认值就是它——
  真模型要装可选依赖，默认不该是个一用就报错的选项。
- `"local"`：LocalEmbedder，sentence-transformers 的封装。构造时不加载
  模型，没装库的机器上也能拿到对象，第一次 embed 才报缺依赖。
- `"multilingual"`（9.2）：同样是 LocalEmbedder，默认模型换成多语言
  的 paraphrase-multilingual-MiniLM-L12-v2——中文查询 / 中文注释也有
  语义区分度。**模型名只是这里的默认值**：传了 `model_name` 就用传的，
  调用方（评估入口）只认 provider 名字，不认具体模型。

**加新 provider（比如将来接 API）就两件事**：写一个 Embedder 子类，
往 `PROVIDERS` 里加一行「名字 → 构造函数」。别的都不动——这就是
「可插拔」的意思。
"""

from collections.abc import Callable, Sequence

from index.embedding import DEMO_ALIASES, Embedder, FakeEmbedder
from index.local_embedding import (
    DEFAULT_LOCAL_MAX_TOKENS,
    DEFAULT_MULTILINGUAL_MAX_TOKENS,
    DEFAULT_MULTILINGUAL_MODEL,
    LocalEmbedder,
)
from tools.base import ToolError


def _fake(aliases: Sequence[Sequence[str]] = DEMO_ALIASES) -> Embedder:
    """默认给演示用小抄；测试可以传自己的词表。"""
    return FakeEmbedder(aliases)


def _local(**kwargs) -> Embedder:
    """参数原样转给 LocalEmbedder（model_name / model_loader / batch_size）。

    顺手声明窗口（9.5）：分段要知道模型的 max_seq_length，而它得是个
    不加载模型就能拿到的常量——就放在 local_embedding 里。换模型名自己传
    `max_tokens=`。
    """
    kwargs.setdefault("max_tokens", DEFAULT_LOCAL_MAX_TOKENS)
    return LocalEmbedder(**kwargs)


def _multilingual(**kwargs) -> Embedder:
    """多语言模型（9.2）：默认 paraphrase-multilingual-MiniLM-L12-v2。

    和 `_local` 是同一种东西，只是默认模型不同——传了 model_name 就用
    传的，换别的多语言模型不用改这里（窗口大小记得一起传）。
    """
    kwargs.setdefault("model_name", DEFAULT_MULTILINGUAL_MODEL)
    kwargs.setdefault("max_tokens", DEFAULT_MULTILINGUAL_MAX_TOKENS)
    return LocalEmbedder(**kwargs)


# 名字 → 构造函数。加 provider 从这里加。
PROVIDERS: dict[str, Callable[..., Embedder]] = {
    "fake": _fake,
    "local": _local,
    "multilingual": _multilingual,
}


def create_embedder(provider: str = "fake", **kwargs) -> Embedder:
    """按名字造一个 Embedder，多余的关键字参数原样转给那个 provider。

    名字不认识就报错并列出可选项——拼错了却悄悄给个默认的，是最难查的
    那种错。
    """
    try:
        factory = PROVIDERS[provider]
    except KeyError:
        known = "、".join(sorted(PROVIDERS))
        raise ToolError(
            f"不认识的 embedding provider：{provider!r}（可选：{known}）"
        ) from None
    return factory(**kwargs)
