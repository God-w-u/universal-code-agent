"""本地 embedding 模型：sentence-transformers 的封装（9.0）。

真模型接进来的第一个实现。三条规矩：

- **模型细节不出这个文件。** Retriever / Scorer 只认识 `Embedder` 接口
  （`dimension` + `embed`）；这儿对模型的要求也只是一组方法——
  `encode(texts, batch_size=...)` 和量维度：**新版本叫
  `get_embedding_dimension()`，老版本叫 `get_sentence_embedding_dimension()`**
  （旧名字在新版会发 FutureWarning，9.6 起优先用新名字、旧的兜底）。
  测试塞一个假模型（就这两个方法）就能把封装逻辑全测了，不用装库、
  不用联网。
- **加载是懒的。** 构造函数不碰模型、不联网——没装库的机器上
  `create_embedder("local")` 照样拿得到对象，报错推迟到第一次真要向量
  的时候，且一律包成 `EmbeddingUnavailable`，附上怎么办（装库；第一次
  用模型要从 HuggingFace 下载，得联网）。
- **空输入不加载模型。** 一条文本都没给，直接回空——为这个去拉起一个
  几百 MB 的模型是白花钱。

sentence-transformers 是**可选依赖**（它拖着 torch，不小），没进
requirements.txt；要用先 `pip install sentence-transformers`。

不碰检索：Retriever、打分、排序一个字没改；这一层只是又一个 Embedder。
"""

from collections.abc import Callable, Sequence
from typing import Any

from index.embedding import Embedder, Vector
from tools.base import ToolError

# 默认模型：sentence-transformers 里最小的常用模型之一（384 维），
# 第一次用会从 HuggingFace 下载。
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"

# 多语言模型（9.2）：中文查询和满篇中文注释的项目也有语义区分度。
# 12 层，比上面的 6 层慢一倍左右，向量同样 384 维；第一次用同样要下载。
DEFAULT_MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 上面两个模型的输入窗口（sentence-transformers 里的 max_seq_length）。
# 9.5 的分段（index/segments.py）拿它当预算。**常量而不是问模型要**：
# 查 max_seq_length 得先加载模型，而缓存 key 由切分结果决定、必须能在
# 无模型时算出来（见 segments.py 开头）。换模型记得连它一起换——
# providers 把它们当默认值传给 LocalEmbedder。
DEFAULT_LOCAL_MAX_TOKENS = 256
DEFAULT_MULTILINGUAL_MAX_TOKENS = 128

# 一次 encode 塞多少条。模型内部也分批算，这个数只影响峰值内存。
DEFAULT_BATCH_SIZE = 32


def _read_dimension(model: Any) -> int:
    """量模型的输出维度：优先新 API，旧名字兜底（9.6）。

    sentence-transformers 把 `get_sentence_embedding_dimension()` 改名成
    `get_embedding_dimension()`，旧名字还在但会发 FutureWarning。先看新
    名字在不在，不在才用旧的——老版本（和测试里的假模型）都只有旧名字。
    两个都没有就是模型接错了（不是 sentence-transformers 模型），报清楚。
    """
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, name, None)
        if callable(getter):
            return int(getter())
    raise ToolError(
        "模型既没有 get_embedding_dimension()，也没有 "
        "get_sentence_embedding_dimension()，量不出向量维度"
    )


class EmbeddingUnavailable(RuntimeError):
    """本地 embedding 跑不起来：没装库 / 模型没下载 / 加载失败。

    单独一个类型，和「参数传错了」（ToolError）分开：它要提示的是
    「装依赖、检查网络」，不是「改调用」。
    """


def load_sentence_transformer(model_name: str) -> Any:
    """默认加载器：import sentence-transformers，把模型拉起来。

    第一次用会从 HuggingFace 下载模型，要联网。两条失败路分开说：
    没装库告诉怎么装；装了拉不动（多半卡在下载）把原因带上。
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingUnavailable(
            "没装 sentence-transformers，用不了本地 embedding 模型："
            "pip install sentence-transformers"
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # 第三方库的错五花八门，统一包成能行动的提示
        raise EmbeddingUnavailable(
            f"加载本地模型 {model_name!r} 失败（第一次用要从 HuggingFace "
            f"下载，得联网）：{exc}"
        ) from exc


class LocalEmbedder(Embedder):
    """本地模型 embedding：把一个 sentence-transformers 模型包成 Embedder。

    构造不加载模型；第一次 `embed()`（或问 `dimension`）才加载，之后一直
    用同一个。换模型改 `model_name`；以后接 API 就再写一个 Embedder
    子类——Retriever 认的是接口，不是这里。

    `model_loader` 是测试口子：默认走上面的 `load_sentence_transformer`，
    测试传一个返回假模型的东西，就能离线把封装逻辑测干净。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_MODEL,
        *,
        model_loader: Callable[[str], Any] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int | None = None,
    ) -> None:
        if batch_size < 1:
            raise ToolError(f"batch_size 至少为 1，收到 {batch_size}")
        if max_tokens is not None and max_tokens < 1:
            raise ToolError(f"max_tokens 至少为 1，收到 {max_tokens}")
        self.model_name = model_name
        self.batch_size = batch_size
        self._max_tokens = max_tokens
        self._loader = (
            load_sentence_transformer if model_loader is None else model_loader
        )
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def max_tokens(self) -> int | None:
        """模型窗口（构造时声明）；None 表示没声明、不切分（9.5）。

        刻意不向模型要这个数——见 embedding.Embedder.max_tokens。providers
        按模型名给默认值（DEFAULT_LOCAL_MAX_TOKENS / ..._MULTILINGUAL_...）；
        换模型名时自己传一个。
        """
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """真实分词数（**诊断用**，会加载模型；切分估计不走这里）。

        和 segments.estimate_tokens 的分工：那个只求「不小于真实值」、
        不碰模型，用来决定怎么切；这个给评估入口量「切完还超不超窗」。
        模型没带 tokenizer 就报错，由调用方决定跳不跳这一项诊断。
        """
        model = self._load_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ToolError(f"模型 {self.model_name!r} 没有 tokenizer，数不了 token")
        return len(tokenizer.encode(text))

    @property
    def loaded(self) -> bool:
        """模型加载了没有。给测试和排查看，行为不依赖它。"""
        return self._model is not None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        # _load_model 末尾一定会把它填上；写出来是给读代码的人和类型检查看的
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        """批量进、批量出，顺序一一对应。空批次直接回空，不惊动模型。"""
        batch = tuple(texts)
        if not batch:
            return ()

        model = self._load_model()
        raw = model.encode(list(batch), batch_size=self.batch_size)
        vectors = tuple(tuple(float(value) for value in vector) for vector in raw)
        if len(vectors) != len(batch):
            # 条数对不上就没法把向量和代码块对上：上游只会拿它去算相似度，
            # 多一条少一条都会静默错位，这里趁早报出来
            raise ToolError(
                f"模型收到 {len(batch)} 条文本，却回了 {len(vectors)} 个向量"
            )
        return vectors

    def _load_model(self) -> Any:
        """加载模型（只一次），顺手把它的维度记下来。失败一律可行动地报。"""
        if self._model is not None:
            return self._model

        try:
            model = self._loader(self.model_name)
        except EmbeddingUnavailable:
            # 自定义 loader 自己就说清楚了，原样冒出
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"加载本地模型 {self.model_name!r} 失败：{exc}"
            ) from exc

        self._model = model
        self._dimension = _read_dimension(model)
        return self._model
