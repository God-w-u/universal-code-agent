"""跨进程的向量缓存：一个命名空间一个 JSON 文件（9.3）。

scorer 的内存缓存只活在一次进程里；每开一次新进程，两千多个块又要从头
嵌一遍。这一层把向量写到磁盘上，让**第二次运行**直接读文件——不引数据库、
不引向量库，就是一个能看、能删、能 grep 的 JSON。

五条规矩：

- **一个命名空间一个文件**。命名空间 = 模型 / provider 的标识
  （`namespace_of`：类名 + 模型名，假货用词表指纹），文件名是命名空间的
  哈希。不同模型自然落在不同文件里；文件头里还记着命名空间和格式版本，
  对不上就整个当空的——**模型一换、格式一改，旧向量自动失效**，不会被
  串用。
- **key 自带失效**。chunk 的 key 是 `id|digest`（`chunk_key`）：文件一改
  digest 就变，旧向量自然查不到，不存在「拿旧向量给新代码打分」。查询的
  key 是 `query:` + 文本哈希（`query_key`）——**不存查询原文**。
- **坏了一点都不许炸**。文件不存在、JSON 坏了、盘满、没权限——读写失败
  一律降级成「没有缓存」，最坏就是回到原来的速度。缓存只影响 embedding
  这一层的快慢，字面打分和 baseline 一个字都不经过它。
- **写盘是整份重写 + 临时文件换名**：半截写坏的文件不会留在原地；真正
  写坏了下次加载也会当空缓存，自愈。整份重写意味着**写一次就不便宜**，
  所以攒着写：批量进（`put_many`）、显式落盘（`flush`），不是每条一写。
- **攒久了要打扫**（9.6）。key 带着 digest，意味着文件一改旧向量就永远
  查不到——同一个 chunk 的旧 digest、旧段号会长期攒在文件里。`prune()`
  按「哪些 key 还算数」加「允许清哪些 chunk」把确认查不到的删掉；查询
  向量永远不动，别的语料的条目也不碰（评估入口里两个语料共用一份文件，
  各扫各的）。
- **整块没了就整个 id 清掉**（13.6）。文件被删、符号被删 / 改名 / 挪了
  行，块连 id 都不在了——`prune()` 那种「留还作数的 key」够不着它们（旧
  条目不在任何 scope 里，会被当成别的语料的东西留着）。`forget_ids()`
  就是干这个的：明确说「这些 id 没了」，把它名下的条目一律删掉。索引
  刷新时由 `EmbeddingScorer.forget()` 调过来。

默认放用户目录下（`default_cache_dir`）——不进项目、不会被代码索引扫到。
不提交任何模型文件或向量数据：这是本机缓存，删了就是慢一点，没有别的。
"""

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from index.chunks import CodeChunk

if TYPE_CHECKING:
    # 只在做类型检查时导入：index.embedding 反过来要用本模块，顶层真 import
    # 会成环
    from index.embedding import Embedder

# 缓存文件的格式版本。结构一变就 +1：旧文件自动当空的。
CACHE_VERSION = 1

# 写盘时保留的小数位：余弦相似度用不到更多精度，文件能小一截。
PRECISION = 6

# 查询向量的 key 前缀（`query:` + 文本哈希，见 query_key：不存原文）。
# chunk 的 key 不带它——prune 靠这个前缀划界：查询向量永远不清，chunk
# 向量才可能被清。
QUERY_PREFIX = "query:"

# 分段 key 结尾的段号后缀（embedding._cache_key 拼上去的 `#s0`、`#s12`）。
_TAIL_SEGMENT = re.compile(r"#s\d+$")


def _chunk_id_of(key: str) -> str:
    """从一个 chunk 向量 key 里拆出 chunk id（prune 匹配 scope 用）。

    key 的形状是 `id|digest`，分段时 digest 后面再加 `#sN`。digest 是
    十六进制、跟在**最后一个** `|` 后面；id（`path:symbol:行区间`）里
    不会有 `|`。先摘掉段号后缀，再按最后一个 `|` 切。拆不出 `|` 就原样
    返回——形状不对的 key 通常也就匹配不上任何 scope，效果是不删。
    """
    base = _TAIL_SEGMENT.sub("", key)
    chunk_id, sep, _digest = base.rpartition("|")
    return chunk_id if sep else key


def namespace_of(embedder: "Embedder") -> str:
    """给 embedder 起一个稳定的命名空间：类名 + 模型的区分信息。

    LocalEmbedder 有 `model_name`（不同模型名 → 不同空间）；FakeEmbedder
    用词表指纹（词表不同、向量就不同，不能共用一个文件）；都没有就只用
    类名。**换了模型而命名空间不变**是这里唯一要防的事——同名同表就是
    同一种向量，串用无妨。
    """
    name = type(embedder).__name__
    model = getattr(embedder, "model_name", None)
    if isinstance(model, str) and model:
        return f"{name}:{model}"
    aliases = getattr(embedder, "aliases", None)
    if aliases:
        fingerprint = hashlib.sha1(repr(aliases).encode("utf-8")).hexdigest()[:12]
        return f"{name}:aliases-{fingerprint}"
    return name


def cache_file(directory: Path, namespace: str) -> Path:
    """命名空间 → 缓存文件路径：名字取哈希，文件名不用塞模型名（太长、
    还有 `/` 这种不能进文件名的字符）。"""
    digest = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:12]
    return Path(directory) / f"vectors-{digest}.json"


def chunk_key(chunk: CodeChunk) -> str:
    """chunk 的缓存 key：id（哪儿）| digest（哪一版）。"""
    return f"{chunk.id}|{chunk.digest}"


def query_key(text: str) -> str:
    """查询的缓存 key：`query:` 前缀 + 文本的哈希。和 chunk key 不会撞。

    **为什么不直接用原文**：这份 JSON 落在用户目录里长期留着，而查询文本
    是「用户问过什么」——任务描述、文件名、路径、甚至粘进去的代码片段。
    拿原文当 key，等于把这些原样抄了一份存进文件，谁打开都看得见（内容
    是明文的，跟向量不一样）。哈希之后一一对应不变：同一句还是同一格，
    没问过的还是落空，命中率和以前完全一样，只是文件里认不出问的是什么。

    `QUERY_PREFIX` 照旧：prune / forget_ids 靠它一眼认出「这是查询向量，
    永远不动」，和向量本身怎么存无关。
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{QUERY_PREFIX}{digest}"


def default_cache_dir() -> Path:
    """默认缓存位置：用户目录下——不进项目、不会被代码索引扫到。"""
    return Path.home() / ".cache" / "universal-code-agent"


class VectorCache:
    """一个 JSON 文件的向量缓存。读写失败只记状态，不往外抛。

    `path` 由调用方给（`cache_file(目录, 命名空间)` 算好）；文件懒加载，
    第一次读的时候才碰磁盘。
    """

    def __init__(self, path: Path, namespace: str) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.load_error: str | None = None
        self.write_error: str | None = None
        self._vectors: dict[str, tuple[float, ...]] | None = None
        self._foreign = False  # 文件属于别的模型 / 别的版本：不读也不写
        self._dirty = False  # 内存里有没落盘的

    @property
    def degraded(self) -> bool:
        """出过读写问题没有（测试和排查看这个，行为不依赖它）。"""
        return self.load_error is not None or self.write_error is not None

    def get(self, key: str) -> tuple[float, ...] | None:
        return self._load().get(key)

    def get_many(self, keys: Iterable[str]) -> dict[str, tuple[float, ...]]:
        """按 key 们取，返回命中的那些（key → 向量）。"""
        vectors = self._load()
        return {key: vectors[key] for key in keys if key in vectors}

    def put_many(self, items: dict[str, tuple[float, ...]], flush: bool = True) -> None:
        """收下一批（key → 向量）。`flush=False` 就先攒在内存里，之后
        `flush()` 一次落盘——整份重写不便宜，别一条一写。"""
        if not items:
            return
        vectors = self._load()
        if self._foreign:
            # 文件是别的模型的：覆盖它等于毁掉人家的缓存，宁可不用
            self.write_error = "缓存文件属于另一个模型或版本，没有覆盖"
            return
        vectors.update(items)
        self._dirty = True
        if flush:
            self.flush()

    def flush(self) -> None:
        """把攒下的写盘（没有攒的、或文件是别人的，就什么都不做）。"""
        if self._dirty and not self._foreign and self._vectors is not None:
            self._flush()

    def prune(
        self, keep: Iterable[str], *, scope: Iterable[str] | None = None
    ) -> int:
        """删掉已经查不到的 chunk 向量，返回删掉的条数（9.6）。

        - `keep`：还作数的**完整 key** 集合（chunk 的，分段时含 `#sN`
          后缀）。不在其中的 chunk 条目就是旧 digest / 旧段号——永远
          查不到了。
        - `scope`：只允许清这些 **chunk id**（`path:symbol:行区间`）名下
          的条目；不给就是「文件里所有 chunk 条目都允许清」。一份缓存
          文件可能被多个语料共用（评估入口里项目 + C++ fixture 就是），
          各自只知道自己的现状——`scope` 就是那条边界：A 语料打扫时
          不许碰 B 语料的条目。
        - 查询向量（`QUERY_PREFIX` 开头）**永远不动**：chunk 的旧 digest
          不会再被问到，查询文本将来还可能再问。

        真删掉了才写盘（没删的调用不碰文件）。失败按缓存的老规矩只降级
        不炸：文件是别的模型 / JSON 坏了 / 写不进去，都当「没打扫」。
        """
        vectors = self._load()
        if self._foreign or not vectors:
            return 0

        keep_set = set(keep)
        allowed = None if scope is None else set(scope)
        return self._drop(
            key
            for key in vectors
            if not key.startswith(QUERY_PREFIX)
            and (allowed is None or _chunk_id_of(key) in allowed)
            and key not in keep_set
        )

    def forget_ids(self, chunk_ids: Iterable[str]) -> int:
        """整个 chunk id 不作数了：把它名下的条目全部删掉，返回删掉的条数。

        和 prune() 的分工：prune 管「id 还在、但这一版（digest / 段号）过时
        了」——按还作数的完整 key 留；这里管「id 自己没了」——文件被删、符号
        被删 / 改名 / 挪了行，这些块的 key 永远不会再被问到，连 id 都凑不齐
        一份「还作数的 key」去 prune，只能整个 id 一起清。

        查询向量还是永远不动。缓存坏了 / 写不进去照老规矩只降级不炸。
        """
        gone = set(chunk_ids)
        if not gone:
            return 0
        vectors = self._load()
        if self._foreign or not vectors:
            return 0
        return self._drop(
            key
            for key in vectors
            if not key.startswith(QUERY_PREFIX) and _chunk_id_of(key) in gone
        )

    def _drop(self, doomed: Iterable[str]) -> int:
        """删掉这批 key，真删了才写盘。返回删掉的条数。

        doomed 可能是生成器（上面两个调用方都是现推的）：先收成一份名单再
        动手——边挑边删就是「遍历着一个正在变小的字典」。
        """
        vectors = self._load()
        removed = 0
        for key in list(doomed):
            if key in vectors:
                del vectors[key]
                removed += 1
        if removed:
            self._dirty = True
            self.flush()
        return removed

    def _load(self) -> dict[str, tuple[float, ...]]:
        if self._vectors is not None:
            return self._vectors
        self._vectors = {}

        if not self.path.exists():
            return self._vectors

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return self._vectors

        if (
            not isinstance(data, dict)
            or data.get("version") != CACHE_VERSION
            or data.get("namespace") != self.namespace
        ):
            self._foreign = True
            self.load_error = "缓存文件不属于当前模型或格式版本，已忽略"
            return self._vectors

        raw = data.get("vectors")
        # 写盘时记下的向量长度：长度对不上的条目一律丢——单条坏不连累整份，
        # 也挡住「文件被手改过、混进了别的模型的向量」这种串用。
        dimension = data.get("dimension")
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    vector = tuple(float(x) for x in value)
                except (TypeError, ValueError):
                    continue  # 单条坏就丢单条，不连累整份
                if isinstance(dimension, int) and len(vector) != dimension:
                    continue
                self._vectors[key] = vector
        return self._vectors

    def _flush(self) -> None:
        assert self._vectors is not None
        payload = {
            "version": CACHE_VERSION,
            "namespace": self.namespace,
            "dimension": (
                len(next(iter(self._vectors.values()))) if self._vectors else None
            ),
            "vectors": {
                key: [round(x, PRECISION) for x in vector]
                for key, vector in self._vectors.items()
            },
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(payload, ensure_ascii=False)
            # 先写临时文件再换名：写到一半断了，原文件还在
            scratch = self.path.with_name(self.path.name + ".tmp")
            scratch.write_text(text, encoding="utf-8")
            scratch.replace(self.path)
            self.write_error = None
            self._dirty = False
        except (OSError, TypeError, ValueError) as exc:
            self.write_error = f"{type(exc).__name__}: {exc}"
