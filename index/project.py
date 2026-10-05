"""Project Index：把「项目里有哪些文件」列成一份结构化清单。

这是代码检索这条线的第一步，只做一件事：**看清项目里有什么**。每个文件
记一条 IndexedFile——相对路径、语言、状态、大小、修改时间、内容摘要。
它不是给模型看的文本（那是 list_files 的活），而是一份数据：后面的
Symbol Index（每个文件里有哪些类 / 函数）、Code Chunk（按结构切块）、
Hybrid Search 和 Code RAG 都从这份清单上接着做。

所以本阶段**不存文件内容**：切块的时候再按路径去读，读到的还是不是当初
索引的那份，由 `digest`（内容 sha256 的前 16 位）回答。没有 embedding、
没有向量库、没有语义搜索——那些是后面阶段的事。

几条刻意的边界：

- **只扫 root 以内**：root 走 tools/paths.py 的 check_root（磁盘根、
  系统目录、不存在的目录都在这儿被挡），目录遍历走 tools/tree.py 的
  iter_files——和 list_files / search_code 是同一个函数。所以「跳过 .git、
  node_modules 这些目录」只有一份实现，索引不另立一套，也不可能列出
  list_files 看不到的文件。
- **只读，不写**：不建文件、不改文件，扫多少次项目都是原样。
- **一个坏文件不拖垮整次扫描**：读不动的记一条 UNREADABLE 接着扫下一个，
  和 search_code「跳过坏文件继续搜」是同一条规矩。
- **全量、可重跑**：build() 每次从头扫一遍，再调一次就是刷新，不留旧账。
  本阶段不做增量——但每条记录上的 size / mtime / digest 就是以后做增量
  要用的东西，现在就得在。
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from index.languages import language_of
from tools.base import ToolError
from tools.paths import check_root
from tools.textfile import decode_text, read_bytes
from tools.tree import iter_files

# 文件的索引状态。英文短 id：这是给后面的阶段和 CLI 汇总用的数据，
# 不是给模型看的文本（要展示的时候由调用方翻译）。
INDEXED = "indexed"  # 文本文件，已经收进索引
BINARY = "binary"  # 二进制，或两种编码都解不开（UTF-16 也会落在这儿）
TOO_LARGE = "too_large"  # 超过 max_file_bytes，太大了不收
UNREADABLE = "unreadable"  # 打不开（被占用、没权限、扫到一半被删了）

# 内容摘要取前多少位十六进制。sha256 全量 64 位太长，16 位（64 bit）
# 在一个项目里撞车的概率约等于零，够用了。
DIGEST_CHARS = 16


@dataclass(frozen=True)
class IndexedFile:
    """索引里的一个文件——一条记录，全是算好的值，不指向别处。

    `size` / `mtime` 在打不开（UNREADABLE）时是 None：stat 都没成功，
    这两个数就无从谈起。`digest` 只有 INDEXED 才有：跳过的文件反正不会
    被切块、被检索，给它算摘要没有意义。
    """

    path: str  # 相对项目根的路径，/ 分隔（和 list_files 一致）
    language: str  # 语言 id；认不出来是 index.languages.UNKNOWN_LANGUAGE
    status: str  # INDEXED / BINARY / TOO_LARGE / UNREADABLE
    size: int | None  # 字节数
    mtime: float | None  # 修改时间（st_mtime 原值，比较用，不格式化）
    digest: str | None  # 内容 sha256 的前 DIGEST_CHARS 位

    @property
    def indexed(self) -> bool:
        """收进索引了吗——只有它是 True 的文件，后面的阶段才该去读它。"""
        return self.status == INDEXED


class ProjectIndex:
    """项目文件索引。用法：建一个 → build() → 读 entries。

        index = ProjectIndex()      # 默认整个项目
        index.build()
        index.entries               # 全部记录（含没收进来的）
        index.files("python")       # 只要已索引的 Python 文件

    文件改过、加过、删过之后想刷新，再调一次 build() 就是全量重建；显式的
    话说 `refresh()`，两件事一模一样（这一层是全量的，没有可省的部分）。

    `generation` 记着这份清单换过几版：上面的符号 / 切块 / 调用图索引拿它
    判断「我手上这份下游数据还是不是对着最新的清单」。

    root 不合法（磁盘根、系统目录、不存在）在构造时就抛 ToolError——
    牢笼得在开始扫之前就成立。
    """

    # 单个文件最大多少字节，比这大的只记一条 TOO_LARGE、不读内容。
    # 源码很少有这么大，真有（压缩过的 JS 之类）也不适合进检索。
    max_file_bytes = 1024 * 1024

    def __init__(self, root: Path | None = None) -> None:
        self.root = check_root(root)
        self._entries: dict[str, IndexedFile] = {}
        self._built = False
        self._generation = 0

    def build(self) -> None:
        """全量扫描 root，重建索引。再调一次就是「刷新」。"""
        try:
            files = sorted(iter_files(self.root))
        except OSError as exc:
            # 整个目录进不去（权限、被占用）是整次扫描失败，不是某个文件的事。
            # 转成 ToolError，和 list_files / search_code 的报法一致。
            raise ToolError(f"读取目录失败：{exc}") from exc

        entries: dict[str, IndexedFile] = {}
        for item in files:
            entry = self._examine(item)
            entries[entry.path] = entry  # 按路径当键：一个文件只可能有一条

        # 扫完才换上新的：中途万一出了意外，手上这份旧索引还是完整的
        self._entries = entries
        self._built = True
        self._generation += 1

    def refresh(self) -> None:
        """显式刷新：全量重扫一遍（文件增、删、改之后调它）。

        和再调一次 `build()` 是同一件事——这一层本来就是全量的。留这个名字
        是为了让「刷新」在调用方读起来是一个动作，而不是猜「再 build 一次
        算不算刷新」。
        """
        self.build()

    @property
    def generation(self) -> int:
        """这份清单换过几版：0 是「一版都没有」，每扫成一次 +1。

        上面的索引拿它判断自己是不是对着旧清单建的——调用方能拿同一份
        ProjectIndex 去问别人重扫了没有（index/symbols.py、callgraph）。
        """
        return self._generation

    @property
    def built(self) -> bool:
        """扫过没有。

        从没扫过的索引是**空的**，但那不等于「项目里没文件」——两件事得分得开，
        不然拿一份没扫的清单去建符号索引，会静默地一个符号都拿不到。
        """
        return self._built

    @property
    def entries(self) -> tuple[IndexedFile, ...]:
        """全部记录，按路径排序。二进制、超大这些没收进来的也在里面。"""
        return tuple(self._entries.values())

    def files(self, language: str | None = None) -> tuple[IndexedFile, ...]:
        """已收进索引的文件；给 language 就只要那一种语言（id 小写）。"""
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.indexed and (language is None or entry.language == language)
        )

    # ------------------------------------------------------------------
    # 看一个文件
    # ------------------------------------------------------------------

    def _examine(self, item: Path) -> IndexedFile:
        """给一个文件定性：什么语言、多大、收不收得进来。绝不往外抛异常。

        顺序是有意的：先 stat（便宜，还能顺手挡掉超大文件），再读内容。
        """
        relative = item.relative_to(self.root).as_posix()
        language = language_of(item)

        try:
            info = item.stat()
        except OSError:
            # 列出来之后、看它之前就没了：记一条，接着扫下一个
            return IndexedFile(
                path=relative,
                language=language,
                status=UNREADABLE,
                size=None,
                mtime=None,
                digest=None,
            )

        if info.st_size > self.max_file_bytes:
            return IndexedFile(
                path=relative,
                language=language,
                status=TOO_LARGE,
                size=info.st_size,
                mtime=info.st_mtime,
                digest=None,
            )

        try:
            raw, too_big = read_bytes(item, self.max_file_bytes)
        except OSError:
            return IndexedFile(
                path=relative,
                language=language,
                status=UNREADABLE,
                size=info.st_size,
                mtime=info.st_mtime,
                digest=None,
            )

        if too_big:
            # 刚 stat 完又长大了（有人正在写这个文件）：按超大处理，不猜
            return IndexedFile(
                path=relative,
                language=language,
                status=TOO_LARGE,
                size=info.st_size,
                mtime=info.st_mtime,
                digest=None,
            )

        # 是文本吗——用 read_file / search_code 同一套解码规则问。UTF-16
        # 这类带 NUL 的"看着像文本"的文件，也会在这儿被归成二进制。
        # 解出来的文字**不留**在索引里，这里只借它做这一个判断。
        try:
            decode_text(raw, relative)
        except ToolError:
            return IndexedFile(
                path=relative,
                language=language,
                status=BINARY,
                size=info.st_size,
                mtime=info.st_mtime,
                digest=None,
            )

        return IndexedFile(
            path=relative,
            language=language,
            status=INDEXED,
            size=info.st_size,
            mtime=info.st_mtime,
            digest=digest_of(raw),
        )


def digest_of(raw: bytes) -> str:
    """内容摘要：sha256 前 DIGEST_CHARS 位。**「同一个文件」只有这一个定义。**

    回答的是「这个文件还是上次索引的那份吗」。光看 mtime 不稳（有些文件
    系统的时间戳只精确到秒，同一秒里改两次看不出来），所以内容摘要也得有。
    建索引时算它（本模块），符号索引核对「文件被改过没有」时算的还是它
    （index/symbols.py）——同一个文件在两处算出来的摘要必须一样。
    """
    return hashlib.sha256(raw).hexdigest()[:DIGEST_CHARS]
