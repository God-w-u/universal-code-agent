"""索引刷新（13.6）：显式 refresh、文件增删改、调用图失效、向量不留旧账。

四条要钉的东西：

1. **刷新拿得到最新数据**：文件加了 / 删了 / 改了，refresh() 之后索引里就是
   当前的样子——全量重建，旧数据一条不留（不做增量，也不做后台监听）。
2. **刷新是点名要的**：默认的 build() 一个字节不改——调用方传来的清单 /
   符号索引照旧不碰（digest 保护那条老规矩就是这么生效的）；refresh() 才
   一路级联，连调用方的东西也重扫。
3. **调用图认得版本**：底下的符号索引一刷新，建在它上面的 CallGraphIndex
   立刻失效（built 变回 False），下一次查询重建——不拿旧行号回答；工具层
   的 refresh() 是同一件事的手动入口。
4. **向量不返回过期数据**：删掉的块，它名下的向量从内存和磁盘缓存一起清掉
   （不是「留着反正查不到」）；改过内容的块 digest 变了，旧 key 查不到，
   刷新后预热时顺手打扫。查询向量永远不动。

全部离线：embedding 用 FakeEmbedder，缓存写在临时目录里，真模型不进测试。
"""

import json
import tempfile
import threading
import unittest
from pathlib import Path

from callgraph import CallGraphIndex
from index import (
    DEMO_ALIASES,
    ChunkIndex,
    FakeEmbedder,
    ProjectIndex,
    Retriever,
    SymbolIndex,
    VectorCache,
    cache_file,
)
from tools import CallGraphTool, SearchChunksTool

# 源码按行写，断言里的行号就是这里的行号。标识符故意起得没意义：
# 这个文件自己也会被索引，用例名和符号名越不相干越不会互相干扰。
ALPHA = "\n".join(
    [
        "def alpha():",  # 1
        '    """壹号。"""',  # 2
        "    return 1",  # 3
        "",  # 4
        "",  # 5
        "def beta():",  # 6
        '    """贰号。"""',  # 7
        "    return 2",  # 8
    ]
) + "\n"

# 同一个文件的另一版：beta 的正文改了内容，行数没变 → chunk id 不变、digest 变
ALPHA_EDITED = ALPHA.replace("    return 2", "    return 200")

# beta 开始调用 alpha（调用图上多一条边，行数还是没变）
ALPHA_CALLING = ALPHA.replace("    return 2", "    return alpha()")

GAMMA = "\n".join(
    [
        "def gamma():",  # 1
        '    """叁号。"""',  # 2
        "    return 3",  # 3
    ]
) + "\n"


class RecordingEmbedder:
    """包一层，记下每一批收到的文本（看「有没有重新嵌」用）。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.batches: list[tuple[str, ...]] = []

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    def embed(self, texts):
        batch = tuple(texts)
        self.batches.append(batch)
        return self.inner.embed(batch)

    def seen(self) -> str:
        """见过所有批次拼起来的全文（找一段文本嵌过没有）。"""
        return "\n".join("\n".join(batch) for batch in self.batches)


class RefreshTestBase(unittest.TestCase):
    """一个临时项目 + 一个独立的临时缓存目录。"""

    def setUp(self):
        project = tempfile.TemporaryDirectory()
        self.addCleanup(project.cleanup)
        self.root = Path(project.name).resolve()

        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        self.cache_dir = Path(cache.name).resolve()

        self.make("pkg/alpha.py", ALPHA)

    def make(self, relative: str, content: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return path

    def remove(self, relative: str) -> None:
        (self.root / relative).unlink()

    def symbols(self) -> SymbolIndex:
        index = SymbolIndex(root=self.root)
        index.build()
        return index

    def chunks(self) -> ChunkIndex:
        index = ChunkIndex(root=self.root)
        index.build()
        return index

    def graph(self) -> CallGraphIndex:
        index = CallGraphIndex(root=self.root)
        index.build()
        return index

    def names(self, index: SymbolIndex) -> list[str]:
        return [item.qualified for item in index.symbols()]

    def chunk_ids(self, index: ChunkIndex) -> list[str]:
        return [item.id for item in index.chunks()]


# ---------------------------------------------------------------------------
# Project Index：刷新拿得到增删改
# ---------------------------------------------------------------------------


class ProjectRefreshTest(RefreshTestBase):
    def test_a_refresh_sees_an_added_file(self):
        index = ProjectIndex(root=self.root)
        index.build()

        self.make("pkg/gamma.py", GAMMA)
        index.refresh()

        self.assertEqual(
            [entry.path for entry in index.files()],
            ["pkg/alpha.py", "pkg/gamma.py"],
        )

    def test_a_refresh_drops_a_deleted_file(self):
        self.make("pkg/gamma.py", GAMMA)
        index = ProjectIndex(root=self.root)
        index.build()

        self.remove("pkg/alpha.py")
        index.refresh()

        self.assertEqual([entry.path for entry in index.files()], ["pkg/gamma.py"])

    def test_a_refresh_sees_the_new_digest(self):
        index = ProjectIndex(root=self.root)
        index.build()
        before = index.files()[0].digest

        self.make("pkg/alpha.py", ALPHA_EDITED)
        index.refresh()

        self.assertNotEqual(index.files()[0].digest, before)

    def test_every_scan_bumps_the_generation(self):
        index = ProjectIndex(root=self.root)

        self.assertEqual(index.generation, 0)  # 一版都没有
        index.build()
        self.assertEqual(index.generation, 1)
        index.refresh()
        self.assertEqual(index.generation, 2)
        index.build()  # build 也是刷新，一样算一版
        self.assertEqual(index.generation, 3)


# ---------------------------------------------------------------------------
# Symbol Index：刷新级联到文件清单
# ---------------------------------------------------------------------------


class SymbolRefreshTest(RefreshTestBase):
    def test_refresh_finds_a_symbol_in_a_new_file(self):
        index = self.symbols()

        self.make("pkg/gamma.py", GAMMA)
        index.refresh()

        self.assertEqual(self.names(index), ["alpha", "beta", "gamma"])

    def test_refresh_drops_the_symbols_of_a_deleted_file(self):
        self.make("pkg/gamma.py", GAMMA)
        index = self.symbols()

        self.remove("pkg/gamma.py")
        index.refresh()

        self.assertEqual(self.names(index), ["alpha", "beta"])
        self.assertEqual(index.skipped, ())

    def test_a_caller_managed_project_is_still_left_alone_by_build(self):
        """老规矩一个字不改：build() 不碰调用方给的清单，refresh() 才碰。"""
        project = ProjectIndex(root=self.root)
        project.build()
        index = SymbolIndex(project)

        self.make("pkg/gamma.py", GAMMA)
        index.build()
        self.assertEqual(self.names(index), ["alpha", "beta"])  # 没扫新文件

        index.refresh()
        self.assertEqual(self.names(index), ["alpha", "beta", "gamma"])

    def test_refresh_bumps_the_generation(self):
        index = self.symbols()

        before = index.generation
        index.refresh()

        self.assertEqual(index.generation, before + 1)


# ---------------------------------------------------------------------------
# Chunk Index：行号、删除、级联
# ---------------------------------------------------------------------------


class ChunkRefreshTest(RefreshTestBase):
    def test_a_refresh_uses_the_new_line_numbers(self):
        index = self.chunks()
        before = index.chunks(symbol="beta")[0]
        self.assertEqual((before.start_line, before.end_line), (6, 8))

        # 前面插两行：beta 整体下移，chunk 的行号必须跟着走
        self.make("pkg/alpha.py", "# 注释\n# 又一行\n" + ALPHA)
        index.refresh()

        after = index.chunks(symbol="beta")[0]
        self.assertEqual((after.start_line, after.end_line), (8, 10))
        self.assertEqual(after.code, 'def beta():\n    """贰号。"""\n    return 2')

    def test_an_added_file_shows_up_as_chunks(self):
        index = self.chunks()

        self.make("pkg/gamma.py", GAMMA)
        index.refresh()

        self.assertEqual(
            self.chunk_ids(index),
            [
                "pkg/alpha.py:alpha:1-3",
                "pkg/alpha.py:beta:6-8",
                "pkg/gamma.py:gamma:1-3",
            ],
        )

    def test_a_deleted_file_leaves_no_chunks(self):
        self.make("pkg/gamma.py", GAMMA)
        index = self.chunks()

        self.remove("pkg/gamma.py")
        index.refresh()

        self.assertEqual(
            self.chunk_ids(index), ["pkg/alpha.py:alpha:1-3", "pkg/alpha.py:beta:6-8"]
        )

    def test_refresh_cascades_through_a_caller_managed_symbol_index(self):
        project = ProjectIndex(root=self.root)
        project.build()
        symbols = SymbolIndex(project)
        symbols.build()
        index = ChunkIndex(symbols=symbols)

        self.make("pkg/gamma.py", GAMMA)
        index.build()
        self.assertEqual(index.chunks(path="pkg/gamma.py"), ())  # 没动人家的

        index.refresh()
        self.assertEqual(
            [item.symbol for item in index.chunks(path="pkg/gamma.py")], ["gamma"]
        )
        self.assertIs(index.symbols, symbols)  # 还是那份，只是被重扫过了

    def test_a_search_after_refresh_finds_the_new_symbol(self):
        index = self.chunks()
        retriever = Retriever(index=index)
        self.assertEqual(retriever.search("gamma"), ())

        self.make("pkg/gamma.py", GAMMA)
        index.refresh()

        found = retriever.search("gamma")
        self.assertEqual(found[0].chunk.symbol, "gamma")


# ---------------------------------------------------------------------------
# Call Graph：符号索引一刷新，图就失效
# ---------------------------------------------------------------------------


class CallGraphRefreshTest(RefreshTestBase):
    def test_a_symbol_refresh_invalidates_the_graph(self):
        index = self.graph()
        self.assertTrue(index.built)
        self.assertEqual(index.callees("beta"), ())

        self.make("pkg/alpha.py", ALPHA_CALLING)
        index.symbols.refresh()

        # 旧图上的东西还躺在对象里，但 built 变回 False：调用方会重建
        self.assertFalse(index.built)
        index.build()
        self.assertTrue(index.built)
        self.assertEqual(index.callees("beta"), ("alpha",))

    def test_an_own_refresh_rebuilds_on_the_new_code(self):
        index = self.graph()

        self.make("pkg/alpha.py", ALPHA_CALLING)
        index.refresh()

        self.assertTrue(index.built)
        self.assertEqual(index.callees("beta"), ("alpha",))

    def test_a_caller_managed_symbol_index_is_rebuilt_from(self):
        symbols = SymbolIndex(root=self.root)
        symbols.build()
        index = CallGraphIndex(symbols=symbols)
        index.build()
        self.assertTrue(index.built)

        self.make("pkg/alpha.py", ALPHA_CALLING)
        symbols.refresh()  # 调用方自己刷的新

        self.assertFalse(index.built)
        index.build()
        self.assertIs(index.symbols, symbols)  # 用回调用方那份（已经不旧了）
        self.assertEqual(index.callees("beta"), ("alpha",))

    def test_a_deleted_file_takes_its_edges_away(self):
        self.make("pkg/gamma.py", "def gamma():\n    return alpha()\n")
        index = self.graph()
        self.assertEqual(index.callees("gamma"), ("alpha",))

        self.remove("pkg/gamma.py")
        index.refresh()

        self.assertEqual(index.edges(path="pkg/gamma.py"), ())

    def test_the_generation_follows_the_builds(self):
        index = self.graph()

        self.assertGreater(index.generation, 0)
        before = index.generation
        index.refresh()

        self.assertEqual(index.generation, before + 1)


# ---------------------------------------------------------------------------
# 工具层：search_code_chunks 的 refresh
# ---------------------------------------------------------------------------


class SearchChunksRefreshTest(RefreshTestBase):
    """off 模式（没有模型）：刷新只跟索引有关。"""

    def tool(self) -> SearchChunksTool:
        return SearchChunksTool(root=self.root)

    def test_a_new_file_becomes_searchable_after_refresh(self):
        tool = self.tool()
        self.assertIn("没有找到", tool.execute("gamma"))

        self.make("pkg/gamma.py", GAMMA)
        tool.refresh()

        self.assertIn("pkg/gamma.py:1-3", tool.execute("gamma"))

    def test_a_deleted_file_is_gone_after_refresh(self):
        self.make("pkg/gamma.py", GAMMA)
        tool = self.tool()
        self.assertIn("pkg/gamma.py", tool.execute("gamma"))

        self.remove("pkg/gamma.py")
        tool.refresh()

        self.assertIn("没有找到", tool.execute("gamma"))

    def test_an_edited_body_is_reflected_after_refresh(self):
        tool = self.tool()
        self.assertIn("pkg/alpha.py:6-8", tool.execute("贰号"))

        self.make("pkg/alpha.py", ALPHA_EDITED)
        tool.refresh()

        # 行号没变、正文变了：还是同一条 chunk（id 一样），正文是新的
        self.assertIn("pkg/alpha.py:6-8", tool.execute("200"))

    def test_refresh_before_the_first_search_is_harmless(self):
        tool = self.tool()

        tool.refresh()  # 从没建过：先来的上一版是空的

        self.assertIn("pkg/alpha.py:1-3", tool.execute("壹号"))

    def test_a_search_and_a_refresh_from_several_threads_do_not_blow_up(self):
        """刷新和查询并发：交替进行不炸，收尾后拿到的是最新的那份索引。"""
        tool = self.tool()
        tool.execute("壹号")  # 先热身一次（索引建好、锁走一遍）
        errors: list[BaseException] = []

        def query() -> None:
            try:
                for _ in range(5):
                    tool.execute("壹号")
            except BaseException as exc:  # 线程里的错要带出来，不然看不见
                errors.append(exc)

        self.make("pkg/gamma.py", GAMMA)
        workers = [threading.Thread(target=query, daemon=True) for _ in range(3)]
        for worker in workers:
            worker.start()
        for _ in range(3):
            tool.refresh()
        for worker in workers:
            worker.join(10)

        self.assertEqual(errors, [])
        self.assertIn("pkg/gamma.py:1-3", tool.execute("gamma"))


class CallGraphToolRefreshTest(RefreshTestBase):
    def test_the_cached_graph_is_used_until_refresh(self):
        tool = CallGraphTool(root=self.root)
        self.assertIn("没有调用任何已知的函数", tool.execute("beta", "callees"))

        self.make("pkg/alpha.py", ALPHA_CALLING)
        # 图是建一次缓存的：文件改了这一眼它还不知道
        self.assertIn("没有调用任何已知的函数", tool.execute("beta", "callees"))

        tool.refresh()
        self.assertIn("alpha", tool.execute("beta", "callees"))

    def test_an_external_symbol_refresh_makes_the_next_query_rebuild(self):
        tool = CallGraphTool(root=self.root)
        tool.execute("beta", "callees")

        self.make("pkg/alpha.py", ALPHA_CALLING)
        tool.index.symbols.refresh()  # 别处刷了符号索引（13.6 起图认得出）

        text = tool.execute("beta", "callees")
        self.assertIn("alpha", text)
        self.assertTrue(tool.index.built)

    def test_a_deleted_file_is_gone_after_refresh(self):
        self.make("pkg/gamma.py", "def gamma():\n    return alpha()\n")
        tool = CallGraphTool(root=self.root)
        self.assertIn("alpha", tool.execute("gamma", "callees"))

        self.remove("pkg/gamma.py")
        tool.refresh()

        self.assertIn("没有找到", tool.execute("gamma", "callees"))


# ---------------------------------------------------------------------------
# 向量缓存：删掉的块不留旧向量
# ---------------------------------------------------------------------------


class ForgetIdsTest(unittest.TestCase):
    """VectorCache.forget_ids 本身：只清点名的 id，别的照旧不动。"""

    GONE = "pkg/a.py:f:1-2|d1"
    GONE_SEGMENT = "pkg/a.py:f:1-2|d1#s0"
    KEPT = "pkg/b.py:g:3-4|d2"
    QUERY = "query:怎么保存"

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name).resolve()

    def cache(self, namespace: str = "Fake:v1") -> VectorCache:
        return VectorCache(cache_file(self.dir, namespace), namespace)

    def stored(self) -> dict:
        path = cache_file(self.dir, "Fake:v1")
        return json.loads(path.read_text(encoding="utf-8"))["vectors"]

    def fill(self, cache: VectorCache) -> None:
        cache.put_many(
            {
                self.GONE: (1.0, 0.0),
                self.GONE_SEGMENT: (0.5, 0.5),
                self.KEPT: (0.0, 1.0),
                self.QUERY: (1.0, 1.0),
            }
        )

    def test_it_drops_every_version_under_the_named_id(self):
        cache = self.cache()
        self.fill(cache)

        dropped = cache.forget_ids({"pkg/a.py:f:1-2"})

        self.assertEqual(dropped, 2)  # 整段 + 分段的那一条
        self.assertEqual(set(self.stored()), {self.KEPT, self.QUERY})

    def test_it_leaves_other_ids_and_queries_alone(self):
        cache = self.cache()
        self.fill(cache)

        cache.forget_ids({"pkg/elsewhere.py:h:1-9"})  # 不存在的 id

        self.assertEqual(
            set(self.stored()),
            {self.GONE, self.GONE_SEGMENT, self.KEPT, self.QUERY},
        )

    def test_an_empty_list_changes_nothing(self):
        cache = self.cache()
        self.fill(cache)

        self.assertEqual(cache.forget_ids(set()), 0)
        self.assertIsNone(cache.write_error)

    def test_a_broken_file_degrades_instead_of_raising(self):
        path = cache_file(self.dir, "Fake:v1")
        path.write_bytes(b"{ not json")
        cache = self.cache()

        self.assertEqual(cache.forget_ids({"pkg/a.py:f:1-2"}), 0)
        self.assertTrue(cache.degraded)


class SearchChunksVectorRefreshTest(RefreshTestBase):
    """fake embedding + 临时缓存目录：刷新之后旧向量一条都不留。"""

    def tool(self) -> tuple[SearchChunksTool, RecordingEmbedder]:
        embedder = RecordingEmbedder(FakeEmbedder(DEMO_ALIASES))
        tool = SearchChunksTool(
            root=self.root,
            embedding="fake",
            embedder=embedder,
            cache_dir=self.cache_dir,
        )
        return tool, embedder

    def stored_keys(self, tool: SearchChunksTool) -> set[str]:
        """磁盘缓存里现在有哪些 key（重新读一份文件，不看内存）。

        文件就是工具自己那份（`tool.vector_cache.path`），命名空间怎么拼的
        不用在这儿再猜一遍。
        """
        path = tool.vector_cache.path
        if not path.exists():
            return set()
        return set(json.loads(path.read_text(encoding="utf-8"))["vectors"])

    @staticmethod
    def ids_of(keys) -> set[str]:
        return {key.split("|", 1)[0] for key in keys if not key.startswith("query:")}

    def test_a_deleted_files_vectors_leave_the_cache(self):
        self.make("pkg/gamma.py", GAMMA)
        tool, _ = self.tool()
        tool.execute("壹号")  # 预热：两份文件都进缓存

        self.assertIn("pkg/gamma.py:gamma:1-3", self.ids_of(self.stored_keys(tool)))

        self.remove("pkg/gamma.py")
        tool.refresh()

        # 刷新当场就清（不用等下一次检索）
        self.assertNotIn("pkg/gamma.py:gamma:1-3", self.ids_of(self.stored_keys(tool)))
        self.assertIn("pkg/alpha.py:alpha:1-3", self.ids_of(self.stored_keys(tool)))
        # 内存里同样不留
        self.assertEqual(
            [identity for identity in tool.scorer._chunks if "gamma" in identity[0]],
            [],
        )

        tool.execute("壹号")  # 刷新之后再查一次：删掉的块还是不在
        self.assertNotIn("pkg/gamma.py:gamma:1-3", self.ids_of(self.stored_keys(tool)))

    def test_a_changed_body_is_embedded_again_under_a_new_key(self):
        tool, embedder = self.tool()
        tool.execute("壹号")
        chunk_id = "pkg/alpha.py:beta:6-8"

        def versions() -> list[str]:
            return [
                key for key in self.stored_keys(tool) if key.startswith(chunk_id + "|")
            ]

        self.assertEqual(len(versions()), 1)
        old_key = versions()[0]
        self.assertNotIn("return 200", embedder.seen())

        self.make("pkg/alpha.py", ALPHA_EDITED)  # 正文改了，行号没变
        tool.refresh()
        tool.execute("壹号")

        # 旧 digest 的那条被打扫掉了，只剩当前这一版
        self.assertEqual(len(versions()), 1)
        self.assertNotEqual(versions()[0], old_key)
        self.assertIn("return 200", embedder.seen())  # 新正文真的重嵌过
        # 内存里也只剩当前这一版（两条 chunk，各一份向量）
        remembered = [
            identity for identity in tool.scorer._chunks if identity[0] == chunk_id
        ]
        self.assertEqual(len(remembered), 1)
        self.assertEqual(
            remembered[0][1], tool.index.chunks(symbol="beta")[0].digest
        )
        self.assertEqual(len(tool.scorer._chunks), 2)

    def test_a_new_file_is_embedded_after_refresh(self):
        tool, embedder = self.tool()
        tool.execute("壹号")

        self.make("pkg/gamma.py", GAMMA)
        tool.refresh()
        tool.execute("叁号")

        self.assertIn("def gamma():", embedder.seen())
        self.assertIn("pkg/gamma.py:gamma:1-3", self.ids_of(self.stored_keys(tool)))

    def test_the_query_vectors_are_never_dropped_by_refresh(self):
        tool, _ = self.tool()
        tool.execute("壹号")
        queries = {key for key in self.stored_keys(tool) if key.startswith("query:")}
        self.assertTrue(queries)

        self.remove("pkg/alpha.py")
        tool.refresh()

        self.assertTrue(queries <= self.stored_keys(tool))

    def test_a_degraded_tool_does_not_retry_the_model_on_refresh(self):
        """模型坏了降级过：刷新不把它重新叫醒，字面检索照常。"""
        tool, embedder = self.tool()
        tool.execute("壹号")
        tool.guard.disable("假装模型坏了")
        batches = len(embedder.batches)

        self.make("pkg/gamma.py", GAMMA)
        tool.refresh()

        self.assertTrue(tool.guard.disabled)
        self.assertIn("pkg/gamma.py:1-3", tool.execute("gamma"))
        self.assertEqual(len(embedder.batches), batches)  # 没有再嵌过任何文本


if __name__ == "__main__":
    unittest.main()
