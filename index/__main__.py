"""`python -m index`：在真项目上跑自带的评估 case 集，最多四组可对比。

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m index
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m index --hybrid
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m index --local
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m index --multilingual
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m index --multilingual --no-cache
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m index --multilingual --no-segment

默认只跑字面打分（8.5 的 baseline）；`--hybrid` 再加一组 FakeEmbedder 的；
`--local` 再加一组英文本地模型（all-MiniLM-L6-v2）；`--multilingual` 再加
一组多语言模型（paraphrase-multilingual-MiniLM-L12-v2）——它把四组全跑
齐。四组摆在同一个 case 集上比 Hit@1/3/5——embedding 有没有用、真模型比
假货强多少、中英文查询各吃什么模型，看数字说话。

9.3 起 hybrid 组都走加速管线（结果和以前一模一样）：打分前把全部 chunk
一次批量预处理（而不是逐个 chunk 逐次调用模型），向量落一份用户目录下的
磁盘缓存——**第二次运行**直接读缓存，不重新算。每组报告后面多一行
`[embedding] ...` 统计（调用次数 / 嵌入条数 / 缓存命中 / 预热耗时）。
`--no-cache` 关掉缓存（实验对照用）；不关时缓存默认在
`~/.cache/universal-code-agent`，不进项目、不会被代码索引扫到。

9.5 起 hybrid 组默认**分段**：模型窗口装不下的块切成几段、各算一个向量，
打分取最像的那段（见 index/segments.py）。`--no-segment` 关掉它，复现
9.4 的「一个块一个向量」——两个开关配同一份缓存目录就是前后对照，两份
向量落在不同命名空间，不会串。每组还会印一行 `[segment] ...`：拿**真实
分词器**数一遍「还有多少条超窗」，分段前后各是多少，看数字。

9.5 起评估集分两份语料：项目自己（`project`）+ `tests/fixtures/cpp` 下那个
迷你 C++ 工程（`cpp_fixture`，生产索引按忽略名单不扫 `fixtures/`，所以它
不会混进上面的语料）。case 用 `corpus` 字段挑语料；报告多了「分组结果」
几行（整体 / 语言 / 查询语言 / 查询类型 / 语料 / 反例）。

`--local` / `--multilingual` 是**真实实验**，不在测试里：要装
sentence-transformers（可选依赖）、第一次还要下载模型。跑不动（没装库 /
下不下来）时给一句可行动的话，前面几组的结果照常留在屏幕上，退出码非 0。
测试全离线，真模型永远不进测试。

选哪组模型是 provider 的事（index/providers.py），这里只给名字——加一种
模型不用动这个文件。

入口放在这儿而不是 index/evaluation.py 里，是因为 index/__init__.py 会导入
那个模块，再用 `-m` 直接跑它，Python 会打一行 RuntimeWarning。
"""

import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from index import (
    DEFAULT_CORPUS,
    DEMO_ALIASES,
    ChunkIndex,
    Embedder,
    EmbeddingScorer,
    EmbeddingUnavailable,
    EvaluationCase,
    Evaluator,
    FakeEmbedder,
    Retriever,
    cache_for,
    combined_stats_line,
    compact_text,
    create_embedder,
    default_cache_dir,
    hybrid_scorers,
    load_cases,
    segment_chunk,
)
from tools.base import ToolError

# C/C++ 评估的目标代码：一个很小的真实 C++ 工程（环形缓冲区），放在测试
# 素材目录里。**只在评估入口建索引**：tools/tree.py 的忽略名单里有
# `fixtures`，所以项目自己的索引和工具都看不见它，生产语料保持原样。
CPP_FIXTURE_CORPUS = "cpp_fixture"
CPP_FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "cpp"
)


def build_corpora() -> dict[str, ChunkIndex]:
    """这次评估要用的语料：项目自己，加上 C++ fixture（在的话）。"""
    project = ChunkIndex()
    project.build()  # 建索引不算检索耗时，先建好
    table = {DEFAULT_CORPUS: project}
    if CPP_FIXTURE_ROOT.is_dir():
        fixture = ChunkIndex(root=CPP_FIXTURE_ROOT)
        fixture.build()
        table[CPP_FIXTURE_CORPUS] = fixture
    return table


def _as_corpora(subject: ChunkIndex | Mapping[str, ChunkIndex]) -> dict[str, ChunkIndex]:
    """一个索引（= 默认语料）或一张语料表，统一成表。"""
    if isinstance(subject, Mapping):
        if not subject:
            raise ToolError("语料表是空的：至少得有一个索引")
        return dict(subject)
    return {DEFAULT_CORPUS: subject}


def baseline_retrievers(
    corpora: Mapping[str, ChunkIndex],
) -> dict[str, Retriever]:
    """每个语料配一个字面打分的 retriever（8.5 的 baseline）。

    评估要的是「语料名 → retriever」，`build_corpora()` 给的是
    「语料名 → 索引」——中间这一步就靠它，测试里也一样用。
    """
    return {name: Retriever(index=corpus) for name, corpus in corpora.items()}


def _hybrid_scorers(
    index: ChunkIndex,
    embedder: Embedder,
    cache_dir: Path | None,
    segment: bool,
):
    """一个 hybrid 组的 scorers：总是批量预热（chunks=），给目录就再挂缓存。"""
    if cache_dir is None:
        return hybrid_scorers(embedder, chunks=index.chunks, segment=segment)
    # 缓存命名空间（模型身份 + 文本管线版本）由 cache_for 一处管（9.4/9.5）：
    # 评估入口和 Agent 工具共用，免得哪天只改了一边
    return hybrid_scorers(
        embedder,
        chunks=index.chunks,
        cache=cache_for(embedder, cache_dir, segmented=segment),
        segment=segment,
    )


def variants(
    index: ChunkIndex | Mapping[str, ChunkIndex],
    argv: list[str],
    local_embedder: Embedder | None = None,
    multilingual_embedder: Embedder | None = None,
    cache_dir: Path | None = None,
) -> list[tuple[str, dict[str, Retriever]]]:
    """这次要跑哪几组：(标题, {语料名: retriever})。组是递进的。

    `index` 给一个索引就是「只有默认语料」的老形状，给一张表就每个语料各
    配一个 retriever（同一组的几个语料共用一份缓存）。`local_embedder` /
    `multilingual_embedder` 是给测试的口子：不传就按 provider 名字造真的
    （构造不加载模型，缺依赖要到第一次 embed 才报，那是跑起来之后的事，
    这里的 try 兜着）。`cache_dir` 不给就不挂磁盘缓存（测试默认走这条，
    不往用户目录写东西）。`--no-segment` 关掉 9.5 的分段。
    """
    table = _as_corpora(index)
    segment = "--no-segment" not in argv
    suffix = "" if segment else "（不分段）"

    def hybrid(title: str, embedder: Embedder) -> tuple[str, dict[str, Retriever]]:
        return (
            title + suffix,
            {
                name: Retriever(
                    index=corpus,
                    scorers=_hybrid_scorers(corpus, embedder, cache_dir, segment),
                )
                for name, corpus in table.items()
            },
        )

    groups = [("字面打分（baseline）", baseline_retrievers(table))]
    if "--hybrid" in argv or "--local" in argv or "--multilingual" in argv:
        groups.append(hybrid("加上假 embedding（hybrid）", FakeEmbedder(DEMO_ALIASES)))
    if "--local" in argv or "--multilingual" in argv:
        embedder = (
            create_embedder("local") if local_embedder is None else local_embedder
        )
        groups.append(hybrid("再加上英文本地模型（local hybrid）", embedder))
    if "--multilingual" in argv:
        embedder = (
            create_embedder("multilingual")
            if multilingual_embedder is None
            else multilingual_embedder
        )
        groups.append(hybrid("再加上多语言模型（multilingual hybrid）", embedder))
    return groups


def overflow_line(
    scorer: EmbeddingScorer, corpora: Mapping[str, ChunkIndex]
) -> str | None:
    """分段（或没分段）之后还有多少文本超过模型窗口——诊断，不是指标。

    用 embedder 的**真实分词器**数（`count_tokens`），因为要回答的就是
    「估计切法到底修好了没有」。拿不到分词器（FakeEmbedder）就返回 None，
    这一行不印。分段时量的是每一段；没分段（`--no-segment`）量的是整块
    紧凑文本——两个数摆一起就是 9.4 → 9.5 的变化。
    """
    counter = getattr(scorer.embedder, "count_tokens", None)
    window = scorer.max_tokens
    if counter is None or window is None:
        return None
    segmented = scorer.segmented
    try:
        total = over = 0
        for corpus in corpora.values():
            for chunk in corpus.chunks():
                texts = (
                    segment_chunk(chunk, window)
                    if segmented
                    else (compact_text(chunk),)
                )
                for text in texts:
                    total += 1
                    if counter(text) > window:
                        over += 1
    except Exception as exc:  # 诊断而已，坏了一行字带过，别拖垮评估
        return f"[segment] 真实分词诊断跳过：{type(exc).__name__}: {exc}"
    label = "分段后" if segmented else "未分段"
    return (
        f"[segment] {label}仍有 {over}/{total} 条超过模型窗口 "
        f"（{over / total:.1%}，窗口 {window} token）"
    )


def run_reports(
    cases: Iterable[EvaluationCase],
    index: ChunkIndex | Mapping[str, ChunkIndex],
    argv: list[str],
    out: Callable[[str], None] | None = None,
    local_embedder: Embedder | None = None,
    multilingual_embedder: Embedder | None = None,
    cache_dir: Path | None = None,
) -> int:
    """把这次要跑的几组挨个跑一遍，一组一段报告打到 out。返回退出码。

    本地模型那组跑不动时说明情况并返回 1——前面几组已经打出来了，不白跑。
    """
    say = print if out is None else out
    table = _as_corpora(index)
    if cache_dir is not None:
        say(f"[cache] 向量缓存目录：{cache_dir}")

    for position, (title, retrievers) in enumerate(
        variants(
            table,
            argv,
            local_embedder=local_embedder,
            multilingual_embedder=multilingual_embedder,
            cache_dir=cache_dir,
        ),
        start=1,
    ):
        if position > 1:
            say("")
        say(f"=== {title} ===")
        scorers = [
            scorer
            for retriever in retrievers.values()
            for scorer in retriever.scorers
            if isinstance(scorer, EmbeddingScorer)
        ]
        try:
            # 预热（批量算向量）从第一条查询里拆出来：耗时不混进单条检索，
            # 也让它跑不动时尽早报
            for scorer in scorers:
                scorer.prefetch()
            report = Evaluator(retrievers).run(cases)
            say(report.summary())
            grouped = report.grouped_summary()
            if grouped:
                say("分组结果：")
                say(grouped)
            for scorer in scorers:
                scorer.flush_cache()
            if scorers:
                say(combined_stats_line(scorers))
                # 每个语料一个 scorer，各自数一遍同一份语料表——数字一样，
                # 只印第一份，免得同一行诊断重复出现
                seen: set[str] = set()
                for scorer in scorers:
                    line = overflow_line(scorer, table)
                    if line is not None and line not in seen:
                        seen.add(line)
                        say(line)
        except EmbeddingUnavailable as exc:
            say(f"[本地模型不可用] {exc}")
            return 1
    return 0


def main(argv: list[str]) -> int:
    cases = load_cases()
    cache_dir = None if "--no-cache" in argv else default_cache_dir()
    return run_reports(cases, build_corpora(), argv, cache_dir=cache_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
