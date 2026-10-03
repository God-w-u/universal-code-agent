"""改完代码之后，接着跑一次验证，并把新老诊断摆在一起。

edit_file 只保证「改的是你指的那一处」，不保证「改对了」——改完还得编一遍
才知道。这个模块就是那一步：Agent 每次工具调用之后都问它一句，模型刚改过
文件它就接着验一次。

它自己一行命令都不拼。跑什么，全交给注册表里现成的 compile / build / run_tests
（怎么拼命令、怎么解析诊断都在 tools/ 里，这里一行都不重复），它只管三件事：

- **挑哪个**：上次编的就是这个文件就原样重编；否则改 Python 用 run_tests，
  改 C/C++ 看工程里有没有构建系统，有就用 build、没有才 compile。
- **跟什么比**：模型自己跑的每一次编译 / 构建 / 测试都留着，当作「之前」，
  和验证跑出来的「之后」对一遍。
- **怎么报**：过没过、原来那条错没了没有、有没有新冒出来的。

C/C++ 的改动是**两关**：编译 / 构建过了，还会接着跑一次 run_tests（7.6 加的）。
编得过只说明编得过——逻辑写错了照样编得过。所以「这轮算不算修好」看
Verification.ok（两关都过），不是看编译那一关的 passed。

这里只跑「一次」验证：不循环、不自动回滚、不自动再改。拿着这次结果决定
要不要再修一轮的是 RepairLoop（agent/repair.py），它读的是这里留下的
Verification——同一次跑出来的结构化版本，和给模型看的文本不会对不上。
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tools import diagnostics
from tools.diagnostics import Diagnostic, Report
from tools.paths import check_root

if TYPE_CHECKING:
    # 只在做类型检查时导入：运行时不认识具体的类，谁先被 import 都不会成环
    from tools.executor import ToolExecutor, ToolResult
    from tools.registry import ToolRegistry

# 能用来验证的就是这三个工具。别的（读文件、搜索、算数）验不了「改完对不对」。
COMPILE = "compile"
BUILD = "build"
RUN_TESTS = "run_tests"
VERIFIERS = (COMPILE, BUILD, RUN_TESTS)

# 会改文件的工具。它跑成功之后要接着验一次。
EDIT = "edit_file"

# 文件按后缀分三类。只有这三类谈得上「编译 / 跑测试」。
_SOURCES = (".c", ".cc", ".cpp", ".cxx")
_HEADERS = (".h", ".hh", ".hpp", ".hxx")
_PYTHON = (".py",)

SOURCE = "source"
HEADER = "header"
PYTHON = "python"
OTHER = "other"

# 这类文件要靠哪些工具验证。挑不出能用的那个时，拿它把原因说清楚。
_NEEDS = {
    SOURCE: (COMPILE, BUILD),
    HEADER: (BUILD,),
    PYTHON: (RUN_TESTS,),
}

# 找构建系统的标记文件，按顺序：CMake、Make、MSVC 那套。
# 值是 glob 写法，字面名字也照样走 glob——两套判断容易只改一边。
_BUILD_FILES = (
    ("cmake", ("CMakeLists.txt",)),
    ("make", ("Makefile", "makefile", "GNUmakefile")),
    ("msbuild", ("*.sln", "*.vcxproj")),
)

# compile / build 在自己那段结果末尾附诊断时的开头（见 tools/diagnostics.render）。
# 比较之前必须把这段切掉：它是**同一批诊断的另一种写法**，再解一遍会当成新诊断。
# 实测：GCC 那一路的 `a.cpp:2:18: error: ...` 被原样渲染回去，再解一次就翻倍成两条，
# 比较结果会变成「旧的还在、又来一条新的」——两句话都是假的。
_APPENDED = "\n\n诊断（共 "

# run_command 每跑完一条命令都这么写头两行（见 tools/run_command._render）：
#
#     命令：<命令>
#     退出码：0
#
# 超时走的是另一句「结果：超时，已强制终止」。要判断过没过只能从这段文本里读：
# compile / build / run_tests 返回的就是文本，改它们的返回值就是动了核心行为。
_EXIT_CODE = re.compile(r"^命令：.*\n退出码：(\d+)", re.MULTILINE)
_TIMEOUT = re.compile(r"^结果：超时", re.MULTILINE)

# 认不出诊断又没通过时，把原文节选给模型看。够看到「项目文件不存在」这类
# 没有行列号的失败原因就行；要看全文模型可以自己再跑一次那个工具。
_RAW_LIMIT = 1200

# run_tests 跑的是一条钉死的命令（见 tools/run_tests.py）：
#     python -m unittest discover -s tests -t .
# 它的测试都在 tests 这个名字的目录里，所以「这个项目有没有测试」就看那儿。
_TESTS_DIR = "tests"


@dataclass(frozen=True)
class Call:
    """一次验证调用：用什么工具、带什么参数。"""

    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Comparison:
    """两批诊断的差。"""

    gone: tuple[Diagnostic, ...]  # 之前有、现在没了
    still: tuple[Diagnostic, ...]  # 还在（位置取新的那次）
    added: tuple[Diagnostic, ...]  # 之前没有、现在冒出来的


@dataclass(frozen=True)
class Verification:
    """一次**真的跑起来了**的验证，结构化的那一份。

    给模型看的是渲染好的文本（check / after_tool_call 的返回值），这个是同一
    件事给程序看的样子：RepairLoop 靠它判断「这轮修好没有」「文件和诊断跟上一轮
    一样吗」。两者出自同一次跑，不会各说各话。

    C/C++ 的验证是**两关**：先编译 / 构建（本体的这几个字段），这一关过了才
    接着跑 run_tests——那一关整份放在 tests 里。编译没过就没有 tests（跑不了，
    产物还是改之前那个），「整条链过没过」看 ok。
    """

    call: Call
    call_text: str  # 「compile main.cpp」这种写法，报告里要用
    passed: bool  # 这一关过没过（tests 那份里就是测试过没过）
    exit_code: int | None  # 没在输出里读到退出码时为 None
    report: Report  # 这一次看到的全部诊断
    diff: Comparison | None  # 和上一次的诊断比；没有基线时为 None
    tests: "Verification | None" = None  # 编译过了之后接着跑的那次 run_tests
    text: str = ""  # 这一关的原始输出。测试那关没有结构化诊断，只能看它

    @property
    def ok(self) -> bool:
        """整条验证链都过了才算数：编译 / 构建过了，该跑的测试也过了。

        「编译通过」只说明这一关过了。代码编得过、测试全红，那不叫修好——
        RepairLoop 判断「这轮成了没有」用的是这个，不是 passed。
        """
        return self.passed and (self.tests is None or self.tests.passed)


class Verifier:
    """改完一次代码，跑一次验证，并把它和上一次的诊断摆在一起。

    registry 和 executor 就是 Agent 手里那两个：工具从注册表里拿，跑的时候走
    执行层（异常拦得住、trace 里也看得到）。这里不新开一条执行路径。
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        executor: "ToolExecutor",
        root: Path | None = None,
    ) -> None:
        self.registry = registry
        self.executor = executor
        self.root = _root_of(registry, root)
        # 上一次编译 / 构建 / 测试看到的东西，作为下一次比较的「之前」
        self._last: tuple[Call, Report] | None = None
        # 最近一次真的跑起来了的验证（模型自己跑的和自动验的都算）。
        # 没验证过、或者最近一次没跑成时为 None——「最新的结论是没有结论」
        self.last: Verification | None = None

    def after_tool_call(
        self,
        tool: str,
        arguments: dict[str, Any],
        result: "ToolResult",
    ) -> str:
        """每次工具调用之后问它一句，返回要接在结果后面的那段（没有就返回空串）。

        改过文件就接着验一次；模型自己跑的编译 / 构建 / 测试则记下来当基线。
        """
        if not result.success:
            # 工具自己就没跑起来（参数不对、文件不存在……），这种结果里没有
            # 代码现在的状态，拿它当基线只会把真的那份顶掉；
            # 也顺手把 last 清掉——这一次什么都没验出来
            self.last = None
            return ""

        if tool == EDIT:
            return self.check(arguments.get("path"))

        self.note(tool, arguments, result.content)
        return ""

    def note(self, tool: str, arguments: dict[str, Any], text: str) -> None:
        """记下一次编译 / 构建 / 测试的结果——它是下一次比较的「之前」。

        别的工具的结构化结果里没有「代码现在什么状态」，记下来反而会把
        真正的基线顶掉，所以只认这三个。
        """
        if tool not in VERIFIERS:
            return
        # 复制一份参数：调用方之后改它，不该影响已经记下的这条
        call = Call(tool, dict(arguments))
        report = diagnostics.parse(_raw(text))
        before = self._last[1] if self._last is not None else None
        self._last = (call, report)
        self.last = _verification(call, text, report, before)

    def check(self, path: Any) -> str:
        """edit_file 改完之后调它：跑一次验证，返回接到编辑结果后面的那段。"""
        # 先清空：下面任何一条提前返回的路都意味着「这一次没验出东西」，
        # 留着上一次的结论，调用方会以为是这次的（改的是 A，指着 B 的旧结果）
        self.last = None

        if not isinstance(path, str) or not path.strip():
            return ""

        call = self._choose(path)
        if call is None:
            return self._explain(path)

        result = self.executor.execute(call.tool, **call.arguments)
        if not result.success:
            # 验证用的工具自己都没跑起来（没装编译器、参数不认……）。
            # 这不是「改错了」，是「验不了」——两种得分开说，
            # 不然模型会去改本来是对的代码
            return f"验证（{_call_text(call)}）没跑成：{result.error}"

        before = self._last[1] if self._last is not None else None
        # 这一次的结果就是下一次改动的「之前」
        self.note(call.tool, call.arguments, result.content)
        diff = self.last.diff if self.last is not None else None
        text = _render(call, result.content, before, diff)

        # 编译 / 构建这一关过了，就再往下走一关：跑测试。编得过不等于改对了
        return text + self._run_tests(call)

    def _run_tests(self, primary: Call) -> str:
        """编译 / 构建过了之后接着跑一次 run_tests，返回要接上去的那段。

        走的是同一个执行层，所以这次跑测试在 trace 里也看得到。不跑的有五种，
        每一种都得说得出道理（前三种不算失败，最后两种本来就没法跑）：

        - 改的就是 Python：run_tests 本来就是它的第一关，不重复跑；
        - 这个 Agent 没装配 run_tests：那就只有编译这一关（7.5 的行为）；
        - 编译 / 构建没过：产物还是改之前那一个，跑测试测的不是这次的代码；
        - 这个项目没有能跑的测试（见 _has_tests）——跳过，但要说一声，
          不然「没跑测试」会被当成「测试过了」；
        - 跑测试的工具自己没跑起来：说「没跑成」，和「测试没过」分开。
        """
        if primary.tool == RUN_TESTS or RUN_TESTS not in self.registry:
            return ""
        if self.last is None or not self.last.passed:
            return ""
        if not _has_tests(self.root):
            return (
                f"  测试：这个项目里没有能跑的测试（{_TESTS_DIR} 不是个包），"
                f"这一关跳过——这次改动只以编译 / 构建为准。"
            )

        result = self.executor.execute(RUN_TESTS)
        if not result.success:
            return f"  测试（{RUN_TESTS}）没跑成：{result.error}"

        report = diagnostics.parse(_raw(result.content))
        tests = _verification(Call(RUN_TESTS, {}), result.content, report, None)
        # 只把这一关记进 self.last，**不**拿它当下一次比较的「之前」：
        # 诊断比的是编译那一关，测试输出不是结构化诊断，让它顶掉基线会把
        # 「上一轮编译报了什么」整条链断掉
        if self.last is not None:
            self.last = replace(self.last, tests=tests)

        lines = [f"  测试（{RUN_TESTS}）：{_outcome(result.content)[1]}"]
        if not tests.passed:
            # 测试红在哪儿得让模型看得到——unittest 的失败详情就在这里
            lines.append("  测试输出（节选）：")
            lines.append(_indent(_clip(result.content)))
        return "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # 挑哪个工具
    # ------------------------------------------------------------------

    def _choose(self, path: str) -> Call | None:
        """这次该跑哪个工具；没有能跑的就返回 None。"""
        if self._last is not None and self.covers(self._last[0], path):
            # 上次跑的那条还管用，就原样再跑一遍。模型多半是先编译看到错
            # 才来改的，用同一条命令（一样的 flags、一样的构建目录）才比得出来
            return self._last[0]

        kind = _kind(path)

        if kind == PYTHON:
            return Call(RUN_TESTS, {}) if RUN_TESTS in self.registry else None

        if kind in (SOURCE, HEADER):
            found = _build_system(self.root, path)
            if found is not None and BUILD in self.registry:
                system, source_root = found
                return Call(BUILD, {"system": system, "source_root": source_root})
            if kind == SOURCE and COMPILE in self.registry:
                # 工程里没有构建系统，就是一个单文件：编它自己
                return Call(COMPILE, {"source": path})

        return None

    def covers(self, call: Call, path: str) -> bool:
        """这条命令，能不能用来验证这个文件。

        RepairLoop 也要用它：模型自己跑了一次通过了的验证时，得先问一句
        「验的是不是正在修的那个文件」才敢说不用修了。
        """
        kind = _kind(path)

        if call.tool == COMPILE:
            # 编的就是这个文件才作数。改的是头文件时**不**拿它顶：上次编的
            # 那个源文件不一定包含这个头文件，那样报出来的「通过」是假的
            return call.arguments.get("source") == path

        if call.tool == BUILD:
            # 整个工程重编一遍，改哪个源文件、头文件都覆盖得到
            return kind in (SOURCE, HEADER)

        # run_tests 验的是行为：只有改 Python 时它才说明问题。改了个 .cpp
        # 却只看测试过没过，等于没验——那是最糟的一种，**假通过**
        return kind == PYTHON

    def _explain(self, path: str) -> str:
        """这次没跑成验证，说清楚是为什么。

        不能什么都不说：模型会把「没消息」当成「通过了」，而它恰恰是
        没被验证过的那一种。
        """
        kind = _kind(path)

        if kind == HEADER:
            # 头文件这一条要排在「工具没装配」前面：不管装没装 build，
            # 「拼一个包含它的源文件」都是唯一真能验证它的办法，
            # 这句话比「缺个工具」有用得多
            hint = ""
            if self._last is not None and self._last[0].tool == COMPILE:
                source = self._last[0].arguments.get("source")
                hint = f"，比如上次编过的 {source}"
            return (
                f"验证：{path} 是头文件，没有能自动验证它的办法——单独编译一个"
                f"头文件报出来的多半是链接错误（它本来就不含 main），不是真问题。"
                f"要验证它，请编译一个包含它的源文件{hint}"
            )

        needs = _NEEDS.get(kind, ())
        if needs:
            # 这类文件本来有对应的验证工具，只是这个 Agent 没装配
            return (
                f"验证：这个 Agent 没装配能验证 {path} 的工具"
                f"（要 {'、'.join(needs)}），这次改动没有验证过。"
            )

        return f"验证：{path} 不是能编译或跑测试的代码文件，这次改动没有验证过。"


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------


def compare(before: Report, after: Report) -> Comparison:
    """把「之前」和「之后」两批诊断对一遍。"""
    pool = list(after.diagnostics)
    gone: list[Diagnostic] = []
    still: list[Diagnostic] = []

    for item in before.diagnostics:
        index = _find(pool, item)
        if index is None:
            gone.append(item)
        else:
            # 报现在的位置：模型要拿新行号去 read_file 核对
            still.append(pool.pop(index))

    # 挑剩下的就是新冒出来的，顺序还是它们在输出里的先后
    return Comparison(tuple(gone), tuple(still), tuple(pool))


def _find(pool: list[Diagnostic], item: Diagnostic) -> int | None:
    """池子里有没有和它算同一条的，返回下标；没有则返回 None。"""
    key = _identity(item)
    for index, other in enumerate(pool):
        if _identity(other) == key:
            return index
    return None


def _identity(item: Diagnostic) -> tuple:
    """认「同一条诊断」时比什么。

    **不比行列号**：改完代码，行号整段挪位是常事——上面删掉一行，同一条错
    就从第 5 行变成第 4 行。按行列号比会把它说成「旧的没了、又来一条新的」，
    而这两句话都是假的，模型会照着这个结论去查一个并不存在的新问题。
    文件 + 级别 + 错误码 + 消息一样，就算同一条。
    """
    return (item.file, item.level, item.code, item.message)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------


def _render(
    call: Call,
    text: str,
    before: Report | None,
    diff: Comparison | None = None,
) -> str:
    """把这次验证摆成给模型看的一段。

    diff 是「和上一次比」的结果；没传就现算（调用方算过一遍了就传进来，
    免得同一件事算两次）。
    """
    ok, verdict = _outcome(text)
    report = diagnostics.parse(_raw(text))
    lines = [f"验证（{_call_text(call)}）：{verdict}"]

    if before is None:
        # 头一次验证也有诊断要看，只是没有「之前」可比
        lines.append("  之前没有编译、构建或测试过，没有可比较的老诊断。")
        lines += _listed("现在的问题", report.diagnostics)
    else:
        lines.append(
            f"  诊断：{_stage('之前', before.diagnostics)}；"
            f"{_stage('现在', report.diagnostics)}。"
        )
        lines += diff_lines(compare(before, report) if diff is None else diff)

    if not ok and not report.has_problems:
        # 没通过，可一条诊断都认不出来——比如 msbuild 的「项目文件不存在」
        # 那种工程级错误。只给一句「没通过」，模型没法接着查
        lines.append("  没能从输出里认出结构化诊断，原文节选：")
        lines.append(_indent(_clip(text)))

    return "\n".join(lines)


def _verification(
    call: Call, text: str, report: Report, before: Report | None
) -> Verification:
    """把一次跑完的验证整理成结构化的那份（给程序看）。"""
    passed, _ = _outcome(text)
    codes = _exit_codes(text)
    failed = next((code for code in codes if code != 0), None)

    return Verification(
        call=call,
        call_text=_call_text(call),
        passed=passed,
        # 两步构建里有一步非 0，报的就是那一步——和「过没过」用的是同一条判断
        exit_code=failed if failed is not None else (codes[-1] if codes else None),
        report=report,
        diff=compare(before, report) if before is not None else None,
        text=text,
    )


def _outcome(text: str) -> tuple[bool, str]:
    """这次验证过没过，以及该拿什么话去说它。"""
    codes = _exit_codes(text)
    if not codes:
        # 没读到退出码就不敢说通过：编译器的报错、命令找不到，都可能一条
        # 结构化诊断都留不下，那种「通过」是假的
        if _TIMEOUT.search(text):
            return False, "没通过（超时被终止）"
        return False, "没通过（没在输出里找到退出码）"

    failed = next((code for code in codes if code != 0), None)
    if failed is None:
        return True, f"通过（退出码 {codes[-1]}）"
    return False, f"没通过（退出码 {failed}）"


def diff_lines(diff: Comparison) -> list[str]:
    """把差摆成三行，空的那种不占位置。"""
    return (
        _listed("已消失", diff.gone)
        + _listed("仍在", diff.still)
        + _listed("新增", diff.added)
    )


def _listed(label: str, items: tuple[Diagnostic, ...]) -> list[str]:
    """「- 标签：第一条」，后面几条对齐着往下排。"""
    if not items:
        return []
    return [f"  - {label}：{items[0].as_line()}"] + [
        f"    {item.as_line()}" for item in items[1:]
    ]


def _call_text(call: Call) -> str:
    """这条验证命令长什么样，模型要认得出它跑的是哪一条。"""
    if call.tool == COMPILE:
        return f"{COMPILE} {call.arguments.get('source', '')}"
    if call.tool == BUILD:
        return f"{BUILD} {call.arguments.get('system', '')}"
    return RUN_TESTS


def _stage(label: str, items: tuple[Diagnostic, ...]) -> str:
    """「之前 1 个 error」/「现在没有 error 或 warning」。

    「现在」和「没有」中间不能有空格（中文里不这么写），所以整句在这里拼好，
    调用方只管把它贴上去。说法和 diagnostics.render 保持一致：
    同一种东西不该有两套说法。
    """
    counted = _counts(items)
    return f"{label} {counted}" if counted else f"{label}没有 error 或 warning"


def _counts(items: tuple[Diagnostic, ...]) -> str:
    """排成「2 个 error，1 个 warning」；一条都没有时返回空串。"""
    total: dict[str, int] = {}
    for item in items:
        total[item.level] = total.get(item.level, 0) + 1

    order = (diagnostics.ERROR, diagnostics.WARNING, diagnostics.NOTE)
    return "，".join(f"{total[level]} 个 {level}" for level in order if level in total)


# ---------------------------------------------------------------------------
# 文本、路径这些小工具
# ---------------------------------------------------------------------------


def _raw(text: str) -> str:
    """去掉 compile / build 在末尾附的那段诊断，只留原始输出。"""
    return text.split(_APPENDED, 1)[0]


def _exit_codes(text: str) -> list[int]:
    """文本里每一次「命令跑完」的退出码。

    build 可能跑两步（cmake 配置 + 编译），每一步都有自己的退出码，
    所以返回的是一个列表：只要有一个非 0，这次构建就没成功。
    """
    return [int(hit.group(1)) for hit in _EXIT_CODE.finditer(text)]


def _clip(text: str) -> str:
    """原文只给开头一段。要看全文，模型可以自己再跑一次那个工具。"""
    body = text.strip()
    if len(body) <= _RAW_LIMIT:
        return body
    return f"{body[:_RAW_LIMIT]}\n…（原文更长，这里只给了开头）"


def _indent(text: str) -> str:
    """整段往右挪两格，和上面的字段对齐。"""
    return "\n".join(f"  {line}" for line in text.split("\n"))


def _has_tests(root: Path) -> bool:
    """这个项目有没有能跑的测试。

    run_tests 跑的是 `python -m unittest discover -s tests -t .`，这条命令
    只有在 tests 是个**能导入的包**时才跑得起来（实测：光有个 tests 目录、
    或者目录里有 test_*.py 但没有 `__init__.py`，都只会得到一句
    「Start directory is not importable」并以非 0 退出）。

    那是「没有测试」，不是「测试没过」——拿它当失败，模型会去改本来是对的
    代码，正是 7.4 里那条「假通过」的镜像：假失败。所以先看这一眼。
    """
    return (root / _TESTS_DIR / "__init__.py").is_file()


def _kind(path: str) -> str:
    """这个文件算哪一类。只看后缀，和文件在哪儿无关。"""
    suffix = Path(path).suffix.lower()
    if suffix in _SOURCES:
        return SOURCE
    if suffix in _HEADERS:
        return HEADER
    if suffix in _PYTHON:
        return PYTHON
    return OTHER


def _build_system(root: Path, path: str) -> tuple[str, str] | None:
    """从要改的文件那一层往上找构建文件，返回 (构建系统, 源码目录)。

    往上找是因为 CMakeLists.txt 一般在工程根上，而改的往往是 src/ 里的文件。
    源码目录是相对 root 的（root 本身算 "."），直接就能当 build 的参数用。
    """
    folder = (root / path).parent
    while True:
        system = _build_file(folder)
        if system is not None:
            try:
                return system, folder.relative_to(root).as_posix()
            except ValueError:
                # 不在 root 里（正常情况下 edit_file 已经拦过了），当没找到
                return None
        # 走到 root 就到头；folder.parent == folder 是盘符根，兜底防死循环
        if folder == root or folder.parent == folder:
            return None
        folder = folder.parent


def _build_file(folder: Path) -> str | None:
    """这个目录里有没有构建文件，有的话是哪一套。"""
    for system, patterns in _BUILD_FILES:
        for pattern in patterns:
            if any(folder.glob(pattern)):
                return system
    return None


def _root_of(registry: "ToolRegistry", root: Path | None) -> Path:
    """验证在哪个项目里跑。

    没明说就跟着注册表里的工具走：验证要找构建文件、要比相对路径，得和
    这些工具在同一个项目里，不然找到的构建系统根本不是它们编译的那个。
    """
    if root is not None:
        return check_root(root)

    for tool in registry.list():
        found = getattr(tool, "root", None)
        if found is not None:
            return check_root(found)

    return check_root(None)
