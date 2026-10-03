"""验证没过就接着修：一轮一轮来，修好、修不动或者到上限就停。

7.4 只做到「改完验一次，把结果摆给模型看」——验完就交回给模型，修不修、怎么修
是模型自己的事。这个模块补上那一步：验不过就说明还没完，模型想收尾也不行
（会被按回去接着修），一直修到下面某一条成立：

- **验证通过了**：修复结束，模型正常给最终回答。7.6 起「通过」是两关都过——
  编译 / 构建过了，测试也过了（Verification.ok）。编译通过只能说明这一关过了；
- **用满 max_repair_rounds**：立即停，报告当前状态；
- **同一次改动又做了一遍**（改动指纹重复，比如 A 改成 B、又改回 A 这么横跳）：
  停，说清楚「检测到重复修改」；
- **连着两轮看到的一模一样**（编译诊断相同，或者测试输出一字不差）：停，
  说清楚「自动修不动了」；
- **改了半天文件一个字节没变**：停；
- **模型连着两次只回话不动手**：停，把它的说法一并报出去。

停下来的地方就是停下来的地方：代码保留最后一次修改后的样子，不回滚、也不再
自动改——要接着修，下一轮对话里模型看着完整历史继续就是了。停下时（以及修好
时）都有一句 Repair Summary：几轮、每轮卡在哪一关、最后成没成。

几个刻意的边界：

- 修复轮数和「工具调用总步骤数」（max_steps）是**两个概念**：一个管「最多修几轮」，
  一个管「一次 run() 里最多问几次模型」。步骤用完而修复还没结束时，这里会
  收个尾把状态报出去，而不是让 AgentError 把这轮对话整个回滚掉。
- 只有 **edit_file 之后的验证没过**才算欠着一次修复。模型自己编一遍看到错、
  但没动手改，那是「在看」，不是在修——那会儿它想直接给结论，不该拦。
- **一次 edit_file 加上它触发的那次验证，就是一轮**（7.8 起）。模型没先编译、
  直接改完第一次验就过的那种也算：轮次状态记成「一轮通过」。7.5–7.7 只认
  「先见过坏状态」的那种，于是「明明一次就改好了，记录里却写着 0 轮」——
  记录和事实对不上。没改代码的纯问答 / 只读工具调用仍然一轮都不记
  （没有 edit_file，就没有「修」这回事），验证没跑成的（验不了、工具没装配）
  也不记。
- 「同一个文件改了好几次」不是停的理由（同一个文件里两个错，本来就该改两次）。
  停的是**同一次改动重复做**：位置和内容都回到之前某一轮去。

它自己一行命令都不拼，也不直接碰工具：验证（含编译之后那次跑测试）全交给
Verifier，跑起来还是走 ToolExecutor——所以每一步在 Trace 里都看得到。
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.verify import (
    EDIT,
    VERIFIERS,
    Comparison,
    Verification,
    Verifier,
    compare,
    diff_lines,
)
from tools.base import ToolError
from tools.diagnostics import Report
from tools.paths import resolve_in_root

if TYPE_CHECKING:
    # 只在做类型检查时导入：运行时不认识具体的类
    from tools.executor import ToolResult

# 一次 run() 里最多自动修几轮。5 是留给正常情况的：一轮修好是常态，
# 需要改第二处、第三处的也有；到第 5 轮还一片红，多半不是「再试一次」的事了。
DEFAULT_MAX_REPAIR_ROUNDS = 5

# edit_file 结果里那一行「位置：…」的开头。改动摘要在那儿，指纹也从那儿取。
_POSITION = "位置："

# 主循环下一步该干什么。
CONTINUE = "continue"  # 接着问模型
FINISH = "finish"  # 可以收尾：把模型自己的回答返回给用户
NUDGE = "nudge"  # 还不能收尾：把 text 当新消息发回给模型，让它接着修
REPORT = "report"  # 自动修复结束了：text 就是给用户的报告


@dataclass(frozen=True)
class Decision:
    """主循环下一步怎么走。"""

    action: str
    text: str = ""


@dataclass(frozen=True)
class RepairRound:
    """一轮修复：一次 edit_file，加上它触发的那次验证（编译 / 构建 + 测试）。

    轮次从 1 起数。一道轮的三个阶段各记各的：**改了什么**（edit、fingerprint）、
    **编译 / 构建这一关**（compiled、exit_code、diff）、**测试那一关**
    （test、test_passed、test_exit_code）。**这轮算不算修好**看 passed。

    one_shot 说的是「没人先看到坏状态」：模型没编译过就直接改，改完第一次验
    就过了。它照样是一轮（7.8 起），状态是「一轮通过」。
    """

    number: int
    path: str
    edit: str  # edit_file 的修改结果摘要：「第 4 行（原文替换，命中 1 处）」
    fingerprint: str  # 这次改动的指纹：同一个地方改成同样的东西，指纹就一样
    verifier: str  # 验证命令：「compile main.cpp」
    exit_code: int | None  # 编译 / 构建那一关的退出码
    diff: Comparison | None  # 和上一轮/上一次验证比的诊断差；没有基线时为 None
    compiled: bool  # 编译 / 构建这一关过没过
    test: str = "没跑"  # 「run_tests 通过（退出码 0）」这种写法
    test_passed: bool | None = None  # 没跑时为 None
    test_exit_code: int | None = None
    one_shot: bool = False  # 改完第一次验就过了（之前没人看到过坏状态）

    @property
    def passed(self) -> bool:
        """这一轮修好了没有：编译 / 构建过了，该跑的测试也过了。

        7.5 只看编译那一关；7.6 起「编译通过」只说明编译这一关过了——
        测试红着不算修好。
        """
        return self.compiled and self.test_passed is not False

    @property
    def state(self) -> str:
        """这一轮的最终状态，一句话：卡在哪一关、过没过。"""
        if self.one_shot:
            # 没人先看到坏状态，改完第一次验就过了。两关当然都过（没过就不会
            # 走到这儿），但这一轮最该记住的是「一轮就通过」
            return "一轮通过"
        if not self.compiled:
            return "编译没过"
        if self.test_passed is False:
            return "测试没过"
        if self.test_passed is None:
            return "编译通过"
        return "编译通过、测试通过"

    def as_line(self) -> str:
        """一行说完这一轮，报告里要用。"""
        outcome = "通过" if self.passed else "没通过"
        code = "" if self.exit_code is None else f"退出码 {self.exit_code} "
        return (
            f"第 {self.number} 轮 · {self.path} {self.edit} → "
            f"{self.verifier} {code}{outcome}；{self.test} · 指纹 {self.fingerprint}"
        )


class RepairLoop:
    """盯着 edit_file 之后的验证结果，决定「接着修」还是「停下来」。

    Agent 每执行完一次工具调用就问它一句（after_tool_call），模型每次想收尾
    也问它一句（review）。它自己不算裸文本：判断用的全是 Verifier 留下的
    结构化结果（agent.verify.Verification）。
    """

    def __init__(
        self,
        verifier: Verifier,
        max_rounds: int = DEFAULT_MAX_REPAIR_ROUNDS,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_repair_rounds 至少为 1")

        self.verifier = verifier
        self.max_rounds = max_rounds
        self.rounds: list[RepairRound] = []
        self.stopped: str | None = None  # 停下来的原因；一旦停下就一直停着
        self.pending = False  # 有一次改完没验过，还欠着没修好

        self._broken = False  # 验证器最近看到的状态是坏的
        self._path = ""  # 正在修的是哪个文件
        self._nudged = False  # 这一轮已经催过模型一次了
        self._digest: tuple[str, str] | None = None  # 上一轮验完时文件长什么样
        self._prints: list[str] = []  # 每一轮改动的指纹，用来认「又改了一遍」
        self._seen: tuple[str, Any] | None = None  # 上一轮卡在哪一关、看到什么
        self._answer = ""  # 模型不肯再动手时说的那句话，报告里带上

    # ------------------------------------------------------------------
    # 看：每次工具调用之后
    # ------------------------------------------------------------------

    def after_tool_call(
        self, tool: str, arguments: dict[str, Any], result: "ToolResult"
    ) -> str:
        """问它一句这轮修得怎么样。返回要接在工具结果后面的一段（没有就空串）。

        只看刚跑完的那次调用的结果——验证是 Verifier 在上一步做的，
        这里不重新跑任何工具。
        """
        if tool == EDIT:
            return self._after_edit(arguments, result)

        if tool in VERIFIERS:
            self._after_verifier()

        return ""

    def _after_verifier(self) -> None:
        """模型自己跑了一次编译 / 构建 / 测试。

        和修复轮次无关（没改代码就谈不上修），但有一个用处：它自己验到通过了，
        说明欠着的那次多半已经不欠了，别再催。
        """
        outcome = self.verifier.last
        if outcome is None:
            return

        if not outcome.passed:
            # 验证器刚看到的还是坏的。是不是在修另说，但肯定不能当作修好了
            self._broken = True
            return

        if self.pending:
            # 「通过了」得看验的是不是正在修的那个文件：模型编了个别的
            # 文件、那个文件恰好没问题，不能拿它把这边一笔勾销
            if self.verifier.covers(outcome.call, self._path):
                self._resolve()
        else:
            self._broken = False

    def _after_edit(self, arguments: dict[str, Any], result: "ToolResult") -> str:
        """模型改了一个文件。"""
        if not result.success:
            if self.pending:
                # 没改成（原文没找到、找到好几处、文件太大……）——这些都在
                # 落盘之前就拦下了，文件一个字节没动。正在修的时候，这属于
                # 「这一轮什么也没发生」：再这么转下去只是把同一个错误重放一遍
                self.halt(
                    f"edit_file 没有改成（{_short(result.error)}），文件没有变化"
                )
            return ""

        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return ""

        outcome = self.verifier.last
        if outcome is None:
            # 这次编辑没验成：没装配能验证它的工具、不是能编译 / 跑测试的文件、
            # 或者验证用的工具自己没跑起来。验都没验，谈不上自动修
            return ""

        return self._round(path, result.content, outcome, arguments)

    # ------------------------------------------------------------------
    # 定：这一轮算不算、要不要停
    # ------------------------------------------------------------------

    def _round(
        self,
        path: str,
        edit_text: str,
        outcome: Verification,
        arguments: dict[str, Any],
    ) -> str:
        """一次「改了 + 验了」的结果，在这里定成一轮、或者定成结束。"""
        digest = _digest(self.verifier.root, path)
        same_file = (
            bool(digest)
            and self._digest is not None
            and self._digest[0] == path
            and self._digest[1] == digest
        )
        fingerprint = _fingerprint(path, edit_text, arguments)
        # 同一个地方、改成同样的东西，又来了一遍：中间隔了几轮也算
        repeated = fingerprint in self._prints
        # 「连着两轮」是轮和轮比，不是拿这一轮跟上一次验证比：上一次可能是
        # 模型自己编出来的基线（它先看到错才来改的），和基线一样只能说明
        # 「没改动」，说明不了「这一轮白改了」
        seen = _seen_of(outcome)
        same_seen = (
            self._seen is not None
            and seen[0] != ""
            and self._seen[0] == seen[0]
            and _same_seeing(self._seen[1], seen[1])
        )
        was_broken = self._broken or self.pending

        self._path = path
        self._digest = (path, digest)
        self._prints.append(fingerprint)
        self._seen = seen

        # ok 而不是 passed：编译 / 构建过了还得看测试那一关过没过
        if outcome.ok:
            self._broken = False
            # 改一次就过也是一轮（7.8 起）：一次 edit_file 加它触发的那次验证，
            # 本来就是一回事。没先见过坏状态的那种标成 one_shot，状态写成
            # 「一轮通过」——不然记录里会出现「明明改好了，却是 0 轮」
            self._record(
                path, edit_text, outcome, fingerprint, one_shot=not was_broken
            )
            self._resolve()
            return (
                f"自动修复：第 {self.rounds[-1].number} 轮通过"
                f"（{_passed_text(outcome)}），修复结束。"
            )

        # 没通过：这一轮就是一次没成的修复
        self._record(path, edit_text, outcome, fingerprint)
        self.pending = True
        self._nudged = False
        return self._failure_note(outcome, same_file, repeated, same_seen)

    def _failure_note(
        self,
        outcome: Verification,
        same_file: bool,
        repeated: bool,
        same_seen: bool,
    ) -> str:
        """一次失败的修复之后：是接着修，还是到此为止。"""
        round_ = self.rounds[-1]

        if round_.number >= self.max_rounds:
            self.halt(
                f"已经连着 {round_.number} 轮没修好"
                f"（上限 {self.max_rounds} 轮）"
            )
            return ""

        if same_file:
            self.halt("这一轮改完文件和上一轮一模一样，诊断没有任何变化")
            return ""

        if repeated:
            # 同一次改动又做了一遍：多半是在两个写法之间来回横跳
            # （A 改成 B、又改回 A），再转下去还是这两下
            self.halt("检测到重复修改（同一个地方又改成一样的内容），停止继续尝试")
            return ""

        if same_seen:
            # 上一轮验完是什么样，现在还是什么样：再催下去，等来的多半
            # 还是同一段输出
            what = "诊断" if self._seen is not None and self._seen[0] == "编译" else "测试输出"
            self.halt(f"连着两轮的{what}完全相同，再修也是原地打转")
            return ""

        stuck = "，卡在测试上" if outcome.passed else ""
        return (
            f"自动修复：第 {round_.number}/{self.max_rounds} 轮没通过{stuck}，接着修"
            f"（改完会自动再验一次）。"
        )

    def _record(
        self,
        path: str,
        edit_text: str,
        outcome: Verification,
        fingerprint: str,
        one_shot: bool = False,
    ) -> None:
        tests = outcome.tests
        self.rounds.append(
            RepairRound(
                one_shot=one_shot,
                number=len(self.rounds) + 1,
                path=path,
                edit=_edit_summary(edit_text),
                fingerprint=fingerprint[:8],
                verifier=outcome.call_text,
                exit_code=outcome.exit_code,
                diff=outcome.diff,
                compiled=outcome.passed,
                test=_test_line(tests),
                test_passed=None if tests is None else tests.passed,
                test_exit_code=None if tests is None else tests.exit_code,
            )
        )

    def _resolve(self) -> None:
        """验收过了：不欠着了。"""
        self.pending = False
        self._nudged = False

    # ------------------------------------------------------------------
    # 答：主循环问它下一步
    # ------------------------------------------------------------------

    def review(self, answer: str | None = None) -> Decision:
        """主循环问：现在能收尾吗。

        answer 是模型刚给出的最终回答；模型这一轮在要求调工具时传 None。
        """
        if self.stopped is not None:
            return Decision(REPORT, self.report())

        if not self.pending:
            # 没有欠着没修好的东西：一次普通的问答 / 工具流程，别多事
            return Decision(CONTINUE if answer is None else FINISH)

        if answer is None:
            return Decision(CONTINUE)

        # 模型想收尾，可最近一次修改还没验过
        if self._nudged:
            # 催过一次了，它还是只回话不动手：再催就是复读
            self._answer = answer
            self.halt("模型没有再修改代码")
            return Decision(REPORT, self.report())

        self._nudged = True
        return Decision(NUDGE, self._nudge_text())

    def _nudge_text(self) -> str:
        """把模型按回去接着修的那句话。"""
        round_ = self.rounds[-1]
        where = "编译 / 构建那一关" if not round_.compiled else "测试那一关"
        return (
            f"刚才那次修改没通过验证（卡在{where}），先别收尾：自动修复已经用掉 "
            f"{round_.number}/{self.max_rounds} 轮，还没修好。\n"
            f"把上面那条工具结果里的诊断当新情况看："
            f"还在的错是没解决的，新增的错是新线索，已消失的别再动"
            f"（测试没过的话，测试输出也在那条结果里）。\n"
            f"接着改就行，改完会自动再验一次；确实改不动，就把原因说清楚。"
        )

    # ------------------------------------------------------------------
    # 停
    # ------------------------------------------------------------------

    def halt(self, reason: str) -> None:
        """停下来。已经停过就不动了——先停的那个原因才是真原因。"""
        if self.stopped is not None:
            return
        self.stopped = reason
        # 停下的那一刻起就不再欠着「下一轮」了；但要接着修的话，
        # 最后一次修改原样留着，没有回滚
        self.pending = False

    def summary(self) -> str:
        """一句话说完这次修复：几轮、每轮卡在哪一关、最后成没成。

        例：「修复 2 轮；第 1 轮编译没过；第 2 轮编译通过、测试通过；最终成功。」
        改一次就过的那种念作「修复 1 轮；第 1 轮就通过；最终成功。」
        一次 edit_file 都没有（纯问答、只读工具）时返回空串——那种流程里
        不该多出这一句。
        """
        if not self.rounds:
            return ""

        parts = [f"修复 {len(self.rounds)} 轮"]
        parts += [_round_text(round_) for round_ in self.rounds]
        if self.stopped is not None:
            parts.append(f"最终没修好（{self.stopped}）")
        elif self.pending:
            parts.append("最终还没修好")
        else:
            parts.append("最终成功")
        return "；".join(parts) + "。"

    def report(self) -> str:
        """停下来时给用户的那段：为什么停、各轮都干了什么、代码现在什么样。"""
        lines = []
        summary = self.summary()
        if summary:
            lines.append(summary)
        lines.append(f"自动修复停止：{self.stopped}。")

        if self._path:
            lines.append(
                f"代码保留最后一次修改后的样子（{self._path}），没有回滚。"
            )

        if self.rounds:
            lines.append("各轮记录：")
            lines += [f"  {round_.as_line()}" for round_ in self.rounds]
            last = self.rounds[-1]
            if last.diff is not None:
                lines.append("最后一轮的诊断对比：")
                lines += diff_lines(last.diff)

        if self._answer:
            lines.append(f"模型的说明：{self._answer}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _same(before: Report, after: Report) -> bool:
    """两批诊断是不是同一批（认的是「同一条」，不是「同一行」）。"""
    diff = compare(before, after)
    return not diff.gone and not diff.added


def _seen_of(outcome: Verification) -> tuple[str, Any]:
    """这一轮卡在哪一关、那一关看到的是什么。

    - 编译 / 构建没过：看到的是那一批诊断（拿 Report 比，认「同一条」）；
    - 编译过了、测试没过：看到的是那次测试的输出（内容一样才算一样）；
    - 都过了：没有「卡住」这回事，两样都返回空的。
    """
    if not outcome.passed:
        return "编译", outcome.report

    tests = outcome.tests
    if tests is not None and not tests.passed:
        return "测试", _test_signature(tests)
    return "", None


def _same_seeing(before: Any, after: Any) -> bool:
    """两轮看到的东西是不是同一个：诊断按「同一条」认，测试输出按内容认。"""
    if isinstance(before, Report) and isinstance(after, Report):
        return _same(before, after)
    return before == after


def _test_signature(tests: Verification) -> str:
    """一次测试输出的指纹，用来认「连着两轮测试红得一模一样」。

    「Ran 1 test in 0.003s」这一行每次都不同（耗时），留着它两轮永远算不一样，
    这条停止条件就等于没有——所以按行剔掉那种汇总行再算。
    """
    lines = [
        line.rstrip()
        for line in tests.text.replace("\r\n", "\n").split("\n")
        if not line.startswith("Ran ")
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _fingerprint(path: str, edit_text: str, arguments: dict[str, Any]) -> str:
    """这次改动长什么样的指纹：改了哪个文件、动了哪几行、换成了什么。

    有它才认得出「同一次改动又做了一遍」——模型在两处之间来回横跳时
    （A 改成 B、又改回 A），指纹会重回来。位置取 edit_file 自己报的那个
    （它算出来的行才是真改到的那几行），内容取 new_text：**同一个地方、
    改成同样的东西就算同一次修改**，哪怕两次调用带的上下文不一样。
    """
    content = arguments.get("new_text")
    if not isinstance(content, str):
        content = ""
    # 只把 \r\n 归一成 \n，别的都别动：缩进、结尾有没有换行都可能是改动本身，
    # 归一多了会把「不一样的两次改动」认成同一次
    body = f"{path}\n{_position(edit_text)}\n{content.replace('\r\n', '\n')}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _position(text: str) -> str:
    """edit_file 报的改动位置：「第 9–11 行（原文替换，命中 1 处）」。

    拿不到就返回空串——指纹会退化成「同一个文件 + 同样的内容」，也还能用。
    """
    for line in text.split("\n"):
        if line.startswith(_POSITION):
            return line[len(_POSITION) :].strip()
    return ""


def _test_line(tests: Verification | None) -> str:
    """测试那一关的一句话：跑没跑、过没过、退出码多少。"""
    if tests is None:
        return "测试没跑"
    verdict = "通过" if tests.passed else "没通过"
    code = "" if tests.exit_code is None else f"（退出码 {tests.exit_code}）"
    return f"{tests.call_text} {verdict}{code}"


def _passed_text(outcome: Verification) -> str:
    """这一轮过了的那几关都报一下：「compile main.cpp；run_tests 通过」。"""
    text = outcome.call_text
    tests = outcome.tests
    if tests is not None:
        code = "" if tests.exit_code is None else f"（退出码 {tests.exit_code}）"
        text += f"；{tests.call_text} 通过{code}"
    return text


def _round_text(round_: RepairRound) -> str:
    """摘要里怎么念这一轮。

    「一轮通过」那种不能直接接在「第 1 轮」后面（会读成「第 1 轮一轮通过」），
    这里换成「第 1 轮就通过」——同一个意思，读着顺。别的照旧：
    「第 2 轮编译没过」。
    """
    if round_.one_shot:
        return f"第 {round_.number} 轮就通过"
    return f"第 {round_.number} 轮{round_.state}"


def _digest(root: Path, path: str) -> str:
    """文件内容的一个指纹；读不到（文件没了、路径不对）就返回空串。

    用来认「改了等于没改」：原文和目标一模一样时，edit_file 会照常报成功，
    文件也真的被重写了一遍——可内容和改动前一个字节不差，诊断当然一条都不会变。
    这种情况不能一轮一轮地转下去。
    """
    try:
        return hashlib.sha256(resolve_in_root(path, root).read_bytes()).hexdigest()
    except (OSError, ToolError):
        return ""


def _edit_summary(text: str) -> str:
    """从 edit_file 的结果里挑出一行当摘要：「第 4 行（原文替换，命中 1 处）」。"""
    return _position(text) or _short(text, 40)


def _short(text: str, limit: int = 80) -> str:
    """一整段压成一行短话，报告和停止原因里用。"""
    line = text.strip().split("\n", 1)[0].strip()
    return line if len(line) <= limit else f"{line[: limit - 1]}…"
