"""Sandbox 抽象：把「执行一条命令」统一成一个接口和一种结果。

在此之前，命令是谁要跑谁自己去起进程（执行细节都在 RunCommandTool
里）。这个包把它收拢到一个接缝上：

- **接口**：Sandbox 只承诺一件事——在 workspace 以内执行一条命令，
  返回结构化的 CommandResult（命令、工作目录、退出码、两路输出、
  超时状态）。
- **实现**：LocalSandbox（在宿主机上跑，见 local.py）和 DockerSandbox
  （每次执行一个一次性容器，见 docker.py），调用方认的是接口。
- **边界**：cwd 必须落在 workspace 以内，这条检查归 Sandbox 层管，
  不散给调用方——check_run_args / resolve_cwd 就是那一份，两种实现
  共用它，所以「逃逸挡得住」对哪种后端都成立。

结果的文本形式（CommandResult.as_text）和 RunCommandTool 的输出逐字
一致：格式定义在 tools/run_command.py 的 render_result / render_timeout，
这里只调用、不重写——截断和超时文案全项目只有一个出处。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from tools.base import ToolError
from tools.paths import resolve_in_root
from tools.run_command import render_result, render_timeout


def check_run_args(
    command: str, timeout: float | None, default_timeout: float
) -> float:
    """run() 的公共校验：命令非空、timeout 是正数。返回这次实际用的秒数。

    说法和 RunCommandTool 划一：调用方（还有模型）从工具那边换过来时，
    报错措辞对得上，认知不用改。
    """
    if not isinstance(command, str) or not command.strip():
        # 空命令跑出来只会是「退出码 0、什么也没输出」，直接说清楚
        raise ToolError("命令不能为空")

    seconds = default_timeout if timeout is None else timeout
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ToolError(f"timeout 必须是数字，收到 {seconds!r}")
    if seconds <= 0:
        raise ToolError(f"timeout 必须大于 0，收到 {seconds}")
    return float(seconds)


def resolve_cwd(cwd: str | Path | None, workspace: Path) -> Path:
    """把 cwd 解析成绝对路径，并确认没有跑出 workspace。

    省略 = workspace 根目录；相对路径从 workspace 算起；绝对路径也接受，
    但一样要落在 workspace 以内。resolve() 会展开 `..` 和符号链接，判断
    放在它之后，逃逸就堵死了——顺序反了就等于没查。所有实现共用这一份。
    """
    if cwd is None:
        return workspace

    try:
        resolved = resolve_in_root(str(cwd), workspace)
    except ToolError as exc:
        raise ToolError(f"cwd 超出 workspace 范围：{exc}") from None

    if not resolved.is_dir():
        raise ToolError(f"cwd 不存在或不是目录：{cwd!r}")
    return resolved


@dataclass(frozen=True)
class CommandResult:
    """一次命令执行的完整结果。

    超时是「正常结果」不是异常：timed_out 为 True、exit_code 为 None，
    stdout/stderr 是空串（超时的输出不保留，和工具一贯的语义一致）。

    stdout/stderr 是解码后的**完整**文本（编码认不出来时退化成替换字符，
    不抛错）；截断只发生在 as_text() 里——结构化字段给的是数据，
    不是给人看的成品。
    """

    command: str
    cwd: Path
    exit_code: int | None  # None = 超时被杀，没有退出码可言
    stdout: str
    stderr: str
    timeout: float  # 这次实际用的上限（秒）
    timed_out: bool

    @property
    def ok(self) -> bool:
        """命令跑完了而且退出码是 0。超时（exit_code 为 None）不算成功。"""
        return self.exit_code == 0

    def as_text(self) -> str:
        """和 RunCommandTool 完全相同的文本形式（含截断与超时文案）。"""
        if self.timed_out or self.exit_code is None:
            return render_timeout(self.command, self.timeout)
        return render_result(self.command, self.exit_code, self.stdout, self.stderr)


class Sandbox(ABC):
    """执行命令的统一接口。

    子类声明自己的 kind（SandboxManager 按名字选实现要用），并实现
    workspace 和 run。
    """

    kind: str = ""

    @property
    @abstractmethod
    def workspace(self) -> Path:
        """命令能碰到的最外层目录（绝对路径）。"""

    @abstractmethod
    def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """在 workspace 以内执行一条命令。

        cwd 省略 = workspace 根目录；给了就必须落在 workspace 以内，
        越界（`..`、绝对路径指向外面）抛 ToolError。
        timeout 省略 = 这个 sandbox 的默认值；单次调用可以覆盖。
        env 只决定子进程收到哪些环境变量，不参与命令字符串的拼接。
        """
