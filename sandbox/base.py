"""Sandbox 抽象：把「执行一条命令」统一成一个接口和一种结果。

在此之前，命令是谁要跑谁自己去起进程（执行细节都在 RunCommandTool
里）。这个包把它收拢到一个接缝上：

- **接口**：Sandbox 只承诺一件事——在 workspace 以内执行一条命令，
  返回结构化的 CommandResult（命令、工作目录、退出码、两路输出、
  超时状态）。
- **实现**：现在只有 LocalSandbox（在宿主机上跑，见 local.py）；
  将来要在容器里跑就再加一个实现，调用方认的是接口，不是实现。
- **边界**：cwd 必须落在 workspace 以内，这条检查归 Sandbox 层管，
  不散给调用方——不管哪种实现，逃逸都在这一层挡住。

结果的文本形式（CommandResult.as_text）和 RunCommandTool 的输出逐字
一致：格式定义在 tools/run_command.py 的 render_result / render_timeout，
这里只调用、不重写——截断和超时文案全项目只有一个出处。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from tools.run_command import render_result, render_timeout


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
