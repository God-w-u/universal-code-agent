"""LocalSandbox：在宿主机上执行命令，cwd 锁在 workspace 以内。

执行本身一行没自己写，全部复用 tools/run_command.py 的执行核心：
run_process（起进程 + 超时杀掉，那两个 Windows 的坑连同注释都在那边）、
decode_output（解码回退）、render_*（文本格式，as_text 用）。Sandbox
要保证的语义就是那边已经测过的语义，不另起一套。

校验和边界也一行没自己写：命令非空、timeout 合法、cwd 必须落在
workspace 以内（`..`、绝对路径、符号链接一起堵）都在 sandbox/base.py，
local 和 docker 两种实现共用同一份——边界归 Sandbox 层，不是各实现
自己约定。
"""

from pathlib import Path

from sandbox.base import CommandResult, Sandbox, check_run_args, resolve_cwd
from tools.paths import check_root
from tools.run_command import DEFAULT_TIMEOUT, decode_output, run_process


class LocalSandbox(Sandbox):
    """在宿主机上执行命令。

    workspace 省略 = 项目根目录（和其他工具一样，先过 check_root 的
    危险目录检查）；timeout 是这个 sandbox 的默认上限，单次 run 可覆盖。
    """

    kind = "local"

    def __init__(
        self,
        workspace: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout 必须大于 0，收到 {timeout}")
        self._workspace = check_root(workspace)
        self.timeout = timeout

    @property
    def workspace(self) -> Path:
        return self._workspace

    def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """在 workspace 以内执行一条命令。

        参数不合法（空命令、timeout <= 0）和 RunCommandTool 同一个说法，
        都抛 ToolError——接入时调用方就是从那些工具换过来的，报错措辞
        对得上，模型的认知不用改。
        """
        seconds = check_run_args(command, timeout, self.timeout)
        target = resolve_cwd(cwd, self._workspace)

        code, out_raw, err_raw = run_process(
            command, cwd=target, timeout=seconds, env=env
        )
        timed_out = code is None
        return CommandResult(
            command=command,
            cwd=target,
            exit_code=code,
            stdout="" if timed_out else decode_output(out_raw, "标准输出"),
            stderr="" if timed_out else decode_output(err_raw, "标准错误"),
            timeout=seconds,
            timed_out=timed_out,
        )
