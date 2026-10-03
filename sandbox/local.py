"""LocalSandbox：在宿主机上执行命令，cwd 锁在 workspace 以内。

执行本身一行没自己写，全部复用 tools/run_command.py 的执行核心：
run_process（起进程 + 超时杀掉，那两个 Windows 的坑连同注释都在那边）、
decode_output（解码回退）、render_*（文本格式，as_text 用）。Sandbox
要保证的语义就是那边已经测过的语义，不另起一套。

这个文件真正自己做的是**边界**：cwd 可以是 workspace 里的任意一层
子目录，但 resolve() 之后必须仍落在 workspace 以内——`..`、绝对路径、
符号链接三种逃逸都靠「先展开再判断」一起堵住（复用 tools/paths.py 的
resolve_in_root，和只读工具是同一套规则）。
"""

from pathlib import Path

from sandbox.base import CommandResult, Sandbox
from tools.base import ToolError
from tools.paths import check_root, resolve_in_root
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
        if not isinstance(command, str) or not command.strip():
            # 空命令跑出来只会是「退出码 0、什么也没输出」，直接说清楚
            raise ToolError("命令不能为空")

        seconds = self.timeout if timeout is None else timeout
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ToolError(f"timeout 必须是数字，收到 {seconds!r}")
        if seconds <= 0:
            raise ToolError(f"timeout 必须大于 0，收到 {seconds}")

        target = self._checked_cwd(cwd)

        code, out_raw, err_raw = run_process(
            command, cwd=target, timeout=float(seconds), env=env
        )
        timed_out = code is None
        return CommandResult(
            command=command,
            cwd=target,
            exit_code=code,
            stdout="" if timed_out else decode_output(out_raw, "标准输出"),
            stderr="" if timed_out else decode_output(err_raw, "标准错误"),
            timeout=float(seconds),
            timed_out=timed_out,
        )

    def _checked_cwd(self, cwd: str | Path | None) -> Path:
        """把 cwd 解析成绝对路径，并确认没有跑出 workspace。

        省略 = workspace 根目录；相对路径从 workspace 算起；绝对路径
        也接受，但一样要落在 workspace 以内。resolve() 会展开 `..` 和
        符号链接，判断放在它之后，逃逸就堵死了——顺序反了就等于没查。
        """
        if cwd is None:
            return self._workspace

        try:
            resolved = resolve_in_root(str(cwd), self._workspace)
        except ToolError as exc:
            raise ToolError(f"cwd 超出 workspace 范围：{exc}") from None

        if not resolved.is_dir():
            raise ToolError(f"cwd 不存在或不是目录：{cwd!r}")
        return resolved
