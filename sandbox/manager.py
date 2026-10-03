"""SandboxManager：决定这条命令交给哪种 sandbox 跑。

现在只有一种实现（local），管理器看着有点像摆设——它的位置是给下一
步留的：接 Docker 时只在 _KINDS 里加一行（kind 名字 -> 实现类），
调用方（工具、Agent Loop）拿到的还是同一个 Sandbox 接口，不用改。

按名字选实现、名字不认识就报错并列出可选项，和 index 的
create_embedder 是同一套写法；sandbox 也是懒建的——配置不对
（比如 workspace 不存在）推迟到第一次真要跑命令时才报错。
"""

from pathlib import Path

from sandbox.base import CommandResult, Sandbox
from sandbox.local import LocalSandbox
from tools.base import ToolError
from tools.run_command import DEFAULT_TIMEOUT

DEFAULT_KIND = "local"

# kind 名字 -> 实现类。加 docker 就是在这里加一行。
_KINDS: dict[str, type[Sandbox]] = {
    LocalSandbox.kind: LocalSandbox,
}

KINDS = tuple(_KINDS)


class SandboxManager:
    """按 kind 造 sandbox；现在默认（也是唯一）的是 local。"""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        kind: str = DEFAULT_KIND,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if kind not in _KINDS:
            raise ToolError(
                f"不认识的 sandbox 类型 {kind!r}，可用：{'、'.join(KINDS)}"
            )
        self.kind = kind
        self.workspace = workspace
        self.timeout = timeout
        self._sandbox: Sandbox | None = None

    @property
    def sandbox(self) -> Sandbox:
        """当前的 sandbox 实例：第一次要用时才建，之后一直用它。"""
        if self._sandbox is None:
            self._sandbox = _KINDS[self.kind](self.workspace, timeout=self.timeout)
        return self._sandbox

    def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """转发给当前 sandbox，调用方不用先拿实例。"""
        return self.sandbox.run(command, cwd=cwd, timeout=timeout, env=env)
