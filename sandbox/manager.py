"""SandboxManager：决定这条命令交给哪种 sandbox 跑。

现在有两种实现：local（宿主机，默认）和 docker（一次性容器，必须显式
给 image）。按名字选实现、名字不认识就报错并列出可选项，和 index 的
create_embedder 是同一套写法；sandbox 也是懒建的——配置不对
（workspace 不存在、Docker daemon 没起、镜像不在本地）推迟到第一次
真要跑命令时才报错。

加一种后端 = _KINDS 里加一行（kind 名字 -> 实现类），要额外选项的话
再在 _build 里加一支。
"""

from pathlib import Path
from typing import Protocol

from sandbox.base import CommandResult, Sandbox
from sandbox.docker import DockerSandbox
from sandbox.local import LocalSandbox
from tools.base import ToolError
from tools.run_command import DEFAULT_TIMEOUT

DEFAULT_KIND = "local"

DOCKER_KIND = DockerSandbox.kind

class _SandboxFactory(Protocol):
    """造一个 sandbox 的可调用对象（就是下面表里那些实现类）。

    写成「工厂」而不是 `type[Sandbox]`：Sandbox 是抽象基类，它的 __init__
    签名不属于接口的一部分——各后端自己收选项（DockerSandbox 还要 image），
    表里放的其实是「怎么造」，不是「Sandbox 的子类」。
    """

    def __call__(
        self, workspace: Path | None = None, *, timeout: float = ...
    ) -> Sandbox: ...


# kind 名字 -> 实现类。加新后端就是在这里加一行。
_KINDS: dict[str, _SandboxFactory] = {
    LocalSandbox.kind: LocalSandbox,
    DOCKER_KIND: DockerSandbox,
}

KINDS = tuple(_KINDS)


class SandboxManager:
    """按 kind 造 sandbox；默认 local，kind="docker" 时要给 image。"""

    def __init__(
        self,
        workspace: Path | None = None,
        *,
        kind: str = DEFAULT_KIND,
        timeout: float = DEFAULT_TIMEOUT,
        image: str | None = None,
    ) -> None:
        if kind not in _KINDS:
            raise ToolError(
                f"不认识的 sandbox 类型 {kind!r}，可用：{'、'.join(KINDS)}"
            )
        if kind == DOCKER_KIND and not (
            isinstance(image, str) and image.strip()
        ):
            # 配置缺了早报一步，别等第一次跑命令才发现（DockerSandbox
            # 自己也会挡一道，这里是为了报得更早）
            raise ToolError(
                "docker sandbox 必须指定 image（不会自动拉取），例如 "
                "SandboxManager(kind='docker', image='python:3.12-slim')"
            )

        self.kind = kind
        self.workspace = workspace
        self.timeout = timeout
        self.image = image
        self._sandbox: Sandbox | None = None

    @property
    def sandbox(self) -> Sandbox:
        """当前的 sandbox 实例：第一次要用时才建，之后一直用它。"""
        if self._sandbox is None:
            self._sandbox = self._build()
        return self._sandbox

    def _build(self) -> Sandbox:
        """按 kind 装配实例。只有要额外选项的后端才在这儿单独一支。"""
        if self.kind == DOCKER_KIND:
            return DockerSandbox(
                self.workspace, image=self.image, timeout=self.timeout
            )
        return _KINDS[self.kind](self.workspace, timeout=self.timeout)

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
