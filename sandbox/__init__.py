"""Sandbox：执行命令的统一接口（Phase 10）。

- base：Sandbox 抽象 + CommandResult 结果结构（统一 command / cwd /
  workspace / timeout / stdout / stderr / exit_code / 超时状态），
  以及两种后端共用的校验（命令非空、timeout 合法、cwd 边界）。
- local：LocalSandbox —— 在宿主机上执行，cwd 锁在 workspace 以内。
- docker：DockerSandbox —— 每次执行一个一次性容器（只用 Docker CLI，
  不引 SDK），workspace 以 bind mount 挂在 /workspace，不自动 pull。
- manager：SandboxManager —— 按名字选实现，默认 local，docker 要 image。

10.2 起执行类工具（RunCommandTool / CompileTool / BuildTool /
TestTool）都从这里走：各自按 root 建一个 Sandbox 再执行，工作目录锁在
root 以内；要用哪个后端由装配处决定（工具只认 Sandbox 接口）。
测试在 tests/test_sandbox.py（接口与边界）、
tests/test_sandbox_integration.py（工具确实经过这里）和
tests/test_docker_sandbox.py（docker 后端，不需要真装 Docker）。
"""

from sandbox.base import CommandResult, Sandbox
from sandbox.docker import DockerSandbox
from sandbox.local import LocalSandbox
from sandbox.manager import DEFAULT_KIND, KINDS, SandboxManager

__all__ = [
    "CommandResult",
    "Sandbox",
    "LocalSandbox",
    "DockerSandbox",
    "SandboxManager",
    "DEFAULT_KIND",
    "KINDS",
]
