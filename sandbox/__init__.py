"""Sandbox：执行命令的统一接口（Phase 10）。

- base：Sandbox 抽象 + CommandResult 结果结构（统一 command / cwd /
  workspace / timeout / stdout / stderr / exit_code / 超时状态）。
- local：LocalSandbox —— 在宿主机上执行，cwd 锁在 workspace 以内。
- manager：SandboxManager —— 按名字选实现，现在默认 local。

这一阶段只把接口和行为钉住（测试在 tests/test_sandbox.py）；把工具
接上来（RunCommandTool 等改走这里）是下一阶段的事。
"""

from sandbox.base import CommandResult, Sandbox
from sandbox.local import LocalSandbox
from sandbox.manager import DEFAULT_KIND, KINDS, SandboxManager

__all__ = [
    "CommandResult",
    "Sandbox",
    "LocalSandbox",
    "SandboxManager",
    "DEFAULT_KIND",
    "KINDS",
]
