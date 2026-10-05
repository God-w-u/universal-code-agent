"""DockerSandbox（Phase 10.3）测试：命令构造、映射、清理、错误处理。

这一份**不需要装 Docker**：docker CLI 被两层假的东西顶替——
- `sandbox.docker.shutil.which` 被 patch 成「有 docker」（构造时的检查）；
- `sandbox.docker.run_process` 换成一个假的 docker 命令行，它记下每条
  命令、按剧本回结果。
测的仍然是真代码路径：命令怎么拼、结果怎么翻、超时后清理什么，一条不落；
只是没有一个字节真的进 Docker。

真的容器只在 DockerSmokeTest 里跑：本机有 docker、daemon 在、而且本地
已经有候选镜像（或 UAC_TEST_DOCKER_IMAGE 指定的镜像）才跑——不 pull 任何
东西，条件不满足整组跳过。

全部离线（smoke 那组也不联网）。
"""

import base64
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import PROJECT_ROOT
from sandbox import KINDS, DockerSandbox, LocalSandbox, SandboxManager
from sandbox.base import resolve_cwd
from sandbox.docker import CONTAINER_ROOT, NAME_PREFIX
from tools.base import ToolError
from tools.run_command import run_process

IMAGE = "python:3.12-slim"


def run_calls(docker) -> list[dict]:
    """把假 docker 收到的所有 `docker run` 调用挑出来。"""
    return [call for call in docker.calls if call["command"].startswith("docker run")]


def container_name(command: str) -> str:
    """从一条 docker run 命令里把 --name 后面那串抠出来。"""
    return command.split("--name ", 1)[1].split(" ", 1)[0]


def decoded_payload(command: str) -> str:
    """把 docker 命令里 base64 那段抠出来解码，还原成原始命令。"""
    encoded = command.split("echo ", 1)[1].split(" | base64 -d | sh", 1)[0]
    return base64.b64decode(encoded).decode("utf-8")


class FakeDockerCli:
    """假的 docker 命令行：记录每次调用，按剧本返回结果。

    剧本字段名对应三条路：version / image inspect（两次探测）、
    run_*（真正执行那条）、run_timed_out（把它当超时）。
    """

    def __init__(
        self,
        *,
        version_code: int = 0,
        version_stderr: bytes = b"",
        image_code: int = 0,
        image_stderr: bytes = b"",
        run_code: int = 0,
        run_stdout: bytes = b"",
        run_stderr: bytes = b"",
        run_timed_out: bool = False,
    ) -> None:
        self.version_code = version_code
        self.version_stderr = version_stderr
        self.image_code = image_code
        self.image_stderr = image_stderr
        self.run_code = run_code
        self.run_stdout = run_stdout
        self.run_stderr = run_stderr
        self.run_timed_out = run_timed_out
        self.calls: list[dict] = []

    def __call__(self, command, *, cwd, timeout, env=None):
        self.calls.append(
            {"command": command, "cwd": cwd, "timeout": timeout, "env": env}
        )
        if command == "docker version":
            return self.version_code, b"", self.version_stderr
        if command.startswith("docker image inspect"):
            return self.image_code, b"", self.image_stderr
        if command.startswith("docker rm -f"):
            return 0, b"", b""
        if self.run_timed_out:
            return None, b"", b""
        return self.run_code, self.run_stdout, self.run_stderr


class DockerTestBase(unittest.TestCase):
    """临时 workspace + 假 docker（which 和 run_process 两层都换掉）。"""

    image = IMAGE

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

        # 构造时的可用性检查：假装 docker 命令在
        which = patch("sandbox.docker.shutil.which", return_value="docker")
        which.start()
        self.addCleanup(which.stop)

        # 执行路径：所有 docker CLI 调用都进假命令行
        self.docker = FakeDockerCli()
        cli = patch("sandbox.docker.run_process", new=self.docker)
        cli.start()
        self.addCleanup(cli.stop)

        self.sandbox = DockerSandbox(self.root, image=self.image)

    def build(self, **kwargs) -> DockerSandbox:
        return DockerSandbox(self.root, image=self.image, **kwargs)


class DockerCommandShapeTest(DockerTestBase):
    """命令构造：只看拼出来的字符串，一步都不执行。"""

    def command(self, raw="echo hi", cwd=None, env=None) -> str:
        return self.sandbox._docker_command(
            raw, cwd or self.root, env, "uac-sandbox-test"
        )

    def test_it_is_a_docker_run(self):
        self.assertEqual(self.command().split()[:2], ["docker", "run"])

    def test_the_container_is_temporary_and_never_pulls(self):
        text = self.command()

        self.assertIn("--rm", text)
        self.assertIn("--pull=never", text)

    def test_the_image_and_the_name_are_in_the_command(self):
        text = self.command()

        self.assertIn(self.image, text)
        self.assertIn("--name uac-sandbox-test", text)

    def test_the_workspace_is_mounted_at_the_container_root(self):
        self.assertIn(
            f'-v "{self.root.as_posix()}:{CONTAINER_ROOT}"', self.command()
        )

    def test_only_the_workspace_is_mounted(self):
        # 挂进去的只有 workspace 一个目录，没有第二个 -v
        self.assertEqual(self.command().count(" -v "), 1)

    def test_the_container_cwd_starts_at_the_root(self):
        self.assertIn(f"-w {CONTAINER_ROOT} ", self.command())

    def test_the_container_cwd_follows_the_host_cwd(self):
        (self.root / "src").mkdir()
        target = resolve_cwd("src", self.root)

        self.assertIn(f"-w {CONTAINER_ROOT}/src ", self.command(cwd=target))

    def test_env_becomes_sorted_e_flags(self):
        text = self.command(env={"B": "2", "A": "1"})

        self.assertIn('-e "A=1" -e "B=2"', text)

    def test_no_env_means_no_flags_at_all(self):
        # 宿主机环境不会自动倒进容器：不给 env，就一个 -e 都没有
        self.assertNotIn(" -e ", self.command())

    def test_the_raw_command_is_base64_encoded(self):
        # 命令本身不进命令行（引号会在 cmd / sh 之间互相咬），
        # 进去的是 base64，容器里解回原样
        raw = 'echo "带引号 和 空格"'

        self.assertEqual(decoded_payload(self.command(raw=raw)), raw)

    def test_quotes_and_newlines_survive_the_encoding(self):
        raw = '"C:/Program Files/python.exe" "a b.py"\n第二行'

        self.assertEqual(decoded_payload(self.command(raw=raw)), raw)

    def test_the_payload_runs_through_sh_in_the_container(self):
        self.assertIn('| base64 -d | sh"', self.command())


class DockerRunTest(DockerTestBase):
    """执行路径：结果翻译、探测只做一次、超时清理、参数校验。"""

    def test_a_normal_run_returns_the_result(self):
        self.docker.run_stdout = "跑通了".encode("utf-8")

        result = self.sandbox.run("echo hi")

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.ok)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.stdout, "跑通了")
        self.assertEqual(result.cwd, self.root)
        self.assertEqual(result.command, "echo hi")

    def test_the_text_reports_the_original_command_not_the_docker_line(self):
        result = self.sandbox.run("echo hi")

        text = result.as_text()
        self.assertIn("命令：echo hi", text)
        self.assertIn("退出码：0", text)
        self.assertNotIn("--pull=never", text)

    def test_a_non_zero_exit_code_is_a_result_not_an_exception(self):
        self.docker.run_code = 3
        self.docker.run_stderr = "容器里报错了".encode("utf-8")

        result = self.sandbox.run("编译一下")

        self.assertEqual(result.exit_code, 3)
        self.assertFalse(result.ok)
        self.assertIn("退出码：3", result.as_text())
        self.assertIn("容器里报错了", result.stderr)

    def test_the_probes_run_once_before_the_first_execution(self):
        self.sandbox.run("echo one")
        self.sandbox.run("echo two")

        commands = [call["command"] for call in self.docker.calls]
        self.assertEqual(commands[0], "docker version")
        self.assertIn("image inspect", commands[1])
        self.assertEqual(commands.count("docker version"), 1)
        self.assertEqual(len(run_calls(self.docker)), 2)

    def test_each_run_gets_its_own_container_name(self):
        self.sandbox.run("echo one")
        self.sandbox.run("echo two")

        names = [container_name(call["command"]) for call in run_calls(self.docker)]
        self.assertEqual(len(set(names)), 2)
        for name in names:
            self.assertTrue(name.startswith(NAME_PREFIX))

    def test_the_effective_timeout_reaches_the_docker_call(self):
        self.sandbox = self.build(timeout=9)

        self.sandbox.run("echo hi")
        self.assertEqual(run_calls(self.docker)[0]["timeout"], 9)

        self.sandbox.run("echo hi", timeout=2.5)
        self.assertEqual(run_calls(self.docker)[1]["timeout"], 2.5)

    def test_a_timeout_is_a_normal_result_with_no_output(self):
        self.docker.run_timed_out = True
        self.docker.run_stdout = "不该留下".encode("utf-8")

        result = self.sandbox.run("sleep 100")

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertFalse(result.ok)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertIn("超时", result.as_text())

    def test_a_timeout_kills_and_removes_that_container(self):
        self.docker.run_timed_out = True

        self.sandbox.run("sleep 100")

        run_command = run_calls(self.docker)[0]["command"]
        name = container_name(run_command)
        self.assertEqual(self.docker.calls[-1]["command"], f"docker rm -f {name}")

    def test_the_cleanup_happens_after_the_run(self):
        self.docker.run_timed_out = True

        self.sandbox.run("sleep 100")

        commands = [call["command"] for call in self.docker.calls]
        self.assertLess(
            next(i for i, c in enumerate(commands) if c.startswith("docker run")),
            next(i for i, c in enumerate(commands) if c.startswith("docker rm -f")),
        )

    def test_a_bad_command_or_timeout_never_reaches_docker(self):
        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("   ")
        self.assertIn("不能为空", str(caught.exception))

        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi", timeout=0)
        self.assertIn("必须大于 0", str(caught.exception))

        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", timeout="一会儿")

        self.assertEqual(self.docker.calls, [])

    def test_a_cwd_escape_is_rejected_before_docker(self):
        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi", cwd="..")

        self.assertIn("workspace", str(caught.exception))
        self.assertEqual(self.docker.calls, [])

    def test_an_absolute_escape_is_rejected_before_docker(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        outside = Path(other.name).resolve()

        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", cwd=str(outside))

        self.assertEqual(self.docker.calls, [])


class DockerErrorTest(DockerTestBase):
    """Docker 层面的错误：说清楚，而且绝不悄悄换成本机执行。"""

    def test_a_dead_daemon_is_a_clear_error(self):
        self.docker.version_code = 1
        self.docker.version_stderr = b"Cannot connect to the Docker daemon."

        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi")

        message = str(caught.exception)
        self.assertIn("daemon", message)
        self.assertIn("Cannot connect", message)
        self.assertEqual(run_calls(self.docker), [])

    def test_a_missing_image_is_a_clear_error(self):
        self.docker.image_code = 1
        self.docker.image_stderr = b"Error: No such image: python:3.12-slim"

        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi")

        message = str(caught.exception)
        self.assertIn(self.image, message)
        self.assertIn("不会自动 pull", message)
        self.assertEqual(run_calls(self.docker), [])

    def test_failures_never_fall_back_to_local(self):
        self.docker.version_code = 1

        with patch.object(LocalSandbox, "run") as local_run:
            with self.assertRaises(ToolError):
                self.sandbox.run("echo hi")

        local_run.assert_not_called()

    def test_a_failed_probe_is_not_remembered_forever(self):
        # daemon 刚起来 / 镜像刚拉下来，下一次 run 要重试而不是记仇
        self.docker.version_code = 1
        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi")

        self.docker.version_code = 0
        result = self.sandbox.run("echo hi")

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(run_calls(self.docker)), 1)


class DockerConstructionTest(unittest.TestCase):
    """构造时的检查：image 必须有、docker 命令必须找得到。"""

    def test_the_image_is_required(self):
        with patch("sandbox.docker.shutil.which", return_value="docker"):
            with self.assertRaises(ToolError) as caught:
                DockerSandbox(PROJECT_ROOT, image=None)
            with self.assertRaises(ToolError):
                DockerSandbox(PROJECT_ROOT, image="   ")

        self.assertIn("image", str(caught.exception))

    def test_the_image_is_keyword_only(self):
        # 「显式指定」就是字面意思：不给位置参数留后门
        with patch("sandbox.docker.shutil.which", return_value="docker"):
            with self.assertRaises(TypeError):
                DockerSandbox(PROJECT_ROOT, "python:3.12-slim")

    def test_a_missing_docker_cli_is_a_clear_error(self):
        with patch("sandbox.docker.shutil.which", return_value=None):
            with self.assertRaises(ToolError) as caught:
                DockerSandbox(PROJECT_ROOT, image=IMAGE)

        message = str(caught.exception)
        self.assertIn("docker", message)
        self.assertIn("PATH", message)

    def test_a_bad_timeout_is_rejected_like_local(self):
        with patch("sandbox.docker.shutil.which", return_value="docker"):
            with self.assertRaises(ValueError):
                DockerSandbox(PROJECT_ROOT, image=IMAGE, timeout=0)


class DockerManagerTest(DockerTestBase):
    """管理器：两种 kind 都认得，docker 必须带 image，默认还是 local。"""

    def test_both_kinds_are_selectable(self):
        self.assertEqual(KINDS, ("local", "docker"))

    def test_the_manager_builds_a_docker_sandbox(self):
        manager = SandboxManager(self.root, kind="docker", image=self.image)

        sandbox = manager.sandbox

        self.assertIsInstance(sandbox, DockerSandbox)
        self.assertEqual(sandbox.image, self.image)
        self.assertEqual(sandbox.workspace, self.root)
        self.assertEqual(sandbox.timeout, manager.timeout)

    def test_docker_without_an_image_is_refused_at_construction(self):
        with self.assertRaises(ToolError) as caught:
            SandboxManager(self.root, kind="docker")

        self.assertIn("image", str(caught.exception))

    def test_local_is_still_the_default(self):
        manager = SandboxManager(self.root)

        self.assertEqual(manager.kind, "local")
        self.assertIsInstance(manager.sandbox, LocalSandbox)

    def test_run_delegates_to_the_docker_sandbox(self):
        manager = SandboxManager(self.root, kind="docker", image=self.image)

        result = manager.run("echo hi")

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(len(run_calls(self.docker)), 1)


# 真实 smoke 用的候选镜像：本机有哪个用哪个，一个都不 pull。
# 想指定就设 UAC_TEST_DOCKER_IMAGE。
_SMOKE_CANDIDATES = (
    "alpine",
    "busybox",
    "debian:12",
    "ubuntu:24.04",
    "ubuntu:22.04",
    "python:3.12-slim",
    "python:3.11-slim",
)


def smoke_image() -> str | None:
    """本机本来就有、可以直接用的镜像；没有（或没 Docker）返回 None。

    这个探测只在导入测试模块时跑一次：没有 docker 就是一次 which，
    有 docker 也只是几条本地 inspect，不联网。
    """
    if shutil.which("docker") is None:
        return None

    code, _, _ = run_process("docker version", cwd=PROJECT_ROOT, timeout=20)
    if code != 0:
        return None

    override = os.environ.get("UAC_TEST_DOCKER_IMAGE")
    candidates = (override, *_SMOKE_CANDIDATES) if override else _SMOKE_CANDIDATES
    for image in candidates:
        code, _, _ = run_process(
            f'docker image inspect "{image}"', cwd=PROJECT_ROOT, timeout=20
        )
        if code == 0:
            return image
    return None


_SMOKE_IMAGE = smoke_image()


@unittest.skipUnless(
    _SMOKE_IMAGE, "本机没有可用的 Docker 或现成镜像（不 pull 任何东西）"
)
class DockerSmokeTest(unittest.TestCase):
    """真的在容器里跑一遍：本机有 docker + 有现成镜像才跑。

    走的就是产品路径：DockerSandbox → docker run → 容器里的 sh。
    """

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.sandbox = DockerSandbox(self.root, image=_SMOKE_IMAGE, timeout=60)

    def test_a_command_runs_in_the_container(self):
        result = self.sandbox.run("echo hello-from-container")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello-from-container", result.stdout)

    def test_the_container_cwd_maps_the_host_cwd(self):
        (self.root / "sub").mkdir()

        result = self.sandbox.run("pwd", cwd="sub")

        self.assertEqual(result.exit_code, 0)
        self.assertIn(f"{CONTAINER_ROOT}/sub", result.stdout)

    def test_the_workspace_is_writable_and_lands_on_the_host(self):
        result = self.sandbox.run("echo made-in-container > made.txt")

        self.assertEqual(result.exit_code, 0)
        self.assertTrue((self.root / "made.txt").is_file())
        self.assertIn(
            "made-in-container",
            (self.root / "made.txt").read_text(encoding="utf-8"),
        )

    def test_only_the_explicit_env_reaches_the_container(self):
        with patch.dict(os.environ, {"UAC_HOST_ONLY": "不该漏进去"}):
            result = self.sandbox.run(
                "echo [$UAC_SMOKE_MARK][$UAC_HOST_ONLY]",
                env={"UAC_SMOKE_MARK": "docker-ok"},
            )

        self.assertIn("[docker-ok][]", result.stdout)

    def test_a_non_zero_exit_code_comes_back(self):
        result = self.sandbox.run("exit 3")

        self.assertEqual(result.exit_code, 3)
        self.assertIn("退出码：3", result.as_text())

    def test_a_timeout_kills_and_cleans_up_the_container(self):
        result = self.sandbox.run("sleep 30", timeout=3)

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)

        code, out, _ = run_process(
            'docker ps -a --filter "name=uac-sandbox-" --format "{{.Names}}"',
            cwd=PROJECT_ROOT,
            timeout=30,
        )
        self.assertEqual(code, 0)
        self.assertNotIn(b"uac-sandbox-", out)

    def test_a_missing_image_errors_without_pulling(self):
        sandbox = DockerSandbox(
            self.root, image="uac-definitely-not-an-image:0", timeout=30
        )

        with self.assertRaises(ToolError) as caught:
            sandbox.run("echo hi")

        self.assertIn("不会自动 pull", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
