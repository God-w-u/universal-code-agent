"""Sandbox（Phase 10.1）测试：接口、边界检查、执行语义。

- **接口**：Sandbox 是抽象类；LocalSandbox 的 kind / workspace /
  默认 timeout；SandboxManager 按名字选实现，默认 local。
- **边界**：cwd 必须落在 workspace 以内——`..`、绝对路径两种逃逸都由
  Sandbox 层挡住，而且被挡住时命令一步都不许跑。
- **执行**：正常执行、cwd、workspace 内路径、stdout/stderr 分开、
  退出码、超时（超时是「正常结果」，不是异常）。
- **一致性**：同一条命令分别交给 RunCommandTool 和 LocalSandbox，
  输出**逐字**相同——这一阶段不接入工具，但语义必须是同一份
  （截断、超时文案、解码回退都算）。接入后哪天对不上，先炸的就是
  这几个用例。

超时那几条和 test_run_command.py 一样用项目根目录当 workspace：
被杀的只是 shell，真正睡着的子进程会一直活到睡醒，而 Windows 上
有进程占着的目录删不掉，临时目录清理会直接报 PermissionError。

全部在临时目录里跑本机解释器，不联网。
"""

import dataclasses
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from config import PROJECT_ROOT
from sandbox import (
    DEFAULT_KIND,
    KINDS,
    CommandResult,
    LocalSandbox,
    Sandbox,
    SandboxManager,
)
from tools.base import ToolError
from tools.run_command import DEFAULT_TIMEOUT, RunCommandTool


def slow_command(seconds: float = 5) -> str:
    """睡一会儿的命令，用来测超时。用 -c 不落文件，没有清理问题。"""
    return f'"{sys.executable}" -c "import time;time.sleep({seconds})"'


def fast_command() -> str:
    return f'"{sys.executable}" -c "print(1+1)"'


class SandboxTestBase(unittest.TestCase):
    """搭一个临时目录当 workspace，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.sandbox = LocalSandbox(self.root)

    def write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def python(self, *args: str) -> str:
        """用当前这个解释器跑命令，不依赖系统里装了什么。"""
        return f'"{sys.executable}" ' + " ".join(args)


class SandboxShapeTest(SandboxTestBase):
    """接口的形状：抽象类、kind、workspace、默认值、构造校验。"""

    def test_local_sandbox_is_a_sandbox(self):
        self.assertIsInstance(self.sandbox, Sandbox)
        self.assertEqual(LocalSandbox.kind, "local")

    def test_workspace_is_an_absolute_path(self):
        self.assertEqual(self.sandbox.workspace, self.root)
        self.assertTrue(self.sandbox.workspace.is_absolute())

    def test_default_workspace_is_the_project_root(self):
        self.assertEqual(LocalSandbox().workspace, PROJECT_ROOT)

    def test_the_default_timeout_comes_from_run_command(self):
        self.assertEqual(self.sandbox.timeout, DEFAULT_TIMEOUT)
        self.assertGreater(DEFAULT_TIMEOUT, 0)

    def test_sandbox_itself_is_abstract(self):
        with self.assertRaises(TypeError):
            Sandbox()

    def test_a_drive_root_workspace_is_rejected(self):
        with self.assertRaises(ToolError):
            LocalSandbox(Path("C:/"))

    def test_a_missing_workspace_is_rejected(self):
        with self.assertRaises(ToolError):
            LocalSandbox(Path("C:/definitely/not/a/real/dir/xyz"))

    def test_a_non_positive_timeout_is_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            LocalSandbox(self.root, timeout=0)
        with self.assertRaises(ValueError):
            LocalSandbox(self.root, timeout=-1)

    def test_the_result_is_a_frozen_dataclass(self):
        self.write("ok.py", "print('x')")

        result = self.sandbox.run(self.python("ok.py"))

        self.assertIsInstance(result, CommandResult)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.exit_code = 9


class SandboxExecutionTest(SandboxTestBase):
    """执行语义：退出码、两路输出、空命令、环境变量。"""

    def test_a_successful_command_reports_zero(self):
        self.write("ok.py", "print('跑通了')")

        result = self.sandbox.run(self.python("ok.py"))

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.ok)
        self.assertFalse(result.timed_out)
        self.assertIn("跑通了", result.stdout)

    def test_the_command_and_cwd_are_recorded(self):
        self.write("ok.py", "print('x')")
        command = self.python("ok.py")

        result = self.sandbox.run(command)

        self.assertEqual(result.command, command)
        self.assertEqual(result.cwd, self.root)

    def test_the_two_streams_are_kept_apart(self):
        self.write(
            "both.py",
            "import sys\nprint('去标准输出')\nsys.stderr.write('去标准错误\\n')",
        )

        result = self.sandbox.run(self.python("both.py"))

        self.assertIn("去标准输出", result.stdout)
        self.assertNotIn("去标准错误", result.stdout)
        self.assertIn("去标准错误", result.stderr)
        self.assertNotIn("去标准输出", result.stderr)

    def test_structured_output_is_raw_while_as_text_marks_empty(self):
        # 结构化字段给的是数据（空就是空串）；「（空）」是给人看的
        # 显示格式，只在 as_text() 里出现
        self.write("silent.py", "pass")

        result = self.sandbox.run(self.python("silent.py"))

        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertIn("（空）", result.as_text())

    def test_a_non_zero_exit_code_is_a_result_not_an_exception(self):
        self.write("bad.py", "import sys\nsys.exit(3)")

        result = self.sandbox.run(self.python("bad.py"))

        self.assertEqual(result.exit_code, 3)
        self.assertFalse(result.ok)
        self.assertFalse(result.timed_out)
        self.assertIn("退出码：3", result.as_text())

    def test_commands_run_in_the_workspace_by_default(self):
        self.write(
            "create.py", "open('made.txt', 'w', encoding='utf-8').write('x')"
        )

        self.sandbox.run(self.python("create.py"))

        self.assertTrue((self.root / "made.txt").exists())

    def test_an_empty_command_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("")
        self.assertIn("不能为空", str(caught.exception))

        with self.assertRaises(ToolError):
            self.sandbox.run("   ")

    def test_a_non_positive_timeout_is_rejected_at_call_time(self):
        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi", timeout=0)
        self.assertIn("必须大于 0", str(caught.exception))

        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", timeout=-1)

    def test_a_wrong_timeout_type_is_rejected(self):
        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", timeout="一会儿")

    def test_env_is_passed_to_the_child(self):
        # env 这条路下一阶段要留给 MSVC（vcvars），先钉住它能用
        env = dict(os.environ)
        env["UAC_SANDBOX_MARK"] = "sandbox-env-ok"
        self.write("env.py", "import os\nprint(os.environ.get('UAC_SANDBOX_MARK'))")

        result = self.sandbox.run(self.python("env.py"), env=env)

        self.assertIn("sandbox-env-ok", result.stdout)


class SandboxCwdTest(SandboxTestBase):
    """cwd：workspace 以内的路径都该能跑。"""

    def marker(self, subdir: str) -> None:
        """在 subdir 里放一个会在当前目录写文件的脚本。"""
        self.write(
            f"{subdir}/marker.py",
            "open('made.txt', 'w', encoding='utf-8').write('x')",
        )

    def test_a_relative_subdirectory_is_accepted(self):
        self.marker("sub")

        result = self.sandbox.run(self.python("marker.py"), cwd="sub")

        self.assertEqual(result.cwd, self.root / "sub")
        self.assertTrue((self.root / "sub" / "made.txt").exists())
        self.assertFalse((self.root / "made.txt").exists())

    def test_a_nested_relative_path_is_accepted(self):
        self.marker("a/b")

        result = self.sandbox.run(self.python("marker.py"), cwd="a/b")

        self.assertEqual(result.cwd, self.root / "a" / "b")
        self.assertTrue((self.root / "a" / "b" / "made.txt").exists())

    def test_an_absolute_path_inside_the_workspace_is_accepted(self):
        self.marker("sub")

        result = self.sandbox.run(
            self.python("marker.py"), cwd=str(self.root / "sub")
        )

        self.assertEqual(result.cwd, self.root / "sub")

    def test_a_path_object_is_accepted(self):
        self.marker("sub")

        result = self.sandbox.run(self.python("marker.py"), cwd=Path("sub"))

        self.assertEqual(result.cwd, self.root / "sub")

    def test_the_workspace_root_itself_is_allowed(self):
        self.write("ok.py", "print('x')")

        result = self.sandbox.run(self.python("ok.py"), cwd=".")

        self.assertEqual(result.cwd, self.root)

    def test_cwd_is_normalized_before_it_is_used(self):
        # sub/../other 仍在 workspace 里：展开之后是 other，照跑
        self.write("sub/keep.txt", "x")
        self.marker("other")

        result = self.sandbox.run(self.python("marker.py"), cwd="sub/../other")

        self.assertEqual(result.cwd, self.root / "other")
        self.assertTrue((self.root / "other" / "made.txt").exists())

    def test_a_missing_cwd_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi", cwd="nope")

        self.assertIn("不是目录", str(caught.exception))

    def test_a_file_as_cwd_is_rejected(self):
        self.write("f.txt", "x")

        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", cwd="f.txt")


class SandboxEscapeTest(SandboxTestBase):
    """边界：`..` 和绝对路径都不许把 cwd 带出 workspace。"""

    def setUp(self):
        super().setUp()
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        self.outside = Path(other.name).resolve()

    def test_a_dotdot_escape_is_rejected(self):
        with self.assertRaises(ToolError) as caught:
            self.sandbox.run("echo hi", cwd="..")

        message = str(caught.exception)
        self.assertIn("workspace", message)
        self.assertIn("超出", message)

    def test_a_deep_dotdot_escape_is_rejected(self):
        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", cwd="../..")

    def test_a_relative_escape_to_a_sibling_is_rejected(self):
        # 相对路径里混着 .. 也一样：先 resolve 再判断，绕不过去
        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", cwd=f"../{self.outside.name}")

    def test_an_absolute_path_outside_is_rejected(self):
        with self.assertRaises(ToolError):
            self.sandbox.run("echo hi", cwd=str(self.outside))

    def test_the_command_does_not_run_when_cwd_is_rejected(self):
        # 命令是「在当前目录写一个文件」：真跑起来就会在界外留下证据
        command = f'"{sys.executable}" -c "open(\'made.txt\', \'w\').write(\'x\')"'

        with self.assertRaises(ToolError):
            self.sandbox.run(command, cwd=str(self.outside))

        self.assertFalse((self.outside / "made.txt").exists())


class SandboxTimeoutTest(unittest.TestCase):
    """超时：命令跑太久就杀掉，按正常结果返回，不抛异常。

    workspace 用项目根目录——理由和 test_run_command.py 里那段一样：
    被杀的只是 shell，睡着的子进程会一直占着 workspace 目录，临时目录
    清理会因此失败。睡 5 秒、超时 0.3 秒：真生效每条零点几秒就结束。
    """

    def setUp(self):
        self.sandbox = LocalSandbox(PROJECT_ROOT, timeout=0.3)

    def test_a_slow_command_times_out(self):
        result = self.sandbox.run(slow_command(), timeout=0.3)

        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertFalse(result.ok)

    def test_the_timeout_is_not_raised(self):
        result = self.sandbox.run(slow_command(), timeout=0.3)

        self.assertIsInstance(result, CommandResult)

    def test_the_timeout_text_says_so(self):
        result = self.sandbox.run(slow_command(), timeout=0.3)

        text = result.as_text()
        self.assertIn("超时", text)
        self.assertIn("0.3", text)
        self.assertIn("没有保留", text)

    def test_the_timeout_actually_fires(self):
        started = time.perf_counter()
        self.sandbox.run(slow_command(), timeout=0.3)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 3)

    def test_a_fast_command_is_unaffected(self):
        result = self.sandbox.run(fast_command(), timeout=30)

        self.assertEqual(result.exit_code, 0)
        self.assertIn("2", result.stdout)

    def test_the_default_applies_and_can_be_overridden(self):
        # 默认 0.3（构造时给的）会超时……
        self.assertTrue(self.sandbox.run(slow_command()).timed_out)

        # ……单次给 30 就不会
        patient = LocalSandbox(PROJECT_ROOT, timeout=30)
        self.assertFalse(patient.run(fast_command(), timeout=30).timed_out)

    def test_the_limit_actually_used_is_recorded(self):
        result = self.sandbox.run(slow_command(), timeout=0.5)

        self.assertEqual(result.timeout, 0.5)

    def test_output_is_not_kept_on_timeout(self):
        result = self.sandbox.run(slow_command(), timeout=0.3)

        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

    def test_the_timeout_text_matches_run_command(self):
        # 同一条慢命令，工具和 sandbox 的超时文案必须逐字一致
        command = slow_command()
        expected = RunCommandTool(root=PROJECT_ROOT, timeout=0.3).run(
            command=command, timeout=0.3
        )

        result = LocalSandbox(PROJECT_ROOT, timeout=0.3).run(command, timeout=0.3)

        self.assertEqual(result.as_text(), expected)


class SandboxMatchesRunCommandTest(SandboxTestBase):
    """同一条命令，sandbox 的 as_text() 和 RunCommandTool 的输出逐字相同。

    钉的是「复用同一份语义」：截断、空输出标记、解码回退、退出码文案，
    全都要对得上；对不上说明某一方偷偷长出了第二份实现。
    """

    def assert_same_text(self, command: str) -> CommandResult:
        expected = RunCommandTool(root=self.root).run(command=command)

        result = self.sandbox.run(command)

        self.assertEqual(result.as_text(), expected)
        return result

    def test_success_output_matches(self):
        self.write("ok.py", "print('一致吗')")

        self.assert_same_text(self.python("ok.py"))

    def test_non_zero_exit_matches(self):
        self.write("bad.py", "import sys\nsys.exit(5)")

        self.assert_same_text(self.python("bad.py"))

    def test_empty_output_matches(self):
        self.write("silent.py", "pass")

        self.assert_same_text(self.python("silent.py"))

    def test_long_output_clipping_matches(self):
        self.write("big.py", "print('开头标记')\nprint('x' * 50000)\nprint('结尾标记')")

        result = self.assert_same_text(self.python("big.py"))

        self.assertIn("省略中间", result.as_text())

    def test_long_stderr_clipping_matches(self):
        self.write(
            "bigerr.py",
            "import sys\nsys.stderr.write('x' * 50000)\nsys.stderr.write('结尾')\n",
        )

        self.assert_same_text(self.python("bigerr.py"))

    def test_gbk_output_matches(self):
        self.write(
            "gbk.py",
            "import sys\nsys.stdout.buffer.write('中文输出'.encode('gbk'))",
        )

        self.assert_same_text(self.python("gbk.py"))

    def test_binary_output_matches(self):
        self.write("bin.py", "import sys\nsys.stdout.buffer.write(b'\\x00\\x01\\xff')")

        self.assert_same_text(self.python("bin.py"))


class SandboxManagerTest(SandboxTestBase):
    """管理器：默认 local、懒建且只建一次、按名字选实现。"""

    def test_the_default_kind_is_local(self):
        manager = SandboxManager(self.root)

        self.assertEqual(manager.kind, DEFAULT_KIND)
        self.assertEqual(DEFAULT_KIND, "local")
        # 10.3 起 docker 也是可选项（默认没变，还是 local）
        self.assertEqual(KINDS, ("local", "docker"))

    def test_it_builds_a_local_sandbox(self):
        manager = SandboxManager(self.root)

        self.assertIsInstance(manager.sandbox, LocalSandbox)
        self.assertEqual(manager.sandbox.workspace, self.root)

    def test_the_sandbox_is_built_once(self):
        manager = SandboxManager(self.root)

        self.assertIs(manager.sandbox, manager.sandbox)

    def test_run_delegates_to_the_sandbox(self):
        self.write("ok.py", "print('走管理器')")

        result = SandboxManager(self.root).run(self.python("ok.py"))

        self.assertIsInstance(result, CommandResult)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("走管理器", result.stdout)

    def test_the_timeout_is_passed_through(self):
        manager = SandboxManager(PROJECT_ROOT, timeout=0.3)

        self.assertTrue(manager.run(slow_command()).timed_out)

    def test_an_unknown_kind_is_refused_with_the_options(self):
        with self.assertRaises(ToolError) as caught:
            SandboxManager(self.root, kind="remote")

        message = str(caught.exception)
        self.assertIn("remote", message)
        self.assertIn("local", message)
        # 可选项列表里现在还有 docker（10.3 加的）
        self.assertIn("docker", message)

    def test_a_bad_workspace_is_reported_at_the_first_run(self):
        # 懒建：配置不对推迟到真要用的时候才炸，构造时不联网、不碰磁盘
        manager = SandboxManager(Path("C:/definitely/not/a/real/dir/xyz"))

        with self.assertRaises(ToolError):
            manager.run("echo hi")


if __name__ == "__main__":
    unittest.main()
