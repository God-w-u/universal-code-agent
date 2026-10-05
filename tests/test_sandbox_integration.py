"""Sandbox 接入（Phase 10.2）测试：四个执行工具确实经过 Sandbox。

手法是 RecordingSandbox：每次调用先把 (command, cwd, timeout, env) 记下来，
再原样转给真的 LocalSandbox。所以工具拿到的还是真命令跑出来的真结果
（退出码、解码、截断、超时都照旧），同时「执行确实从 Sandbox 走的」有了
证据——没走的话记录是空的或对不上。

覆盖：
- RunCommandTool / CompileTool / BuildTool / TestTool 各自经过 Sandbox
  （BuildTool 的 cmake 两步、TestTool 的每一条命令都算）。
- timeout、非 0 退出码、env 透传三个老行为不变。
- workspace 边界：工具自己的路径校验先拦下的，Sandbox 一次都不许被叫到；
  危险 root 的检查也还在。

超时那几条和 test_run_command.py 一样用项目根目录当 workspace：被杀的只是
shell，睡着的子进程会一直占着目录，Windows 上临时目录清理会因此失败
（tests/test_sandbox.py 顶部有同样的说明）。

全部离线：假编译器 / 假构建工具都是本机解释器的脚本，不联网。
"""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from config import PROJECT_ROOT
from sandbox import CommandResult, LocalSandbox, Sandbox
from tools import BuildTool, CompileTool, RunCommandTool, TestTool
from tools.base import ToolError


def slow_command(seconds: float = 5) -> str:
    """睡一会儿的命令，用来测超时。用 -c 不落文件，没有清理问题。"""
    return f'"{sys.executable}" -c "import time;time.sleep({seconds})"'


class RecordingSandbox(Sandbox):
    """把每次调用记下来，再原样转给真的 LocalSandbox。

    行为一点不打折（记录完就转发），所以工具输出的东西和没接 sandbox
    时逐字相同；记录只用来证明「这一步真的从这儿过了」。
    """

    kind = "recording"

    def __init__(self, workspace: Path, timeout: float = 30.0) -> None:
        self._inner = LocalSandbox(workspace, timeout=timeout)
        self.calls: list[dict] = []

    @property
    def workspace(self) -> Path:
        return self._inner.workspace

    def run(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        self.calls.append(
            {"command": command, "cwd": cwd, "timeout": timeout, "env": env}
        )
        return self._inner.run(command, cwd=cwd, timeout=timeout, env=env)

    def commands(self) -> list[str]:
        """只取命令字符串，按顺序比断言时好读。"""
        return [call["command"] for call in self.calls]


class IntegrationTestBase(unittest.TestCase):
    """临时目录当 root，配一个 RecordingSandbox。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.sandbox = RecordingSandbox(self.root)
        self.calls = self.sandbox.calls

    def write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        return path

    def python(self, *args: str) -> str:
        """用当前这个解释器跑命令，不依赖系统里装了什么。"""
        return f'"{sys.executable}" ' + " ".join(args)


class RunCommandToolSandboxTest(IntegrationTestBase):
    """RunCommandTool：执行交给 Sandbox，自己只剩翻结果。"""

    def test_the_command_actually_goes_through_the_sandbox(self):
        self.write("ok.py", "print('跑通了')")
        command = self.python("ok.py")
        tool = RunCommandTool(root=self.root, sandbox=self.sandbox)

        text = tool.run(command=command)

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["command"], command)
        self.assertIn("跑通了", text)

    def test_the_tools_default_timeout_reaches_the_sandbox(self):
        self.write("ok.py", "print('x')")
        tool = RunCommandTool(root=self.root, timeout=7, sandbox=self.sandbox)

        tool.run(command=self.python("ok.py"))

        self.assertEqual(self.calls[0]["timeout"], 7)

    def test_a_per_call_timeout_reaches_the_sandbox(self):
        self.write("ok.py", "print('x')")
        tool = RunCommandTool(root=self.root, timeout=7, sandbox=self.sandbox)

        tool.run(command=self.python("ok.py"), timeout=3)

        self.assertEqual(self.calls[0]["timeout"], 3)

    def test_env_reaches_the_sandbox_and_the_child(self):
        # env 不在 parameters 里（模型不能自己传环境），走内层的
        # run_with_code——compile / build 用的就是这条路
        env = dict(os.environ)
        env["UAC_WIRE_MARK"] = "wire-10.2"
        self.write("env.py", "import os\nprint(os.environ.get('UAC_WIRE_MARK'))")
        tool = RunCommandTool(root=self.root, sandbox=self.sandbox)

        code, text = tool.run_with_code(command=self.python("env.py"), env=env)

        self.assertEqual(code, 0)
        self.assertIs(self.calls[0]["env"], env)
        self.assertIn("wire-10.2", text)

    def test_no_env_means_none_is_passed_through(self):
        self.write("ok.py", "print('x')")
        tool = RunCommandTool(root=self.root, sandbox=self.sandbox)

        tool.run(command=self.python("ok.py"))

        self.assertIsNone(self.calls[0]["env"])

    def test_a_non_zero_exit_code_is_still_a_result(self):
        self.write("bad.py", "import sys\nsys.exit(4)")
        tool = RunCommandTool(root=self.root, sandbox=self.sandbox)

        text = tool.run(command=self.python("bad.py"))

        self.assertIn("退出码：4", text)
        self.assertIn("非 0", text)

    def test_commands_still_run_inside_the_root(self):
        # cwd 由 Sandbox 定成 workspace（就是 root）：相对路径写出来的
        # 文件必须落在 root 里，边界才算真的还在
        self.write(
            "create.py", "open('made.txt', 'w', encoding='utf-8').write('x')"
        )
        tool = RunCommandTool(root=self.root, sandbox=self.sandbox)

        tool.run(command=self.python("create.py"))

        self.assertTrue((self.root / "made.txt").exists())

    def test_the_output_is_identical_to_the_direct_path(self):
        # 接入不许改一个字：同一个工具、两种接法，结果逐字相同
        self.write(
            "mix.py",
            "import sys\nprint('出来')\nsys.stderr.write('错了\\n')\nsys.exit(6)",
        )
        command = self.python("mix.py")

        wired = RunCommandTool(root=self.root, sandbox=self.sandbox).run(
            command=command
        )
        direct = RunCommandTool(root=self.root).run(command=command)

        self.assertEqual(wired, direct)

    def test_the_default_sandbox_is_built_by_the_manager(self):
        # 不传 sandbox 时按 root 装一个 LocalSandbox：workspace 就是 root，
        # timeout 跟工具一致——这是四个工具的日常状态
        tool = RunCommandTool(root=self.root, timeout=7)

        self.assertIsInstance(tool.sandbox, LocalSandbox)
        self.assertEqual(tool.sandbox.workspace, self.root)
        self.assertEqual(tool.sandbox.timeout, 7)

    def test_an_injected_sandbox_is_used_as_is(self):
        tool = RunCommandTool(root=self.root, sandbox=self.sandbox)

        self.assertIs(tool.sandbox, self.sandbox)


class RunCommandToolTimeoutTest(unittest.TestCase):
    """超时也要经过 Sandbox，而且照旧立刻返回。

    workspace 用项目根目录，理由见本文件顶部。睡 5 秒、超时 0.3 秒：
    真生效每条零点几秒就结束。
    """

    def setUp(self):
        self.sandbox = RecordingSandbox(PROJECT_ROOT)
        self.tool = RunCommandTool(
            root=PROJECT_ROOT, timeout=0.3, sandbox=self.sandbox
        )

    def test_a_slow_command_times_out_through_the_sandbox(self):
        started = time.perf_counter()
        text = self.tool.run(command=slow_command(), timeout=0.3)
        elapsed = time.perf_counter() - started

        self.assertIn("超时", text)
        self.assertIn("已强制终止", text)
        self.assertEqual(self.sandbox.calls[0]["timeout"], 0.3)
        self.assertLess(elapsed, 3)


# 假编译器：参数原样报出来，固定退出 0。够验「命令拼得对不对、结果怎么
# 处理」——不是验 g++ 能不能编译（那种事归 test_compile.py 的假编译器）。
FAKE_COMPILER = '''\
"""假编译器。"""
import sys

print("假编译器收到：" + " ".join(sys.argv[1:]))
'''


class CompileToolSandboxTest(IntegrationTestBase):
    """CompileTool：拼命令、查路径还是它的活，执行走 Sandbox。"""

    def setUp(self):
        super().setUp()
        script = self.write("fake_compiler.py", FAKE_COMPILER)
        # 和 test_compile.py 同一个注入方式：compiler= 一整条命令，
        # 解释器和脚本路径都用引号包住，防止路径里有空格
        self.compiler = f'"{sys.executable}" "{script}"'

    def tool(self, **kwargs) -> CompileTool:
        return CompileTool(
            root=self.root, compiler=self.compiler, sandbox=self.sandbox, **kwargs
        )

    def test_the_compile_command_goes_through_the_sandbox(self):
        self.write("main.cpp", "int main() { return 0; }\n")

        text = self.tool().run(source="main.cpp", output="main.exe")

        self.assertEqual(len(self.calls), 1)
        self.assertIn("main.cpp", self.calls[0]["command"])
        self.assertIn("假编译器收到", text)
        self.assertIn("退出码：0", text)

    def test_flags_and_output_are_still_composed_by_the_tool(self):
        self.write("main.cpp", "int main() { return 0; }\n")

        self.tool().run(source="main.cpp", output="main.exe", flags="-O2")

        command = self.calls[0]["command"]
        self.assertIn("-O2", command)
        self.assertIn("main.cpp -o main.exe", command)

    def test_a_non_msvc_compiler_still_passes_no_env(self):
        # _env() 只对 MSVC 铺 vcvars：别的编译器 env 就该是 None（继承）
        self.write("main.cpp", "int main() { return 0; }\n")

        self.tool().run(source="main.cpp")

        self.assertIsNone(self.calls[0]["env"])

    def test_a_rejected_source_path_never_reaches_the_sandbox(self):
        with self.assertRaises(ToolError):
            self.tool().run(source="../outside.cpp")

        self.assertEqual(self.calls, [])


# 假构建工具：第一个参数是「我是谁」（.bat 传进来的），后面才是命令行。
# msbuild 多报一个环境记号，验 vcvars 那份环境透传到了子进程。
FAKE_BUILD = '''\
"""假构建工具：够用的那部分行为。"""
import os
import sys

tool, args = sys.argv[1], sys.argv[2:]
print(f"fake {tool} " + " ".join(args))

if tool == "msbuild":
    print("marker: " + os.environ.get("FAKE_VS_MARKER", "（没传进来）"))
'''

BAT = '@echo off\n"{python}" "{script}" {name} %*\nexit /b %ERRORLEVEL%\n'


class BuildToolSandboxTest(IntegrationTestBase):
    """BuildTool：几步、什么顺序、什么环境还是它的活，执行走 Sandbox。"""

    def setUp(self):
        super().setUp()
        script = self.write("fake_build.py", FAKE_BUILD)
        for name in ("cmake", "make", "msbuild"):
            (self.root / f"{name}.bat").write_text(
                BAT.format(python=sys.executable, script=script, name=name),
                encoding="utf-8",
            )
        # 用真的 PATH：shell 真的去找了一圈才找到它们，which 也是
        env = patch.dict(
            os.environ, {"PATH": f"{self.root}{os.pathsep}{os.environ['PATH']}"}
        )
        env.start()
        self.addCleanup(env.stop)

    def test_a_make_build_goes_through_the_sandbox(self):
        tool = BuildTool(root=self.root, sandbox=self.sandbox)

        text = tool.run(system="make")

        self.assertEqual(self.sandbox.commands(), ["make"])
        self.assertIn("fake make", text)
        self.assertIn("退出码：0", text)

    def test_every_cmake_step_goes_through_the_sandbox(self):
        tool = BuildTool(root=self.root, sandbox=self.sandbox)

        text = tool.run(system="cmake")

        self.assertEqual(
            self.sandbox.commands(),
            ["cmake -S . -B build", "cmake --build build"],
        )
        # 步骤本身和顺序还是 BuildTool 拼的：两步都标了号
        self.assertIn("[第 1/2 步]", text)
        self.assertIn("[第 2/2 步]", text)

    def test_the_msbuild_env_still_reaches_the_child(self):
        # vcvars 那份环境由 BuildTool 铺，透传链一路到子进程：sandbox
        # 收到的是它，子进程读到的也是它
        self.write("demo.sln", "")
        tool = BuildTool(root=self.root, sandbox=self.sandbox)

        with patch("tools.build.vsenv.vcvars_env") as vcvars:
            vcvars.return_value = dict(os.environ, FAKE_VS_MARKER="vs-env-ok")
            text = tool.run(system="msbuild")

        self.assertEqual(self.sandbox.commands(), ["msbuild demo.sln"])
        self.assertEqual(self.calls[0]["env"].get("FAKE_VS_MARKER"), "vs-env-ok")
        self.assertIn("marker: vs-env-ok", text)
        self.assertIn("退出码：0", text)

    def test_a_rejected_source_root_never_reaches_the_sandbox(self):
        tool = BuildTool(root=self.root, sandbox=self.sandbox)

        with self.assertRaises(ToolError):
            tool.run(system="make", source_root="../outside")

        self.assertEqual(self.calls, [])


class TestToolSandboxTest(IntegrationTestBase):
    """TestTool：命令还是它自己那条，执行走 Sandbox。"""

    def tool(self, **kwargs) -> TestTool:
        self.write("suite.py", kwargs.pop("suite", "print('用例全过')"))
        return TestTool(
            root=self.root,
            command=self.python("suite.py"),
            sandbox=self.sandbox,
            **kwargs,
        )

    def test_the_test_command_goes_through_the_sandbox(self):
        text = self.tool().run()

        self.assertEqual(self.sandbox.commands(), [self.python("suite.py")])
        self.assertIn("用例全过", text)
        self.assertIn("退出码：0", text)

    def test_a_failing_suite_is_still_a_result(self):
        tool = self.tool(suite="import sys\nprint('3 failed')\nsys.exit(1)")

        text = tool.run()

        self.assertIn("退出码：1", text)
        self.assertIn("3 failed", text)

    def test_the_tool_timeout_reaches_the_sandbox(self):
        tool = self.tool(timeout=7)

        tool.run()

        self.assertEqual(self.calls[0]["timeout"], 7)


class TestToolTimeoutTest(unittest.TestCase):
    """测试套件跑太久：超时经过 Sandbox，照旧正常返回。"""

    def test_a_slow_suite_times_out_through_the_sandbox(self):
        sandbox = RecordingSandbox(PROJECT_ROOT)
        tool = TestTool(
            root=PROJECT_ROOT,
            timeout=0.3,
            command=slow_command(),
            sandbox=sandbox,
        )

        started = time.perf_counter()
        text = tool.run()
        elapsed = time.perf_counter() - started

        self.assertIn("超时", text)
        self.assertEqual(sandbox.calls[0]["timeout"], 0.3)
        self.assertLess(elapsed, 3)


class DangerousRootStillRefusedTest(unittest.TestCase):
    """危险 root（盘符根目录）在构造时就被拒——还没到 Sandbox 那一步。"""

    def test_every_execution_tool_still_checks_its_root(self):
        with self.assertRaises(ToolError):
            RunCommandTool(root=Path("C:/"))
        with self.assertRaises(ToolError):
            CompileTool(root=Path("C:/"))
        with self.assertRaises(ToolError):
            BuildTool(root=Path("C:/"))
        with self.assertRaises(ToolError):
            TestTool(root=Path("C:/"))


if __name__ == "__main__":
    unittest.main()
