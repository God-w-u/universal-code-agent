"""RunCommandTool 测试：正常执行、stdout / stderr、退出码、空命令、Agent 调用。

全部在临时目录里跑本机解释器，不联网。
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import Agent, ContextManager
from config import PROJECT_ROOT
from tools import RunCommandTool, Tool, ToolError, ToolRegistry
from tools.run_command import DEFAULT_TIMEOUT


def sections(result: str) -> tuple[str, str]:
    """把结果拆成 (标准输出, 标准错误) 两段。"""
    _, _, rest = result.partition("标准输出：\n")
    stdout, _, stderr = rest.partition("\n\n标准错误：\n")
    return stdout, stderr


class RunCommandToolTestBase(unittest.TestCase):
    """搭一个临时目录当作 root，用例结束后自动删除。"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.tool = RunCommandTool(root=self.root)

    def write(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        return path

    def python(self, *args: str) -> str:
        """用当前这个解释器跑命令，不依赖系统里装了什么。"""
        return f'"{sys.executable}" ' + " ".join(args)

    def run_command(self, command: str) -> str:
        return self.tool.run(command=command)


class RunCommandToolBasicTest(RunCommandToolTestBase):
    def test_runs_a_command_and_reports_success(self):
        self.write("ok.py", "print('跑通了')")

        result = self.run_command(self.python("ok.py"))

        self.assertIn("退出码：0", result)
        self.assertIn("跑通了", result)

    def test_the_command_is_echoed_back(self):
        self.write("ok.py", "print('x')")
        command = self.python("ok.py")

        self.assertIn(command, self.run_command(command))

    def test_stdout_is_captured(self):
        self.write("out.py", "print('这是标准输出')")

        stdout, _ = sections(self.run_command(self.python("out.py")))

        self.assertIn("这是标准输出", stdout)

    def test_stderr_is_captured(self):
        self.write("err.py", "import sys\nsys.stderr.write('这是标准错误\\n')")

        _, stderr = sections(self.run_command(self.python("err.py")))

        self.assertIn("这是标准错误", stderr)

    def test_the_two_streams_are_kept_apart(self):
        self.write(
            "both.py",
            "import sys\nprint('去标准输出')\nsys.stderr.write('去标准错误\\n')",
        )

        stdout, stderr = sections(self.run_command(self.python("both.py")))

        self.assertIn("去标准输出", stdout)
        self.assertNotIn("去标准错误", stdout)
        self.assertIn("去标准错误", stderr)
        self.assertNotIn("去标准输出", stderr)

    def test_empty_stdout_is_marked(self):
        self.write("silent.py", "pass")

        stdout, _ = sections(self.run_command(self.python("silent.py")))

        self.assertEqual(stdout, "（空）")

    def test_empty_stderr_is_marked(self):
        self.write("silent.py", "pass")

        _, stderr = sections(self.run_command(self.python("silent.py")))

        self.assertEqual(stderr, "（空）")

    def test_commands_run_in_the_project_root(self):
        self.write("create.py", "open('made.txt', 'w', encoding='utf-8').write('x')")

        self.run_command(self.python("create.py"))

        self.assertTrue((self.root / "made.txt").exists())

    def test_relative_paths_resolve_inside_root(self):
        self.write("data.txt", "文件内容")
        self.write(
            "read.py", "print(open('data.txt', encoding='utf-8').read())"
        )

        result = self.run_command(self.python("read.py"))

        self.assertIn("文件内容", result)

    def test_shell_operators_work(self):
        # shell=True 的意义：能直接跑带 && 、管道的命令字符串
        self.write("a.py", "print('第一步')")
        self.write("b.py", "print('第二步')")

        command = f"{self.python('a.py')} && {self.python('b.py')}"
        stdout, _ = sections(self.run_command(command))

        self.assertIn("第一步", stdout)
        self.assertIn("第二步", stdout)


class RunCommandToolExitCodeTest(RunCommandToolTestBase):
    def test_non_zero_exit_code_is_reported(self):
        self.write("bad.py", "import sys\nsys.exit(3)")

        result = self.run_command(self.python("bad.py"))

        self.assertIn("退出码：3", result)

    def test_non_zero_exit_does_not_raise(self):
        # 明确钉住：命令失败是「结果」，不是「异常」
        self.write("bad.py", "import sys\nsys.exit(1)")

        self.run_command(self.python("bad.py"))  # 不抛就是通过

    def test_failure_is_stated_in_words(self):
        self.write("bad.py", "import sys\nsys.exit(1)")

        self.assertIn("命令失败", self.run_command(self.python("bad.py")))

    def test_output_of_a_failing_command_is_kept(self):
        self.write(
            "bad.py",
            "import sys\nprint('前半截输出')\nsys.stderr.write('出错原因\\n')\n"
            "sys.exit(2)",
        )

        result = self.run_command(self.python("bad.py"))

        self.assertIn("前半截输出", result)
        self.assertIn("出错原因", result)

    def test_exit_code_is_zero_on_success(self):
        self.write("ok.py", "pass")

        self.assertIn("退出码：0", self.run_command(self.python("ok.py")))


class RunCommandToolBadInputTest(RunCommandToolTestBase):
    def test_empty_command_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.run_command("")

        self.assertIn("不能为空", str(ctx.exception))

    def test_whitespace_only_command_rejected(self):
        with self.assertRaises(ToolError):
            self.run_command("   ")

    def test_missing_command_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run()

    def test_wrong_command_type_rejected(self):
        with self.assertRaises(ToolError):
            self.run_command(123)


class RunCommandToolTimeoutTest(unittest.TestCase):
    """超时：命令跑太久就杀掉，按正常结果返回，不抛异常。

    这个类刻意不搭临时目录，直接用项目根目录当 root。

    原因是超时只杀得掉 shell，真正干活的子进程（那个睡着的 python）会一直活到
    睡醒。它的工作目录就是工具的 root，而 Windows 上只要有进程以某个目录为
    工作目录，那个目录就删不掉——临时目录清理会直接报 PermissionError。
    用项目根目录就没有需要清理的东西，测试也就不会时快时慢。

    睡 5 秒、超时 0.3 秒：真生效的话每条用例零点几秒就结束，真坏了要等 5 秒。
    """

    def setUp(self):
        self.tool = RunCommandTool(root=PROJECT_ROOT, timeout=0.3)

    def slow_command(self, seconds: float = 5) -> str:
        # 用 -c 而不是脚本文件：不留任何文件，就没有清理问题
        return f'"{sys.executable}" -c "import time;time.sleep({seconds})"'

    def fast_command(self) -> str:
        return f'"{sys.executable}" -c "print(1+1)"'

    def run_with_timeout(self, seconds: float = 0.3, command: str | None = None) -> str:
        return self.tool.run(command=command or self.slow_command(), timeout=seconds)

    def test_a_slow_command_times_out(self):
        self.assertIn("超时", self.run_with_timeout())

    def test_the_timeout_is_not_raised(self):
        self.run_with_timeout()  # 不抛就是通过

    def test_the_result_is_a_normal_string(self):
        self.assertIsInstance(self.run_with_timeout(), str)

    def test_the_command_is_named_in_the_result(self):
        command = self.slow_command()

        self.assertIn(command, self.run_with_timeout(command=command))

    def test_the_limit_is_named_in_the_result(self):
        self.assertIn("0.3", self.run_with_timeout(0.3))

    def test_it_says_the_command_was_killed(self):
        self.assertIn("终止", self.run_with_timeout())

    def test_the_timeout_actually_fires(self):
        """回归测试，钉住两个实测出来的坑。

        坑一：subprocess.run(timeout=) 在 Windows + shell=True 下超时是「假的」。
        它内部 kill 掉 shell 之后还要再 communicate() 收一次残余输出，
        而 shell 的子进程还占着管道，那一下会一直等到命令自己跑完。
        实测设 1 秒的超时等了 8 秒。

        坑二：用 with 语句会 close() 管道，而管道还有个读取线程堵在 read() 上，
        主线程得排队等它。实测设 0.3 秒的超时，close() 又堵了 4.73 秒。
        """
        started = time.perf_counter()
        self.run_with_timeout(0.3)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 3)

    def test_a_fast_command_is_unaffected(self):
        result = self.tool.run(command=self.fast_command(), timeout=30)

        self.assertIn("退出码：0", result)
        self.assertIn("2", result)

    def test_calling_without_a_timeout_uses_the_tool_default(self):
        tool = RunCommandTool(root=PROJECT_ROOT, timeout=0.3)

        self.assertIn("超时", tool.run(command=self.slow_command()))

    def test_calling_with_a_timeout_overrides_the_tool_default(self):
        tool = RunCommandTool(root=PROJECT_ROOT, timeout=30)

        self.assertIn("超时", tool.run(command=self.slow_command(), timeout=0.3))

    def test_a_short_default_can_be_overridden_upwards(self):
        tool = RunCommandTool(root=PROJECT_ROOT, timeout=0.1)

        result = tool.run(command=self.fast_command(), timeout=30)

        self.assertIn("退出码：0", result)

    def test_the_default_timeout_is_sane(self):
        self.assertEqual(RunCommandTool(root=PROJECT_ROOT).timeout, DEFAULT_TIMEOUT)
        self.assertGreater(DEFAULT_TIMEOUT, 0)


class RunCommandToolTimeoutArgumentTest(RunCommandToolTestBase):
    def test_zero_timeout_at_construction_rejected(self):
        with self.assertRaises(ValueError):
            RunCommandTool(root=self.root, timeout=0)

    def test_negative_timeout_at_construction_rejected(self):
        with self.assertRaises(ValueError):
            RunCommandTool(root=self.root, timeout=-1)

    def test_zero_timeout_at_call_time_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.tool.run(command="echo hi", timeout=0)

        self.assertIn("必须大于 0", str(ctx.exception))

    def test_wrong_timeout_type_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(command="echo hi", timeout="一会儿")

    def test_timeout_is_optional_in_schema(self):
        parameters = RunCommandTool(root=self.root).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(list(parameters["properties"]), ["command", "timeout"])
        self.assertEqual(parameters["required"], ["command"])


class RunCommandToolEncodingTest(RunCommandToolTestBase):
    def test_gbk_output_is_decoded(self):
        self.write(
            "gbk.py",
            "import sys\nsys.stdout.buffer.write('中文输出'.encode('gbk'))",
        )

        result = self.run_command(self.python("gbk.py"))

        self.assertIn("中文输出", result)

    def test_binary_output_does_not_raise(self):
        # 命令输出不保证是干净文本，不能像 read_file 那样直接拒绝
        self.write("bin.py", "import sys\nsys.stdout.buffer.write(b'\\x00\\x01\\xff')")

        result = self.run_command(self.python("bin.py"))

        self.assertIn("退出码：0", result)


class RunCommandToolOutputLimitTest(RunCommandToolTestBase):
    def setUp(self):
        super().setUp()
        self.tool = RunCommandTool(root=self.root)

    def write_exact(self, name: str, chars: int, to_stderr: bool = False) -> None:
        """写一个输出正好 chars 个字符的脚本（不带换行，方便卡边界）。"""
        stream = "sys.stderr" if to_stderr else "sys.stdout"
        self.write(name, f"import sys\n{stream}.write('x' * {chars})\n")

    def test_long_output_is_clipped(self):
        self.write("big.py", "print('x' * 50000)")

        result = self.run_command(self.python("big.py"))

        self.assertLess(len(result), 25_000)

    def test_clipping_keeps_the_beginning(self):
        self.write("big.py", "print('开头标记')\nprint('x' * 50000)")

        self.assertIn("开头标记", self.run_command(self.python("big.py")))

    def test_clipping_keeps_the_end(self):
        # 跑测试时结论在最末尾，只留开头等于把最有用的那句扔了
        self.write("big.py", "print('x' * 50000)\nprint('结尾标记')")

        self.assertIn("结尾标记", self.run_command(self.python("big.py")))

    def test_clipping_says_it_was_clipped(self):
        self.write("big.py", "print('x' * 50000)")

        self.assertIn("省略中间", self.run_command(self.python("big.py")))

    def test_short_output_is_not_clipped(self):
        self.write("small.py", "print('很短')")

        self.assertNotIn("省略中间", self.run_command(self.python("small.py")))

    def test_the_limit_is_actually_bounded(self):
        self.assertLessEqual(RunCommandTool.max_output_chars, 100_000)

    def test_output_exactly_at_the_limit_is_not_clipped(self):
        self.write_exact("exact.py", RunCommandTool.max_output_chars)

        self.assertNotIn("省略中间", self.run_command(self.python("exact.py")))

    def test_output_one_char_over_the_limit_is_clipped(self):
        self.write_exact("over.py", RunCommandTool.max_output_chars + 1)

        self.assertIn("省略中间", self.run_command(self.python("over.py")))

    # --- 标准错误单独一套限制，不能只测标准输出 ---

    def test_long_stderr_is_clipped(self):
        self.write_exact("bigerr.py", 50_000, to_stderr=True)

        result = self.run_command(self.python("bigerr.py"))

        self.assertLess(len(result), 25_000)
        self.assertIn("省略中间", result)

    def test_stderr_clipping_keeps_the_beginning(self):
        self.write(
            "bigerr.py",
            "import sys\n"
            "sys.stderr.write('错误开头')\n"
            "sys.stderr.write('x' * 50000)\n",
        )

        _, stderr = sections(self.run_command(self.python("bigerr.py")))

        self.assertIn("错误开头", stderr)

    def test_stderr_clipping_keeps_the_end(self):
        self.write(
            "bigerr.py",
            "import sys\n"
            "sys.stderr.write('x' * 50000)\n"
            "sys.stderr.write('错误结尾')\n",
        )

        _, stderr = sections(self.run_command(self.python("bigerr.py")))

        self.assertIn("错误结尾", stderr)

    def test_the_two_streams_have_separate_budgets(self):
        # 标准错误爆掉，不该连坐把标准输出也砍了
        self.write(
            "both.py",
            "import sys\n"
            "print('标准输出内容')\n"
            "sys.stderr.write('x' * 50000)\n",
        )

        stdout, stderr = sections(self.run_command(self.python("both.py")))

        self.assertIn("标准输出内容", stdout)
        self.assertNotIn("省略中间", stdout)
        self.assertIn("省略中间", stderr)

    def test_both_streams_clipped_still_fits_the_context_budget(self):
        # 两路各留 2 万，加上外壳仍要低于 ContextManager 的单条上限：
        # 否则工具自己写好的截断说明会被上下文层再砍一次，模型就看不懂了
        self.write(
            "both.py",
            "import sys\n"
            "sys.stdout.write('x' * 50000)\n"
            "sys.stderr.write('y' * 50000)\n",
        )

        result = self.run_command(self.python("both.py"))

        self.assertLess(len(result), ContextManager().max_tool_chars)

    # --- 退出码在内容之外，截断不能把它弄丢 ---

    def test_exit_code_survives_clipping(self):
        self.write(
            "big.py", "import sys\nprint('x' * 50000)\nsys.exit(7)\n"
        )

        result = self.run_command(self.python("big.py"))

        self.assertIn("退出码：7", result)
        self.assertIn("省略中间", result)

    def test_success_exit_code_survives_clipping(self):
        self.write_exact("big.py", 50_000)

        self.assertIn("退出码：0", self.run_command(self.python("big.py")))

    def test_exit_code_survives_stderr_clipping(self):
        self.write(
            "bigerr.py",
            "import sys\n"
            "sys.stderr.write('x' * 50000)\n"
            "sys.exit(9)\n",
        )

        result = self.run_command(self.python("bigerr.py"))

        self.assertIn("退出码：9", result)
        self.assertIn("省略中间", result)


class RunCommandToolShapeTest(unittest.TestCase):
    def test_it_is_a_tool(self):
        self.assertIsInstance(RunCommandTool(root=PROJECT_ROOT), Tool)
        self.assertEqual(RunCommandTool.name, "run_command")

    def test_schema_shape(self):
        parameters = RunCommandTool(root=PROJECT_ROOT).to_schema()["function"][
            "parameters"
        ]

        self.assertEqual(list(parameters["properties"]), ["command", "timeout"])
        self.assertEqual(parameters["required"], ["command"])
        self.assertEqual(parameters["properties"]["command"]["type"], "string")
        self.assertEqual(parameters["properties"]["timeout"]["type"], "number")

    def test_default_root_is_project_root(self):
        self.assertEqual(RunCommandTool().root, PROJECT_ROOT)

    def test_drive_root_rejected(self):
        with self.assertRaises(ToolError):
            RunCommandTool(root=Path("C:/"))

    def test_missing_root_rejected(self):
        with self.assertRaises(ToolError):
            RunCommandTool(root=Path("C:/definitely/not/a/real/dir/xyz"))


class StubClient:
    """按剧本回话的假客户端。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages, tools)


def tool_call_reply(name: str, arguments: str, call_id: str = "c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def final_reply(content: str = "好的"):
    return {"role": "assistant", "content": content}


class RunCommandToolAgentTest(RunCommandToolTestBase):
    """走一遍完整的 Agent 调用链：模型要求执行命令 → 结果回灌。"""

    # 超时那几条用项目根目录当 root，理由见 RunCommandToolTimeoutTest 的说明：
    # 被杀的只是 shell，睡着的子进程会一直占着 root 目录，临时目录清理会失败。
    timeout_root = PROJECT_ROOT

    def build(self, replies, root=None):
        return Agent(
            client=StubClient(replies),
            tools=ToolRegistry([RunCommandTool(root=root or self.root)]),
        )

    def test_agent_can_run_a_command(self):
        self.write("ok.py", "print('命令结果')")
        agent = self.build(
            [
                tool_call_reply(
                    "run_command", json.dumps({"command": self.python("ok.py")})
                ),
                final_reply("跑完了"),
            ]
        )

        answer = agent.run("跑一下 ok.py")

        self.assertEqual(answer, "跑完了")
        self.assertIn("命令结果", agent.messages[3]["content"])

    def test_a_failing_command_does_not_break_the_turn(self):
        self.write("bad.py", "import sys\nsys.exit(1)")
        agent = self.build(
            [
                tool_call_reply(
                    "run_command", json.dumps({"command": self.python("bad.py")})
                ),
                final_reply("它失败了"),
            ]
        )

        answer = agent.run("跑一下 bad.py")

        self.assertEqual(answer, "它失败了")
        self.assertIn("退出码：1", agent.messages[3]["content"])

    def test_the_call_is_recorded_in_the_trace(self):
        self.write("ok.py", "print('x')")
        agent = self.build(
            [
                tool_call_reply(
                    "run_command", json.dumps({"command": self.python("ok.py")})
                ),
                final_reply(),
            ]
        )

        agent.run("跑一下")

        self.assertEqual(len(agent.get_trace()), 1)
        self.assertEqual(agent.get_trace()[0].tool, "run_command")
        self.assertTrue(agent.get_trace()[0].success)

    def slow_command(self, seconds: float = 5) -> str:
        return f'"{sys.executable}" -c "import time;time.sleep({seconds})"'

    def test_a_timed_out_command_does_not_break_the_turn(self):
        agent = self.build(
            [
                tool_call_reply(
                    "run_command",
                    json.dumps({"command": self.slow_command(), "timeout": 0.3}),
                ),
                final_reply("那条命令超时了"),
            ],
            root=self.timeout_root,
        )

        answer = agent.run("跑一条慢命令")

        self.assertEqual(answer, "那条命令超时了")
        self.assertIn("超时", agent.messages[3]["content"])

    def test_the_timeout_is_recorded_in_the_trace(self):
        # 超时是「工具正常返回了一个说超时的结果」，不是工具失败
        agent = self.build(
            [
                tool_call_reply(
                    "run_command",
                    json.dumps({"command": self.slow_command(), "timeout": 0.3}),
                ),
                final_reply(),
            ],
            root=self.timeout_root,
        )

        agent.run("跑一条慢命令")

        self.assertEqual(len(agent.get_trace()), 1)
        self.assertTrue(agent.get_trace()[0].success)
        self.assertEqual(agent.get_trace()[0].error, "")

    def test_a_timeout_can_be_followed_by_a_successful_command(self):
        agent = self.build(
            [
                tool_call_reply(
                    "run_command",
                    json.dumps({"command": self.slow_command(), "timeout": 0.3}),
                    call_id="c1",
                ),
                tool_call_reply(
                    "run_command",
                    json.dumps(
                        {"command": f'"{sys.executable}" -c "print(2+2)"'}
                    ),
                    call_id="c2",
                ),
                final_reply(),
            ],
            root=self.timeout_root,
        )

        agent.run("先跑慢的再跑快的")

        self.assertIn("超时", agent.messages[3]["content"])
        self.assertIn("4", agent.messages[5]["content"])
        self.assertEqual(len(agent.get_trace()), 2)

    def test_it_is_registered_in_main(self):
        # main.build_agent() 默认会去读 LLM 配置（API Key / 模型名）；把
        # 客户端整个换成替身，装配不需要任何真实配置，也不会联网
        import main

        with patch("main.LLMClient", StubClient):
            agent = main.build_agent()

        self.assertIn("run_command", [tool.name for tool in agent.tools.list()])


if __name__ == "__main__":
    unittest.main()
