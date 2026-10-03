"""工具：在项目目录下执行一条命令，拿回退出码和输出。

这是第一个「会真的动系统」的工具：前面几个工具都是只读的，它不一样。
也是第一个会「主动打断」的工具：命令跑太久就杀掉。

安全边界复用 tools/paths.py（命令的工作目录固定在 root），
解码复用 tools/textfile.py（命令输出的编码和源码一样不保证）。

真正的执行是模块级的：run_process（起进程、超时杀掉）、decode_output
（解码回退）、render_result / render_timeout（结果文本的格式）。
RunCommandTool 只是其中一个调用方；Phase 10 的 Sandbox
（sandbox/local.py）复用同一套，不另写一份 subprocess 逻辑——
超时语义和输出截断全项目以这几个函数为唯一标准。
"""

import subprocess
from pathlib import Path

from tools.base import Tool, ToolError
from tools.paths import check_root
from tools.textfile import decode_text

# 不给 timeout 参数时默认等多少秒。
# 多数命令几秒内就结束；真正慢的（完整构建、全量测试）由调用方指定更大的值。
DEFAULT_TIMEOUT = 30.0

# 标准输出 / 标准错误各自最多返回这么多字符，超出就掐中间。
# RunCommandTool.max_output_chars 是它的别名，Sandbox 的结果文本也用这个数。
MAX_OUTPUT_CHARS = 20_000


def decode_output(raw: bytes, name: str) -> str:
    """把命令输出解成文本。

    这里和 read_file 的处理不一样：read_file 解不开就直接拒绝，
    命令输出解不开则退化成「把不能认的字节换成 �」继续返回。
    因为命令输出本来就可能是二进制 dump、彩色控制符、进度条，
    直接拒绝等于把整条结果丢掉，连退出码都看不到了。
    """
    if not raw:
        return ""

    try:
        return decode_text(raw, name)
    except ToolError:
        return raw.decode("utf-8", errors="replace")


def run_process(
    command: str,
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> tuple[int | None, bytes, bytes]:
    """在 cwd 下跑一条命令，返回 (退出码, 原始标准输出, 原始标准错误)。

    退出码为 None 表示超时被杀——这时两路输出都不保留（返回空字节），
    和工具一贯的语义一致。命令非空、timeout 大于 0 由调用方保证。

    这是全项目唯一真正 fork 进程的地方，RunCommandTool 和 Sandbox
    （sandbox/local.py）共用它，subprocess 的坑只在下面这一份注释里。
    """
    # 这里刻意不用 subprocess.run()，也不用 with 语句，两个坑都实测过：
    #
    # 1. subprocess.run(timeout=) 在 Windows + shell=True 下超时是假的。
    #    它内部 kill 掉 shell 之后，还会再调一次 communicate() 收残余输出，
    #    而 shell 的子进程还占着管道没退出，于是那一下一直等到命令自己跑完。
    #    实测：设 1 秒的超时，等了 8 秒才返回。
    # 2. with 语句退出时会 close() 管道，同样要排队等那个还堵在 read() 上的
    #    读取线程。实测：设 0.3 秒的超时，close() 又堵了 4.73 秒。
    #
    # 所以：communicate(timeout=) 负责准时叫停，kill() 负责动手，
    # 之后立刻返回，剩下的交给解释器。
    process = subprocess.Popen(
        command,
        shell=True,  # 直接跑命令字符串，"a && b"、管道、带参数都靠它
        cwd=cwd,  # 调用方保证已经在允许范围内
        env=env,  # None = 继承当前环境
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        # 只回收 shell 本身（SIGKILL / TerminateProcess 无法被忽略，不会卡住）。
        # 不碰管道：上面那 4.73 秒就是这么来的。
        process.wait()
        return None, b"", b""

    return process.returncode, stdout, stderr


def _clip(text: str, limit: int) -> str:
    """输出太长就掐掉中间，保留开头和结尾。

    保留结尾是必须的：跑测试时「3 failed, 374 passed」这种结论在最末尾，
    只留开头等于把最有用的那一句扔了。
    """
    if not text.strip():
        return "（空）"

    if len(text) <= limit:
        return text

    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n…（省略中间 {omitted} 个字符）\n{text[-half:]}"


def render_result(
    command: str,
    code: int,
    stdout: str,
    stderr: str,
    limit: int = MAX_OUTPUT_CHARS,
) -> str:
    """把「退出码 + 已解码的两路输出」渲染成结果文本。

    格式就是工具对外的那一段。Sandbox 的 CommandResult.as_text() 也走
    这里，两边的截断和文案只有一个出处。
    """
    if code == 0:
        head = f"命令：{command}\n退出码：0"
    else:
        # 明说一句，省得模型只盯着输出、把「命令其实失败了」看漏
        head = f"命令：{command}\n退出码：{code}（非 0，命令失败）"

    return (
        f"{head}\n\n"
        f"标准输出：\n{_clip(stdout, limit)}\n\n"
        f"标准错误：\n{_clip(stderr, limit)}"
    )


def render_timeout(command: str, seconds: float) -> str:
    """超时也走正常返回这条路，不抛异常。"""
    return (
        f"命令：{command}\n"
        f"结果：超时，已强制终止（超过 {seconds:g} 秒仍未结束）\n\n"
        f"标准输出：\n（命令超时被终止，输出没有保留）\n\n"
        f"标准错误：\n（命令超时被终止，输出没有保留）\n\n"
        f"…（可以调大 timeout 再试。等待输入的命令永远不会自己结束，"
        f"别重试，换个跑法）"
    )


class RunCommandTool(Tool):
    """在项目根目录下执行一条命令。"""

    name = "run_command"
    description = (
        "在项目根目录下执行一条命令，返回退出码、标准输出和标准错误。"
        "可以用来跑构建、跑测试、调用编译器或解释器。"
        "退出码非 0 也会正常返回结果（错误信息就在里面），不算调用失败。"
        "命令超过 timeout 秒还没结束会被强制终止，同样按正常结果返回。"
    )
    parameters = {
        "command": {
            "type": "string",
            "description": (
                "要执行的命令，例如 python -m unittest discover -s tests -t ."
            ),
            "required": True,
        },
        "timeout": {
            "type": "number",
            "description": (
                f"最多等多少秒，默认 {DEFAULT_TIMEOUT:g}；"
                "跑完整构建或全量测试时可以调大"
            ),
            "required": False,
        },
    }

    # 标准输出 / 标准错误各自最多返回这么多字符，超出就掐中间
    max_output_chars = MAX_OUTPUT_CHARS

    def __init__(
        self, root: Path | None = None, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout 必须大于 0，收到 {timeout}")

        self.root = check_root(root)
        self.timeout = timeout

    def execute(
        self,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        return self.run_with_code(command, timeout=timeout, env=env)[1]

    def run_with_code(
        self,
        command: str,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int | None, str]:
        """执行命令，返回 (退出码, 结果文本)。

        退出码为 None 表示命令超时被杀掉了，那种情况下没有退出码可言。
        一次要跑好几步的工具（build 就是：配置没成功就不用接着编译了）
        靠它判断「这一步过没过」；只有一个命令的调用方用 execute() 就够了。

        env 是给 MSVC 那条路用的：cl.exe 不在 PATH 上，得先跑一遍
        vcvars64.bat 把环境铺好（见 tools/vsenv.py），再把那份环境传进来。
        传 None 就是继承当前进程的环境，也就是原来的行为。
        它只影响子进程收到哪些变量，不参与命令字符串的拼接。
        """
        if not command.strip():
            # 空命令几乎肯定是模型写错了，直接说清楚，别让它跑出一个
            # 「退出码 0、什么也没输出」的空结果
            raise ToolError("命令不能为空")

        seconds = self.timeout if timeout is None else timeout
        if seconds <= 0:
            raise ToolError(f"timeout 必须大于 0，收到 {seconds}")

        # 真正跑进程的是模块级的 run_process（那几个 Windows 坑的注释
        # 就在它头上）。工作目录固定在 root，相对路径不会溜到别处去。
        code, stdout, stderr = run_process(
            command, cwd=self.root, timeout=seconds, env=env
        )
        if code is None:
            return None, self._timeout_result(command, seconds)

        return code, self._render(
            command,
            code,
            decode_output(stdout, "标准输出"),
            decode_output(stderr, "标准错误"),
        )

    def _render(self, command: str, code: int, stdout: str, stderr: str) -> str:
        # 薄壳：格式的实现在模块级 render_result，Sandbox 的 as_text() 共用
        # 的就是那一份，两边永远不会各自漂移。
        return render_result(command, code, stdout, stderr, self.max_output_chars)

    def _timeout_result(self, command: str, seconds: float) -> str:
        return render_timeout(command, seconds)
