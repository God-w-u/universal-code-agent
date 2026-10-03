"""工具：跑这个项目的单元测试。

执行部分一点没自己写，全部交给 RunCommandTool：subprocess、超时、
输出解码、截断说明都已经在那边踩过坑也测过了，这里再抄一份等于把坑也抄一份。
这一层只负责一件事——把「跑测试」这个动作固定成一条确定的命令。
"""

from pathlib import Path

from tools.base import Tool
from tools.paths import check_root
from tools.run_command import RunCommandTool

# 第一版只支持这一条：项目自己的单元测试入口。
TEST_COMMAND = "python -m unittest discover -s tests -t ."

# 跑测试默认等多久。比 run_command 的 30 秒宽松：全量测试天然更慢
# （本项目 444 条用例实测约 7.5 秒），慢一点只是多等一会儿，
# 给少了则会把一次本来正常的测试跑判成超时，白等一轮还得重来。
DEFAULT_TEST_TIMEOUT = 120.0


class TestTool(Tool):
    """执行项目的单元测试，返回结果文本。"""

    name = "run_tests"
    description = (
        f"运行这个项目的单元测试（{TEST_COMMAND}），"
        "返回退出码、标准输出和标准错误。"
        "想确认「刚才的改动有没有弄坏什么」时直接用它，不用自己拼 run_command。"
        "测试不通过（退出码非 0）也会正常返回结果，错误信息就在输出里，不算调用失败。"
        "跑太久会被强制终止，同样按正常结果返回。"
    )
    parameters = {
        "timeout": {
            "type": "number",
            "description": f"最多等多少秒，默认 {DEFAULT_TEST_TIMEOUT:g}",
            "required": False,
        },
    }

    def __init__(
        self,
        root: Path | None = None,
        timeout: float = DEFAULT_TEST_TIMEOUT,
        command: str = TEST_COMMAND,
    ) -> None:
        self.root = check_root(root)
        # 默认是上面钉死的那条；留成参数只是为了让测试能换成一个假项目，
        # 不是给模型用的——它不在 parameters 里，模型改不了。
        self.command = command
        self.command_tool = RunCommandTool(root=self.root, timeout=timeout)

    def run_tests(self, timeout: float | None = None) -> str:
        """跑一次项目的单元测试，把 RunCommandTool 的结果原样返回。"""
        # 直接调 execute() 而不是 run()：command 是本工具自己定的常量，
        # 唯一来自外部的 timeout 已经过不了本工具自己的参数校验，
        # 没必要再过一遍内层工具的校验（而且显式传 None 会被判成「不是 number」）。
        return self.command_tool.execute(self.command, timeout=timeout)

    def execute(self, timeout: float | None = None) -> str:
        return self.run_tests(timeout=timeout)
