"""一键测试：用项目 `.venv` 里的 Python 跑全部 unittest（跨平台）。

用法（三种都行，结果一模一样）：

    run_tests.bat                    # Windows：双击或命令行，完全不依赖 PATH
    python run_tests.py              # 任何 Python：自动换到 .venv 里的解释器
    .venv/bin/python run_tests.py    # macOS / Linux

做的事只有一件——在项目根目录下跑：

    .venv 的 python -m unittest discover -s tests -t .

和直接敲上面那条命令的结果一字不差，只是把「找 .venv 解释器、设控制台
编码」这两件手工活收进来。多给的参数原样转给 unittest（例如 `-v`）。

刻意不做的事：

- **不联网、不需要 API Key、不加载任何真实模型**：测试全用替身对象；
  需要 Docker / 真实 embedding 模型的用例自己会 skip（见 README「一键测试」）。
- **不依赖全局 Python PATH**：解释器路径直接从项目里的 `.venv` 算出来；
  `.venv` 不存在就报错说清楚怎么建，绝不用系统 Python 凑合。
- **不吞退出码**：全过是 0，有失败 / 报错是非 0——CI 直接拿这个码当结论。
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def venv_python() -> Path:
    """项目 .venv 里 Python 解释器的路径（Windows 和别处目录不同）。"""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main(argv: list[str]) -> int:
    python = venv_python()
    if not python.exists():
        print(f"[一键测试] 没找到 {python}", file=sys.stderr)
        print(
            "[一键测试] 先建环境（见 README「环境安装」）：\n"
            "    python -m venv .venv\n"
            "    .venv 里装依赖：pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    args = ["-m", "unittest", "discover", "-s", "tests", "-t", ".", *argv]
    command = [str(python), *args]

    # 中文控制台（Windows 默认 GBK）下 unittest 的输出会炸编码；统一用
    # UTF-8——测试自己的子进程也会继承这条（有测试在钉 Windows 下的行为）
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    print(f"[一键测试] 解释器：{python}")
    print(f"[一键测试] 命令：python {' '.join(args)}")
    print("[一键测试] 不联网、不需要 API Key；需要 Docker / 真模型的用例会自动 skip")
    print("-" * 72)

    code = subprocess.call(command, cwd=ROOT, env=env)

    print("-" * 72)
    print(f"[一键测试] 退出码：{code}（0 = 全过；非 0 = 有失败，见上面的汇总）")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
