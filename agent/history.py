"""每次 run() 的修复过程落一份 JSON，过后还能翻出来看。

RepairLoop 和 Repair Summary 都只在内存里：run() 一返回，那几轮改了哪儿、
哪一关没过、为什么停，就只剩屏幕上那几行文字了。这个模块把它写成文件——
一个 run() 一份，放在项目根目录下的 repair_history/ 里。

写下来的东西**照抄 RepairRound**，一个字都不重算：轮次、文件、改了哪一处、
指纹、验证命令、退出码、诊断变化、编译 / 构建过没过、测试结果——都是
agent/repair.py 里已经有的字段。报告里怎么说的，文件里就怎么记，两边不会
各说各话（7.6 那句 Repair Summary 也是原样抄进去的）。

几条刻意的边界：

- **只是记录，不是判断**：写不写、写得成写不成，都改不了修复的结果。RepairLoop
  该怎么修还怎么修，它根本不知道有这个模块存在。判断和记录的代码也不在一处。
- **写失败不往外抛**：目录建不出来、盘满、文件被占着——一律只在 last_error
  里留一句话。用户的任务已经做完了，不能因为「附带记一笔」失败就把它带崩。
- **一份一个文件名**：时间打头（人一眼看得出生在什么时候）+ 一小段随机数
  （机器保证不会重名）。绝不覆盖之前任何一份——历史记录被后来的覆盖掉，
  那就不叫历史了。
- **临时文件 + os.replace**：和 edit_file 落盘是同一套。半截 JSON 比没有更糟：
  翻的时候解不开，还不知道是这次没记上还是文件本来就这样。
- **只在 run() 正常跑完时记**：中途抛异常（模型调用失败、工具轮数超限之外
  的失败）那一轮本来就整个回滚了，没有「结果」可记。
"""

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

from agent.repair import RepairLoop, RepairRound
from agent.verify import Comparison
from config import PROJECT_ROOT

# 历史记录放哪儿：项目根目录下单独一个名字。名字直白，不跟别的文件混在一起，
# 想清空就是删掉这一个目录。
DIRECTORY_NAME = "repair_history"

# 一次 run() 的最终状态，三种。看的是「有没有修复轮」和「停没停」，
# 不是「模型说自己修好没有」。
NO_REPAIR = "没修"  # 一次修复都没发生（普通问答、改完就过的一次修改）
FIXED = "修好了"  # 修完了，没有停下来的理由
UNFIXED = "没修好"  # 停了，或者还欠着一次没验过的修改


class RepairHistory:
    """把每次 run() 的结果写成一个 JSON 文件。

    - directory：写到哪儿，默认是项目根目录下的 repair_history/，不存在会自己建。
    - record() 每调一次就新起一个文件，不覆盖之前任何一份。
    - 写失败不往外抛：原因留在 last_error 里，谁想知道谁去看。
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = (
            Path(directory)
            if directory is not None
            else PROJECT_ROOT / DIRECTORY_NAME
        )
        self.last_path: Path | None = None  # 最近一次写成的文件；失败时为 None
        self.last_error: str | None = None  # 最近一次没写成的原因；成功时清空
        self.paths: list[Path] = []  # 这一路写成的所有文件，按先后

    def record(self, request: str, repair: RepairLoop) -> Path | None:
        """记一次 run()。写成了返回那个文件，没写成返回 None。

        这里故意抓得很宽（连 json 序列化本身出毛病也算）：写历史是附带的事，
        绝不能因为它把一个已经做完的任务变成失败。真出事了，原因原样留在
        last_error 里——不吞掉，只是不往外炸。
        """
        try:
            path = self._write(_document(request, repair))
        except Exception as exc:  # noqa: BLE001 —— 「绝不往外抛」就是这个函数的职责
            self.last_path = None
            self.last_error = f"这次修复记录没能写进 {self.directory}：{exc}"
            return None

        self.last_path = path
        self.last_error = None
        self.paths.append(path)
        return path

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------

    def _write(self, document: dict) -> Path:
        """把这份记录写成一个新文件：目录按需建，临时文件写完再原子换名。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
        target = self._free_name()

        try:
            handle = tempfile.NamedTemporaryFile(
                dir=self.directory,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            )
        except OSError as exc:
            raise OSError(f"在 {self.directory} 里建不出临时文件：{exc}") from exc

        temp = Path(handle.name)
        try:
            with handle:
                handle.write(data)
                handle.flush()
                # 刷到盘上再换名：不然系统一崩，换过去的名字指向的可能还是空文件
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise OSError(f"写 {target.name} 失败：{exc}") from exc

        return target

    def _free_name(self) -> Path:
        """挑一个还没被占用的文件名：时间打头，后面跟一小段随机数。

        时间那截是给人看的（什么时候跑的），随机那截是给机器的：同一秒里
        连跑两次，时间一模一样，光靠它就得覆盖。理论上撞不上，撞上了就再抽
        一个——宁可重抽，也不覆盖以前那份。
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        while True:
            target = self.directory / f"{stamp}-{uuid.uuid4().hex[:6]}.json"
            if not target.exists():
                return target


# ---------------------------------------------------------------------------
# 一份记录长什么样
# ---------------------------------------------------------------------------


def _document(request: str, repair: RepairLoop) -> dict:
    """这一轮 run() 的全部内容，就是那个 JSON 的结构。"""
    return {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "request": request,  # 用户让 Agent 干什么，翻记录时先看这一条
        "status": _status(repair),
        "fixed_round": _fixed_round(repair),  # 第几轮修好的；没修好是 null
        "stopped": repair.stopped,  # 没停就是 null
        "summary": repair.summary(),  # 给用户看的那句话；没修过是空串
        "round_count": len(repair.rounds),
        "rounds": [_round(item) for item in repair.rounds],
    }


def _status(repair: RepairLoop) -> str:
    """这次 run() 的最后状态。一轮没修也算记录，只是状态是「没修」。"""
    if not repair.rounds:
        return NO_REPAIR
    if repair.stopped is not None or repair.pending:
        return UNFIXED
    return FIXED


def _fixed_round(repair: RepairLoop) -> int | None:
    """这次是在第几轮修好的；没修好（或者压根没改代码）就是 null。

    全从已有的东西推：`_status` 说「修好了」，就说明没停、不欠、而且有轮次。
    这种情况下**最后一轮一定是过了的那一轮**——没过的话 pending 还欠着，
    或者早就因为某条停止规矩停下来了（那 status 就不是「修好了」）。前面的
    失败轮不算数：它们只是路上的台阶，不是修好的那一轮。

    所以 `fixed_round` 有值 ⟺ `status` 是「修好了」，两个字段不会各说各话。
    """
    if _status(repair) != FIXED:
        return None
    return repair.rounds[-1].number


def _round(item: RepairRound) -> dict:
    """一轮记录。字段全部照抄 RepairRound，不重新算一遍。"""
    return {
        "round": item.number,
        "file": item.path,
        "edit": item.edit,
        "fingerprint": item.fingerprint,
        "state": item.state,
        "compile": {
            "command": item.verifier,
            "passed": item.compiled,
            "exit_code": item.exit_code,
        },
        "diagnostics": _diagnostics(item.diff),
        "test": (
            None
            if item.test_passed is None
            else {
                "passed": item.test_passed,
                "exit_code": item.test_exit_code,
                "detail": item.test,
            }
        ),
    }


def _diagnostics(diff: Comparison | None) -> dict | None:
    """诊断变化：消失了哪些、还在哪些、新冒出哪些。没有基线时是 null。

    每条都用 as_line() 渲染成「文件:行:列: 级别: 消息」那一套——和模型、
    和用户看到的写法一致，翻记录的人不用再学一种格式。
    """
    if diff is None:
        return None
    return {
        "gone": [item.as_line() for item in diff.gone],
        "still": [item.as_line() for item in diff.still],
        "added": [item.as_line() for item in diff.added],
    }
