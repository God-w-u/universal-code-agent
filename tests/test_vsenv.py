"""tools/vsenv.py 测试：找 VS、跑 vcvars、把环境解析出来。

这一层打交道的全是外部进程（vswhere、cmd、vcvars64.bat），真跑起来结果
就取决于这台机器装没装 VS 了。所以这里用的是**真的文件系统 + 假的进程**：
临时目录里搭出 VS 那套目录结构，把 subprocess.run 换掉。
钉住的是「怎么找、怎么解析、找不到怎么办」，不是「微软的脚本对不对」。

全部离线，不联网。
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import vsenv

# 假的 vcvars 输出：set 吐出来的就是这个样子。
# 第二行刻意带两个 =，值里本来就可能出现它（PATH 里有 =C: 这类条目）
FAKE_SET = (
    "PATH=C:\\VS\\bin;C:\\Windows\\System32\r\n"
    "INCLUDE=C:\\VS\\include\r\n"
    "LIB=C:\\VS\\lib\r\n"
    "=C:=C:\\project\r\n"
)


def completed(stdout: bytes = b"", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=b"")


class VsEnvTestCase(unittest.TestCase):
    """搭一个假的 VS 安装：<临时目录>/Microsoft Visual Studio/Installer/vswhere.exe
    和 <另一个临时目录>/VC/Auxiliary/Build/vcvars64.bat。"""

    def setUp(self):
        vsenv.reset_cache()
        self.addCleanup(vsenv.reset_cache)

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()

        # 装 vswhere 的地方，靠 ProgramFiles(x86) 找到
        self.programs = self.base / "Programs"
        self.installer = self.programs / "Microsoft Visual Studio" / "Installer"
        self.installer.mkdir(parents=True)
        self.vswhere = self.installer / "vswhere.exe"
        self.vswhere.write_text("假的 vswhere", encoding="utf-8")

        # VS 自己装在哪
        self.install = self.base / "VS"
        self.vcvars = self.install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        self.vcvars.parent.mkdir(parents=True)
        self.vcvars.write_text("@echo off\r\n", encoding="utf-8")

        env = patch.dict(os.environ, {"ProgramFiles(x86)": str(self.programs)})
        env.start()
        self.addCleanup(env.stop)

    def installed_path(self, *lines: str) -> bytes:
        """假装 vswhere 回答了 VS 装在哪。"""
        return ("\r\n".join(lines) + "\r\n").encode("utf-8")


class FindVswhereTest(VsEnvTestCase):
    def test_it_is_found_under_program_files(self):
        self.assertEqual(vsenv.find_vswhere(), self.vswhere)

    def test_a_missing_file_gives_none(self):
        self.vswhere.unlink()

        self.assertIsNone(vsenv.find_vswhere())

    def test_the_environment_variable_is_respected(self):
        # 不写死 C 盘：换个 ProgramFiles(x86) 就得跟着换地方找
        with patch.dict(os.environ, {"ProgramFiles(x86)": str(self.base / "nope")}):
            self.assertIsNone(vsenv.find_vswhere())


class FindVcvarsTest(VsEnvTestCase):
    def run_vswhere(self, stdout: bytes = b"", returncode: int = 0, effect=None):
        return patch(
            "tools.vsenv.subprocess.run",
            return_value=completed(stdout, returncode),
            side_effect=effect,
        )

    def test_it_asks_vswhere_where_vs_is(self):
        with self.run_vswhere(self.installed_path(str(self.install))) as run:
            self.assertEqual(vsenv.find_vcvars(), self.vcvars)

        asked = run.call_args[0][0]
        self.assertIn(str(self.vswhere), asked)
        self.assertIn("-property", asked)
        self.assertIn("installationPath", asked)

    def test_it_asks_for_a_vs_that_has_the_cpp_toolset(self):
        # 不筛的话，只装了 C# 的 VS 也会被选中，然后发现没有 vcvars
        with self.run_vswhere(self.installed_path(str(self.install))) as run:
            vsenv.find_vcvars()

        asked = run.call_args[0][0]
        self.assertIn("-requires", asked)
        self.assertIn(vsenv.VC_COMPONENT, asked)

    def test_it_returns_none_when_vswhere_is_missing(self):
        self.vswhere.unlink()

        with self.run_vswhere(self.installed_path(str(self.install))) as run:
            self.assertIsNone(vsenv.find_vcvars())

        run.assert_not_called()  # 没有 vswhere 就没什么可问的

    def test_a_failing_vswhere_gives_none(self):
        with self.run_vswhere(b"", returncode=1):
            self.assertIsNone(vsenv.find_vcvars())

    def test_an_empty_answer_gives_none(self):
        # 一台装了 VS 的机器也没有：vswhere 正常退出，但什么都不输出
        with self.run_vswhere(b""):
            self.assertIsNone(vsenv.find_vcvars())

    def test_a_path_without_vcvars_in_it_gives_none(self):
        # 装了个没有 C++ 的 VS：安装目录是真的，但里面没有 vcvars64.bat
        with self.run_vswhere(self.installed_path(str(self.base / "别的"))):
            self.assertIsNone(vsenv.find_vcvars())

    def test_extra_lines_are_skipped(self):
        # vswhere 输出里混进空行时不能当成「安装在一个空路径下」
        with self.run_vswhere(self.installed_path("", str(self.install), "")):
            self.assertEqual(vsenv.find_vcvars(), self.vcvars)

    def test_a_broken_vswhere_gives_none_instead_of_raising(self):
        # 超时、权限不足之类的，只该让 VS 用不了，不该让整个工具炸掉
        for effect in (subprocess.TimeoutExpired("vswhere", 1), OSError("起不来")):
            with self.subTest(effect=type(effect).__name__):
                with patch("tools.vsenv.subprocess.run", side_effect=effect):
                    self.assertIsNone(vsenv.find_vcvars())


class VcvarsEnvTest(VsEnvTestCase):
    def setUp(self):
        super().setUp()
        self.answers = [
            completed(self.installed_path(str(self.install))),  # vswhere
            completed(FAKE_SET.encode("utf-8")),  # vcvars
        ]

    def run_both(self, vcvars_result=None):
        if vcvars_result is not None:
            self.answers[1] = vcvars_result
        return patch("tools.vsenv.subprocess.run", side_effect=self.answers)

    def test_the_environment_is_parsed(self):
        with self.run_both():
            env = vsenv.vcvars_env()

        self.assertEqual(env["INCLUDE"], r"C:\VS\include")
        self.assertEqual(env["LIB"], r"C:\VS\lib")
        self.assertIn(r"C:\VS\bin", env["PATH"])

    def test_a_value_containing_an_equals_sign_survives(self):
        # 只切第一个 =：路径里出现等号是常事，切错了整条 PATH 就废了
        with self.run_both(completed(b"PATH=a=b=c\r\n")):
            env = vsenv.vcvars_env()

        self.assertEqual(env["PATH"], "a=b=c")

    def test_cmd_per_drive_entries_are_dropped(self):
        # `=C:=C:\project` 是 cmd 给每个盘记当前目录用的。它的名字以 = 开头，
        # 按第一个 = 切出来就是空串——留着就会在环境里塞一个空名字的变量
        with self.run_both():
            env = vsenv.vcvars_env()

        self.assertNotIn("", env)
        self.assertNotIn("=C:", env)

    def test_it_calls_vcvars_and_then_dumps_the_environment(self):
        with self.run_both() as run:
            vsenv.vcvars_env()

        command = run.call_args_list[1][0][0]
        self.assertIn(str(self.vcvars), command)
        self.assertIn("call", command)
        self.assertIn("&& set", command)
        # 横幅要挡掉，不然它会被当成一行环境变量解析
        self.assertIn(">nul", command)

    def test_a_failing_vcvars_gives_none(self):
        with self.run_both(completed(b"", returncode=1)):
            self.assertIsNone(vsenv.vcvars_env())

    def test_vcvars_that_prints_nothing_gives_none(self):
        # 解析不出任何变量时不能返回一个空字典：那是「铺好了」的意思
        with self.run_both(completed(b"")):
            self.assertIsNone(vsenv.vcvars_env())

    def test_a_broken_vcvars_gives_none_instead_of_raising(self):
        with patch("tools.vsenv.subprocess.run", side_effect=OSError("起不来")):
            self.assertIsNone(vsenv.vcvars_env())

    def test_no_vs_means_no_cmd_at_all(self):
        self.vswhere.unlink()

        with self.run_both() as run:
            self.assertIsNone(vsenv.vcvars_env())

        run.assert_not_called()

    def test_the_result_is_cached(self):
        # vcvars 一次要跑两秒多，同一台机器上没理由每次编译都重跑
        with self.run_both() as run:
            first = vsenv.vcvars_env()
            second = vsenv.vcvars_env()

        self.assertIs(first, second)
        self.assertEqual(run.call_count, 2)  # vswhere 一次 + vcvars 一次

    def test_a_failure_is_cached_too(self):
        # 「没有 VS」这个结论一样贵：每次编译都去问一遍 vswhere 也是白等
        self.vswhere.unlink()

        with self.run_both() as run:
            vsenv.vcvars_env()
            vsenv.vcvars_env()

        run.assert_not_called()

    def test_reset_cache_forces_a_new_look(self):
        with self.run_both():
            vsenv.vcvars_env()

        vsenv.reset_cache()

        with self.run_both() as run:
            vsenv.vcvars_env()

        self.assertEqual(run.call_count, 2)


class FindProgramTest(VsEnvTestCase):
    def test_it_prefers_the_current_path(self):
        # agent 本来就是从「Developer Command Prompt」里起来的：cl 直接就在
        # PATH 上，没必要再去起一次 vcvars
        with patch("tools.vsenv.shutil.which", return_value=r"C:\nowhere\cl.exe"):
            with patch("tools.vsenv.vcvars_env") as env:
                self.assertEqual(vsenv.find_program("cl"), r"C:\nowhere\cl.exe")

        env.assert_not_called()

    def test_it_falls_back_to_the_vs_environment(self):
        env = {"PATH": r"C:\VS\bin"}
        with patch("tools.vsenv.shutil.which", side_effect=[None, r"C:\VS\bin\cl.exe"]):
            with patch("tools.vsenv.vcvars_env", return_value=env):
                self.assertEqual(vsenv.find_program("cl"), r"C:\VS\bin\cl.exe")

    def test_the_vs_path_is_the_one_searched(self):
        with patch("tools.vsenv.shutil.which", side_effect=[None, None]) as which:
            with patch("tools.vsenv.vcvars_env", return_value={"PATH": "只在这条里找"}):
                vsenv.find_program("cl")

        self.assertEqual(which.call_args_list[1].kwargs["path"], "只在这条里找")

    def test_no_vs_gives_none(self):
        with patch("tools.vsenv.shutil.which", return_value=None):
            with patch("tools.vsenv.vcvars_env", return_value=None):
                self.assertIsNone(vsenv.find_program("cl"))

    def test_an_empty_path_in_the_environment_does_not_raise(self):
        with patch("tools.vsenv.shutil.which", return_value=None):
            with patch("tools.vsenv.vcvars_env", return_value={}):
                self.assertIsNone(vsenv.find_program("cl"))


class DecodeTest(unittest.TestCase):
    def test_gbk_output_is_decoded(self):
        # 中文 Windows 上 cmd 吐的是 GBK，源码那一套解码在这儿一样要用
        self.assertEqual(vsenv._text("路径".encode("gbk")), "路径")

    def test_undecodable_bytes_do_not_raise(self):
        # 解错一位的结果只是「找不到 VS」，不该让调用方炸掉
        self.assertIsInstance(vsenv._text(b"\xff\xfe\x00\x01"), str)


if __name__ == "__main__":
    unittest.main()
