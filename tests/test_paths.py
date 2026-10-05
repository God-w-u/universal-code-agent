"""tools/paths.py 的文件名规则：哪些名字一律不碰。

这一个判断被三处用着——遍历（list_files / search_code / 索引）、read_file、
edit_file。每一处都有自己的端到端用例（见 test_list_files.py 的
ListFilesToolSensitiveTest、test_read_file.py / test_edit_file.py 里的
同名用例）；这里钉的是**规则本身**，以及它「不依赖调用方先替它规范化」
这一条——端到端用例碰不到那个角落，因为 resolve() 会先把名字收拾干净。

全部离线，不联网。
"""

import unittest

from tools.paths import is_sensitive_file


class SensitiveNameTest(unittest.TestCase):
    def test_secret_names_are_sensitive(self):
        for name in (
            ".env",
            ".env.local",
            ".env.production",
            "server.pem",
            "deploy.key",
            "cert.p12",
            "store.pfx",
            "keystore.jks",
            "id_rsa",
            "id_rsa.pub",
            "id_ed25519",
            "my.ppk",
            ".netrc",
            ".pgpass",
            "credentials",
            "credentials.json",
            ".htpasswd",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_sensitive_file(name))

    def test_case_does_not_matter(self):
        # Windows 上尤其要：.ENV 和 .env 是同一个文件
        for name in (".ENV", "ID_RSA", "Server.PEM", "Credentials.JSON"):
            with self.subTest(name=name):
                self.assertTrue(is_sensitive_file(name))

    def test_trailing_dots_and_spaces_do_not_slip_through(self):
        # Windows 的文件名不允许以点或空格结尾，resolve() 也会剥掉它们——
        # 所以现在每条调用路径都碰不到这个角落。但规则本身不该指望调用方
        # 先替它规范化：将来谁拿一个没 resolve 过的 entry.name 来问，
        # ".env " 不能就这么溜过去。
        #
        # 只管**末尾**：开头的空格在 Windows 上是文件名的一部分，" .env"
        # 是另一个文件，不是躲起来的 .env。
        for name in (".env ", ".env.", "id_rsa  ", "server.pem ."):
            with self.subTest(name=name):
                self.assertTrue(is_sensitive_file(name))

    def test_the_public_templates_are_not_sensitive(self):
        for name in (".env.example", ".env.sample", ".env.template", ".ENV.EXAMPLE"):
            with self.subTest(name=name):
                self.assertFalse(is_sensitive_file(name))

    def test_lookalikes_are_not_sensitive(self):
        # 全名 / 通配匹配，不做子串匹配：这些都是普通文件
        for name in (
            "environment.py",
            "env",
            "keys.py",
            "keyboard.py",
            "my.env.example.txt",
            "credentials.txt",
            "notes.md",
            "main.py",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_sensitive_file(name))


if __name__ == "__main__":
    unittest.main()
