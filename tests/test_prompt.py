"""系统提示词测试。

提示词是纯文本，「模型看完会不会变乖」没法在这里证——那要靠真跑一遍。
能钉住的是三件事：

1. 环境那段填的是运行时的真值（项目根目录、操作系统），不是写死的字符串；
2. 提示词和工具注册表对得上：注册表里的工具，提示词里得点到名；
   提示词里点到的名字，注册表里得真有。工具改名或者新增时，这条会把
   提示词一起拉进来改，不会留下一个在说废话的提示词；
3. Agent 默认用的确实是它，外面传了 system_prompt 时又确实听传进来的。

全部离线，不联网。
"""

import platform
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import DEFAULT_SYSTEM_PROMPT, Agent, build_system_prompt
from config import PROJECT_ROOT
from tools import ToolRegistry

PROMPT = DEFAULT_SYSTEM_PROMPT

# calculator 是最早写来演示工具协议的例子，跟代码分析没关系，提示词里
# 刻意不提它。别的工具都得在提示词里有一席之地——注册了一个工具却不说
# 什么时候该用，等于没注册。
NOT_MENTIONED_ON_PURPOSE = ("calculator",)

# 提示词里出现的 snake_case 词。工具名都是这个形状，所以拿它来捞「提示词
# 提到了哪些工具」。offset、cmd.exe、/bin/sh 这些都不会被捞进来。
TOOL_SHAPED = re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b")

# 但 snake_case 不等于工具名：提示词举例说明时写过一个真函数名，它也会被上面那条
# 正则捞出来。这样的词列在这里。当前是空的，留着是因为下次再举例时还得用。
# 和上面的豁免一样，这里不能留已经不在提示词里的词——有一条测试盯着。
NOT_TOOLS = ()


class StubClient:
    """占位客户端。这里只关心 Agent 怎么装配，不需要它真回话。"""

    def chat(self, messages, tools=None):
        return ""

    def chat_with_tools(self, messages, tools):
        return {"role": "assistant", "content": ""}


class PromptAndRegistryTest(unittest.TestCase):
    """提示词和注册表是两份会各自漂移的东西，得互相对着看。"""

    @classmethod
    def setUpClass(cls):
        # main.build_agent() 会去读 API Key，这里把客户端换掉再装配
        import main

        with patch("agent.agent.LLMClient", StubClient):
            cls.registry: ToolRegistry = main.build_agent().tools

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.registry.list()]

    def test_every_registered_tool_is_mentioned(self):
        for name in self.tool_names():
            if name in NOT_MENTIONED_ON_PURPOSE:
                continue
            with self.subTest(tool=name):
                self.assertIn(
                    name,
                    PROMPT,
                    f"注册了 {name} 却没在提示词里说什么时候用它",
                )

    def test_the_prompt_names_no_tool_that_does_not_exist(self):
        # 反向：提示词里提到的工具名，注册表里必须真有。
        # 工具改了名而提示词没跟着改的话，模型会去调一个不存在的工具。
        known = set(self.tool_names())
        mentioned = set(TOOL_SHAPED.findall(PROMPT)) - set(NOT_TOOLS)

        self.assertTrue(mentioned, "一个工具名都没捞到，多半是提示词被改空了")
        self.assertEqual(
            mentioned - known,
            set(),
            "提示词里出现了不存在的工具名。真的是工具就注册它，"
            "只是举例写的函数名就加进 NOT_TOOLS。",
        )

    def test_the_not_tools_list_is_not_stale(self):
        # 豁免名单会过期：提示词里的例子删了，名单却不删，下次真有个工具
        # 叫这个名字就被无声放过了。
        for word in NOT_TOOLS:
            with self.subTest(word=word):
                self.assertIn(word, PROMPT)

    def test_the_exception_is_still_registered(self):
        # 上面那条跳过 calculator 是故意的；万一它哪天被删了，
        # 这行会提醒把豁免名单也删掉，别留个空豁免。
        for name in NOT_MENTIONED_ON_PURPOSE:
            self.assertIn(name, self.tool_names())


class PromptEnvironmentTest(unittest.TestCase):
    def test_it_names_the_real_project_root(self):
        # 不写死路径：提示词得跟着项目实际在哪儿走
        self.assertIn(str(PROJECT_ROOT), PROMPT)

    def test_it_names_the_operating_system(self):
        self.assertIn(platform.system(), PROMPT)

    def test_the_default_root_is_the_project_root(self):
        self.assertEqual(build_system_prompt(), PROMPT)

    def test_a_different_root_shows_up_in_the_prompt(self):
        other = Path("D:/somewhere/else")

        prompt = build_system_prompt(root=other)

        self.assertIn(str(other), prompt)
        self.assertNotIn(str(PROJECT_ROOT), prompt)

    def test_a_root_can_be_passed_as_a_plain_string(self):
        # 传字符串进来也会被 Path 归一化（Windows 上 / 会变成 \），
        # 所以比的得是归一化之后的样子
        self.assertIn(
            str(Path("D:/somewhere/else")),
            build_system_prompt(root="D:/somewhere/else"),
        )


class PromptContentTest(unittest.TestCase):
    """这些不是「文字里出现文字」的废话测试：每条都对应一个真踩过的坑，
    删掉就会退回去。"""

    def test_it_tells_the_model_paths_are_relative(self):
        self.assertIn("相对路径", PROMPT)

    def test_it_tells_the_model_to_cite_file_and_line(self):
        # 实测：模型会说「大概第 105 行左右」——用户没法按这个去核对
        self.assertIn("行号", PROMPT)

    def test_it_points_at_read_file_as_the_source_of_line_numbers(self):
        # 这条换过两版。第一版只写「行号要准」，模型就给出一堆错行号——
        # read_file 那时候返回的是不带行号的原文，它只能自己数，而它数不准。
        # 第二版改成「行号只能从 search_code 拿」，准确了但引用变少：
        # search_code 只能给出含关键词的那几行。现在 read_file 自己带行号了，
        # 就该以它为准。
        self.assertIn("以 read_file 返回的", PROMPT)
        self.assertIn("42|", PROMPT)
        self.assertNotIn("只能从 search_code", PROMPT)

    def test_it_tells_the_model_not_to_fill_in_what_it_did_not_see(self):
        self.assertIn("没看到", PROMPT)

    def test_it_says_a_non_zero_exit_code_is_not_a_tool_failure(self):
        # 这几个工具「失败」时也是正常返回，模型容易把退出码非 0 当成自己出错了
        self.assertIn("退出码非 0", PROMPT)

    def test_it_says_calls_can_be_batched(self):
        # 一轮里多批几个调用只算一步，对 max_steps 的余量影响很大
        self.assertIn("一起发", PROMPT)

    def test_it_says_code_summary_covers_c_and_cpp(self):
        # 这条原来断言的是「只认 Python」。工具支持 C/C++ 之后提示词就得跟着改，
        # 不然模型会绕开 code_summary 逐个 read_file——那正是它要省掉的事
        self.assertIn("C/C++", PROMPT)
        self.assertNotIn("只认 Python", PROMPT)

    def test_it_says_what_code_summary_cannot_do_for_cpp(self):
        # 逐行扫出来的近似结果，漏是常态。不说清楚的话，模型会把「没报出来」
        # 当成「没有」，然后拿这个下结论
        self.assertIn("近似", PROMPT)
        self.assertIn("宏展开", PROMPT)

    def test_it_says_a_long_file_cannot_be_read_whole(self):
        self.assertIn("看不全", PROMPT)

    def test_it_warns_that_output_can_be_truncated(self):
        self.assertIn("省略", PROMPT)

    def test_it_does_not_repeat_the_tool_schemas(self):
        # 参数说明在工具的 description 里，模型本来就收得到；
        # 提示词再抄一遍，改工具时就有第二处要改，迟早对不上。
        for duplicated in ("timeout", "source_root", "build_dir", "case_sensitive"):
            with self.subTest(param=duplicated):
                self.assertNotIn(duplicated, PROMPT)


class AgentUsesPromptTest(unittest.TestCase):
    def test_the_agent_defaults_to_it(self):
        self.assertEqual(Agent(client=StubClient()).system_prompt, PROMPT)

    def test_it_is_the_first_message(self):
        agent = Agent(client=StubClient())

        self.assertEqual(
            agent.messages[0], {"role": "system", "content": PROMPT}
        )

    def test_an_explicit_prompt_wins(self):
        agent = Agent(client=StubClient(), system_prompt="SYS")

        self.assertEqual(agent.system_prompt, "SYS")
        self.assertEqual(agent.messages[0]["content"], "SYS")

    def test_an_empty_prompt_is_honoured(self):
        # 传空串是「不要系统提示词」，不是「没传」——别被当成默认值
        agent = Agent(client=StubClient(), system_prompt="")

        self.assertEqual(agent.messages[0]["content"], "")


if __name__ == "__main__":
    unittest.main()
