"""命令行入口：用户 → Agent.run()（内部含工具调用循环）→ 回答。

运行：
    python main.py

输入 exit 或 quit 退出。
"""

import json
import sys

from agent import Agent, AgentError, RepairHistory, TraceEvent
from config import ConfigError, load_search_embedding
from llm import LLMError
from tools import (
    BuildTool,
    CalculatorTool,
    CodeSummaryTool,
    CompileTool,
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    RunCommandTool,
    SearchChunksTool,
    SearchCodeTool,
    TestTool,
    ToolRegistry,
)

EXIT_WORDS = {"exit", "quit"}


def build_registry() -> ToolRegistry:
    """注册当前项目提供的全部工具。

    `search_code_chunks` 的 embedding 模式从环境变量读（9.4）：默认 off
    （纯字面，行为和以前一字不差）；填 fake / local / multilingual 才接
    语义检索。这里只读一个字符串——**不加载模型**，模型到第一次真调这个
    工具时才加载（见 tools/search_chunks.py）。
    """
    return ToolRegistry(
        [
            BuildTool(),
            CalculatorTool(),
            CompileTool(),
            ListFilesTool(),
            ReadFileTool(),
            SearchCodeTool(),
            SearchChunksTool(embedding=load_search_embedding()),
            CodeSummaryTool(),
            RunCommandTool(),
            TestTool(),
            EditFileTool(),
        ]
    )


def build_agent() -> Agent:
    """装配 Agent，注册当前项目提供的全部工具。

    顺手装上 RepairHistory：每轮对话跑完，修复过程会写一份 JSON 到
    repair_history/（默认位置，见 agent/history.py），过后能翻。
    """
    return Agent(history=RepairHistory(), tools=build_registry())


def print_trace(events: list[TraceEvent]) -> None:
    """把这一轮的工具调用简单打出来，方便看清 Agent 到底干了什么。"""
    if not events:
        print("\n[Trace] 本轮没有调用工具")
        return

    for event in events:
        arguments = json.dumps(event.arguments, ensure_ascii=False)
        print(f"\n[Tool] {event.tool}\narguments: {arguments}")
        print(
            f"\n[Tool Result]\nsuccess={str(event.success).lower()}\n"
            f"duration={event.duration_seconds:.2f}s"
        )
        if event.error:
            print(f"error: {event.error}")


def main() -> int:
    try:
        agent = build_agent()
    except ConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 1

    tool_names = ", ".join(tool.name for tool in agent.tools.list()) or "无"
    # embedding 模式再读一次只为打在横幅上（纯环境变量读取，不加载模型）；
    # build_agent 里已经读过一次，值不合法在那里就已经报错了
    print(
        f"模型：{agent.client.config.model}  工具：{tool_names}"
        f"  search_code_chunks embedding：{load_search_embedding()}（输入 exit 退出）"
    )

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            break

        try:
            answer = agent.run(user_input)
        except LLMError as exc:
            print(f"[调用失败] {exc}", file=sys.stderr)
            continue
        except AgentError as exc:
            print(f"[Agent 失败] {exc}", file=sys.stderr)
            continue
        finally:
            # 放在 finally 里：工具连续失败、轮数超限这些「最需要看 Trace」的
            # 场景全是走异常分支出去的，写在 try 后面根本轮不到。
            # 正常返回时它照样在下面的「最终回答」之前打印，顺序没变。
            print_trace(agent.get_trace())

        # 历史记录没写成不算这一轮失败，但得让人知道：不然翻记录时会以为
        # 「这段时间没修过东西」
        if agent.history is not None and agent.history.last_error:
            print(f"[历史记录] {agent.history.last_error}", file=sys.stderr)

        print(f"\n最终回答：\n{answer}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
