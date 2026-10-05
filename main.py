"""命令行入口：用户 → Agent.run()（内部含工具调用循环）→ 回答。

运行：
    python main.py

输入 exit 或 quit 退出。
"""

import json
import sys

from agent import Agent, AgentError, RepairHistory, TraceEvent
from config import ConfigError, MemoryConfig, load_memory_config, load_search_embedding
from llm import LLMClient, LLMError
from memory import (
    LocalMemoryStore,
    MemoryContext,
    MemoryLimits,
    default_memory_path,
)
from tools import (
    BuildTool,
    CalculatorTool,
    CallGraphTool,
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
            CallGraphTool(),
            CodeSummaryTool(),
            RunCommandTool(),
            TestTool(),
            EditFileTool(),
        ]
    )


def build_memory_store(config: MemoryConfig) -> LocalMemoryStore | None:
    """按配置装 Memory 的 store；关了返回 None。

    上限（最多留几条 / 文件多大）装进 store（11.3）：写超了 store 自己
    去重 + 丢最旧的，Agent 和 MemoryContext 都不用管。
    """
    if not config.enabled:
        return None
    limits = MemoryLimits(
        max_records=config.max_records, max_bytes=config.max_bytes
    )
    return LocalMemoryStore(config.path, limits=limits)


def build_memory(config: MemoryConfig) -> MemoryContext | None:
    """按配置装一份 MemoryContext（CLI 用）；关了返回 None。

    Agent 那边一行 Memory 代码都不走——没装和「装了但没有匹配」走的是
    同一条路。
    """
    store = build_memory_store(config)
    return None if store is None else MemoryContext(store=store)


class AgentFactory:
    """装配一次、各会话共用：造 Agent 时只换「会话自己的那几件」（13.3）。

    共享的（贵、且没有会话状态）：工具注册表——懒建的索引（调用图、
    代码块、embedding）全挂在工具身上，共享它等于**新建会话不重扫项目**；
    模型客户端；Memory store（历史是项目级的，不属于某一个会话）。

    每只 Agent 自己的：messages / trace / 修复循环 / 上下文裁剪 /
    RepairHistory——new_agent() 造一只新的就全有了。Agent 本身不认识
    「会话」这回事，隔离由 api/sessions.py 那层负责。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        client: LLMClient,
        memory_store: LocalMemoryStore | None = None,
    ) -> None:
        self.registry = registry
        self.client = client
        self.memory_store = memory_store

    def new_agent(self) -> Agent:
        """一只全新会话的 Agent：共享件原样接上，会话状态都是新的。

        RepairHistory 也每只一份：它身上的 last_error / paths 是「这只
        Agent 这一路写了什么」，跟着会话走；文件照旧写进 repair_history/。
        """
        memory = (
            None
            if self.memory_store is None
            else MemoryContext(store=self.memory_store)
        )
        return Agent(
            history=RepairHistory(),
            client=self.client,
            tools=self.registry,
            memory=memory,
        )


def build_agent_factory() -> AgentFactory:
    """把共享件装配一次：工具（连着索引缓存）、模型客户端、Memory store。

    HTTP 服务（python -m api）在启动时调一次，之后每个会话拿 new_agent()
    造一只 Agent——不重复扫项目、不重复加载 memory 文件。
    """
    return AgentFactory(
        registry=build_registry(),
        client=LLMClient(),
        memory_store=build_memory_store(load_memory_config()),
    )


def build_agent() -> Agent:
    """装配一只独立 Agent（CLI 用），注册当前项目提供的全部工具。

    顺手装上 RepairHistory：每轮对话跑完，修复过程会写一份 JSON 到
    repair_history/（默认位置，见 agent/history.py），过后能翻。

    Memory 默认开启（11.2）：跑完一轮会往 AGENT_MEMORY_PATH（默认
    ~/.cache/universal-code-agent/memory.jsonl）记一条，下一轮开始前
    按输入检索历史；设 AGENT_MEMORY=off 就整个关掉（见 config.py）。
    """
    return build_agent_factory().new_agent()


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
    # embedding 模式和 Memory 配置再读一次只为打在横幅上（纯环境变量读取）；
    # build_agent 里已经读过，值不合法在那里就已经报错了
    memory_config = load_memory_config()
    if memory_config.enabled:
        memory_text = f"on（{memory_config.path or default_memory_path()}）"
    else:
        memory_text = "off"
    print(
        f"模型：{agent.client.config.model}  工具：{tool_names}"
        f"  search_code_chunks embedding：{load_search_embedding()}"
        f"  Memory：{memory_text}（输入 exit 退出）"
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
        # Memory 降级同理：任务已经做完了，只是「附带的一步」没成
        if agent.memory is not None and agent.memory.last_error:
            print(f"[Memory] {agent.memory.last_error}", file=sys.stderr)

        print(f"\n最终回答：\n{answer}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
