# Universal Code Analysis Agent · 系统学习指南

这份文档是给「想把这个项目从里到外学一遍」的人用的：**只讲知识点和代码在哪，不重复代码本身**。
行号按封版提交 `954ca1a` 校对过，之后改动过的话以文件里的实际内容为准（名字都是准的，按名字搜）。

---

## 0. 这份文档怎么用

### 0.1 三份文档 + 这份指南，各管什么

| 文档 | 回答的问题 | 什么时候看 |
| --- | --- | --- |
| `README.md` | 现在有什么能力、怎么装怎么跑、限制是什么 | 先看一遍，当手册 |
| **本指南** `docs/STUDY_GUIDE.md` | 每个知识点在哪个文件、为什么这么设计、怎么自己验证 | 学习主线 |
| `docs/NOTES.md` | 逐条设计取舍与原 README 的「说明」存档（后端，374 条） | 学到某个模块时当参考书 |
| `docs/PHASES.md` | 开发阶段演进史 + 每阶段的实测数字（9.5 及以前） | 想知道「这东西是怎么长出来的」时看 |

### 0.2 建议的学习路线（四步）

1. **先跑起来**（第 1 章）：跑通测试、起服务、用工作台发一个任务；看到活的东西再读代码，动力完全不同。
2. **抓住主干**（第 2、3 章）：一次任务的完整数据流 + Agent 主循环 + 工具层。**这是全项目的心脏**，其余都是围着它长的。
3. **按模块深入**（第 4~8 章）：检索、调用图、Memory、Sandbox、HTTP、前端。挑你当前最感兴趣的，不必按顺序。
4. **收口**（第 9~12 章）：测试体系、横切设计原则、自检清单。学完这里，你应该能回答第 11 章的每一个问题。

### 0.3 环境与命令速查

```bash
# 装依赖（Python 3.10+）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # Windows
# .venv/bin/python -m pip install -r requirements-dev.txt         # macOS / Linux

# 跑全量测试（约 2 分钟，2474 条）
run_tests.bat            # Windows 双击或命令行
python run_tests.py      # 跨平台，自动切到 .venv

# 静态检查
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy

# 命令行对话（最简形态，先玩这个）
python main.py

# HTTP 服务 + 前端工作台
python -m api                                  # 后端 127.0.0.1:8000
cd frontend && npm install && npm run dev      # 前端 http://localhost:5173
```

> 前端要在 `.env` 里配 `API_CORS_ORIGINS=http://localhost:5173` 才能连（浏览器发 POST 必带 Origin，同源也挡）。
> 国内网络：pip 用 `-i https://pypi.tuna.tsinghua.edu.cn/simple`，HuggingFace 系模型设 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 1. 第一个小时：让它跑起来

### 1.1 命令行对话（`main.py`）

`main.py:174` 是入口：读配置 → 装配工具和 Agent → 循环读输入。

值得注意的装配代码：`main.py:40` `build_registry()` 注册 **12 个工具**（`:48-63`）；`main.py:90` `AgentFactory` 把「贵且无会话状态」的东西（工具注册表、LLM 客户端、Memory store）共享，每会话只新造对话历史 / Trace / 修复循环（`:112-128`）——这个「什么共享、什么每会话一份」的划分是理解整个服务端装配的钥匙。

**动手**：`python main.py`，问一句「这个项目里有几个工具？分别在哪？」看它怎么先 `list_files`/`search_code` 再回答。`exit` 退出。

### 1.2 跑测试

`run_tests.py:41` 是入口；它做三件事：找到 `.venv` 里的解释器（找不到就报错，**绝不拿系统 Python 凑合**，`:43-51`）、把控制台编码设成 UTF-8（中文输出在 GBK 控制台会炸，`:58`）、原样透传退出码（`:65-69`）。

**动手**：跑一次 `run_tests.bat`，看到 `Ran 2474 tests ... OK (skipped=13)`。13 个 skip 的构成见 README「一键测试」一节。

### 1.3 起服务 + 工作台

- 后端入口 `api/__main__.py:42`：先读配置（配置错就起不来，`:58-66`），再造 Agent 工厂与会话管理器。
- 前端入口 `frontend/src/main.tsx`，根组件 `frontend/src/App.tsx:30`。

**动手**：按 0.3 的命令起两端，在页面里发一个任务，观察 URL 里 SSE 的实时过程。按 F12 打开 Network，找 `/api/run/stream` 那条请求，**看原始字节**——你会看到 `event: xxx\ndata: {...}\n\n` 一帧一帧流出来。这一幕是后面第 7、8 章的直接研究对象。

---

## 2. 全局地图

### 2.1 包一览（一句话职责）

| 目录 | 职责 | 入口/核心文件 |
| --- | --- | --- |
| `agent/` | Agent 主体：主循环、上下文裁剪、提示词、Trace、取消、验证、修复、历史 | `agent/agent.py` |
| `llm/` | LLM 客户端封装：重试、退避、用量统计 | `llm/client.py` |
| `tools/` | 12 个工具 + 注册表 + 统一执行层 + 路径/文本等公共辅助 | `tools/executor.py` |
| `index/` | Code RAG：四层索引 + 字面/语义检索 + 评估 | `index/retrieval.py` |
| `callgraph/` | 调用图：提调用点、建图、查 callers/callees/path | `callgraph/index.py` |
| `memory/` | 任务历史记忆：JSONL 存储 + 检索 + 注入 + 清理 | `memory/context.py` |
| `sandbox/` | 命令执行后端抽象：local / docker | `sandbox/base.py` |
| `api/` | HTTP 服务：JSON + SSE，会话、取消、超时 | `api/server.py` |
| `frontend/` | React 工作台：会话 + 实时过程 + 代码位置可视化 | `frontend/src/App.tsx` |
| `tests/` | 60 个测试文件、约 2474 条用例 | `tests/test_api.py` 等 |

### 2.2 一次任务的完整数据流（后端）

```
用户输入
  → Agent.run()                      agent/agent.py:143
     ├─ Memory 检索注入              agent/agent.py:209 → memory/context.py:109
     └─ _run_loop()                  agent/agent.py:263
        ├─ call_llm()                llm/client.py:370（重试/统计/观察者）
        ├─ 模型要调工具？
        │   └─ _run_tool_call()      agent/agent.py:365
        │      ├─ ToolExecutor.execute   tools/executor.py:66（异常/耗时/Trace 都在这层兜）
        │      ├─ Verifier.after_tool_call   agent/verify.py:180（edit_file 后自动验证）
        │      └─ RepairLoop.after_tool_call agent/repair.py:208（记一轮修复）
        └─ 模型没要工具 → RepairLoop.review() 决定收尾/催促
  → 最终回答（修过则前置 Repair Summary）  agent/agent.py:349
```

**记住这条线，后面每一章都是它的一个片段放大。**

### 2.3 不可信输入的四个入口（安全视角的地图）

学习时要一直带着一个问题：**哪些数据是外来的、不可信的？**

| 入口 | 威胁 | 防线在哪 |
| --- | --- | --- |
| 模型返回的 tool_calls | 形状/类型任意（甚至能是 dict） | `agent/agent.py:379-421` 两层校验 |
| 工具参数里的路径 | `..`、绝对路径、符号链接逃逸 | `tools/paths.py:123` `resolve_in_root`（先 resolve 再判断） |
| 命令类工具的路径参数 | 拼进 shell 的注入 | `tools/paths.py:151` `resolve_shell_path`（字符白名单） |
| 模型写的数学表达式 | 代码执行 | `tools/calculator.py:32` AST 白名单求值 |
| 编译器 flags | 改变命令结构（`-o`、`-MF`） | `tools/compile.py:45` 白名单 + `:246` 输出参数检查 |
| Markdown 回答 | XSS | `frontend/src/components/Markdown.tsx`（结构即安全，见 8.5） |

---

## 3. 主干：Agent 循环与工具层

### 3.1 Agent 主循环 — `agent/agent.py`

**要理解的概念**：tool-calling 循环 = 把「用户问题 + 工具说明书」发给模型，模型可能回「我要调工具」，你执行完把结果喂回去，如此往复直到模型给出最终回答。

**代码在哪**

| 东西 | 位置 | 说明 |
| --- | --- | --- |
| `run()` | `agent/agent.py:143` | 一轮任务：准备 → 循环 → 收尾 |
| `_run_loop()` | `agent/agent.py:263` | 循环本体，四段结构 |
| `_run_tool_call()` | `agent/agent.py:365` | 单次工具调用全流程 |
| `_request()` | `agent/agent.py:248` | 拼发请求用的消息列表（Memory 提示插在 system 后） |
| `_finish()` | `agent/agent.py:349` | 收尾：拼 Repair Summary |
| `DEFAULT_MAX_STEPS = 15` | `agent/agent.py:45` | 上限，注释写明是实测 4~6 轮才够后定的 |

**关键设计（每条都值得停下来想一分钟）**

1. **回滚不变量**（`agent/agent.py:211`、`:216`、`:228`）：`run()` 记下进入时的消息长度，任何异常都把新增的消息删回去。原因写在 `:219-229`：assistant 要了 tool_calls 却没有配对的 tool 回复，**上游 API 会拒掉之后的每一次请求**——这是「一次失败污染整个会话」的经典坑。
2. **`except BaseException` 兜底**（`:219`）：连 KeyboardInterrupt 和畸形响应导致的 TypeError 都回滚。
3. **每轮换新 Trace 对象**（`:116-137`）：不是清空旧对象，而是换一个新的——外部持着上一轮列表不会被偷改。
4. **检查点（取消）**：见 3.7。
5. **不可信边界两层**（`:379-421`）：形状不对（拿不到 call_id）只能抛错回滚；参数值不对（JSON 坏）则**回灌错误文本让模型自己改**。还特判了「网关直传 dict arguments」——`json.loads(dict)` 的 TypeError 是穿透异常，回滚都轮不到。
6. **步数耗尽的两种收场**（`:333-342`）：修复还欠着就 halt + 报告（不抛错，否则连改好的文件一起回滚）；否则抛 `AgentError`。

**自检**：`tests/test_agent.py:340-418` 一整组回归测试就是为第 5 条立的碑（docstring 写了当时的历史被污染成什么样）。

### 3.2 工具层三件套

**三个文件、三个职责，别混：**

| 文件 | 职责 | 关键位置 |
| --- | --- | --- |
| `tools/base.py` | 单个工具的定义方式：名字/描述/参数声明 + 参数校验 + schema 生成 | `Tool` :33、`run` :46（校验→execute）、`to_schema` :81 |
| `tools/registry.py` | 按名注册/查询 | `ToolRegistry` :8、`get` :24（未注册会列出可用名） |
| `tools/executor.py` | **统一执行层**：异常、耗时、Trace、位置提取都收在这一层 | `execute` :66、`_run` :73、`_record` :108、`_locations` :129 |

**执行层最重要的一条不变量：永不向外抛异常**（`tools/executor.py:73-97` 三层兜底：`ToolError` / 任意 Exception / 返回值不是 str）。所有失败都变成 `ToolResult(success=False)`。上面 `agent/agent.py:365` 那套两层校验之所以能工作，就是因为这层保证了「执行工具本身不会炸」。

**参数校验的坑**（`tools/base.py:20`）：`_type_ok` 显式挡 bool 被当成 int（bool 是 int 的子类，Python 里 `isinstance(True, int)` 为真）。

**装配处**：`main.py:40` `build_registry()`——这是「全项目有哪些工具」的**唯一清单**。

### 3.3 12 个工具速览

| 工具（注册名） | 类 | 文件 | 一句话 |
| --- | --- | --- | --- |
| `list_files` | `ListFilesTool` | `tools/list_files.py:13` | 递归列文件，跳过噪音目录，上限 500 |
| `read_file` | `ReadFileTool` | `tools/read_file.py:18` | 读文本带行号（`42|内容`），按整行截断 |
| `search_code` | `SearchCodeTool` | `tools/search_code.py:15` | 按子串搜行，返回 `路径:行号: 内容` |
| `code_summary` | `CodeSummaryTool` | `tools/code_summary.py:21` | 提取目录结构（.py 走 ast，C/C++ 逐行） |
| `run_command` | `RunCommandTool` | `tools/run_command.py:156` | 在 root 内跑命令（超时坑最多，见下） |
| `run_tests` | `TestTool` | `tools/run_tests.py:27` | 命令钉死（模型改不了），转交 run_command |
| `compile` | `CompileTool` | `tools/compile.py:78` | 编译单个 C++ 文件，MSVC 自动铺环境 |
| `build` | `BuildTool` | `tools/build.py:47` | CMake/Make/MSBuild 多步构建 |
| `edit_file` | `EditFileTool` | `tools/edit_file.py:87` | 改文件一处，两种改法互斥，原子落盘 |
| `calculator` | `CalculatorTool` | `tools/calculator.py:76` | AST 白名单求值（不用 eval） |
| `search_code_chunks` | `SearchChunksTool` | `tools/search_chunks.py:48` | RAG 检索代码块，**只给位置不给正文** |
| `call_graph` | `CallGraphTool` | `tools/callgraph.py:41` | callers/callees/path 三种查法 |

**几个值得精读的坑（都在注释里）**：

- `tools/run_command.py:55-99`：`subprocess.run(timeout=)` 在 Windows + `shell=True` 下**超时是假的**（kill 了 shell，子进程还活着，communicate 继续等，1 秒超时实测等了 8 秒）。所以用 Popen + communicate + kill + wait，不碰管道。
- `tools/run_command.py:102`：输出截断掐中间保头尾——「3 failed, 374 passed」在末尾，掐尾巴等于把结论砍掉。
- `tools/textfile.py:68` `split_lines`：**全项目行号唯一口径**。用 `split("\n")` 而不是 `splitlines()`（`\x0c` 会被 splitlines 当换行，行号和编译器对不上）。read_file / search_code / diagnostics / edit_file 全走它。
- `tools/search_code.py:74-77`：路径统一相对 root 报，否则限定子目录搜出来的路径喂给 read_file 会对不上。
- `tools/edit_file.py:7-21`：三道防线——原文唯一（0 次/多次都拒绝）、编码保留（写回原编码）、换行保留（把模型抄来的 LF 还原成文件里的 CRLF）。
- `tools/compile.py:310-311`：g++ 的 flag 翻不成 cl 的就**拒绝**——cl 对不认识的参数只 warning 然后按错的编。

**辅助模块**（不属于工具，但工具都在用）：`tools/paths.py`（路径牢笼 + 敏感名单）、`tools/textfile.py`（编码回退）、`tools/tree.py`（遍历过滤）、`tools/diagnostics.py`（编译器输出 → 结构化诊断）、`tools/pyast.py` + `tools/cpp.py`（结构提取）、`tools/locations.py`（正文 → 位置，见 7.4）、`tools/vsenv.py`（VS 环境）。

### 3.4 上下文裁剪 — `agent/context.py`

**要理解的概念**：对话历史无限增长，但模型有上下文窗口。不是简单截断——**截断了 tool 回复而留下要它的 assistant 消息，请求会被拒**。

**代码在哪**：`ContextManager` :110、`build()` :126。两条规则：

1. **先截断**（`:134`）：只对 tool 消息，超过 5 万字符砍掉并附「不是完整结果」。上限比 read_file 自己的 4 万宽（`:21-23` 注释）——否则 read_file 写的截断说明会被再砍一次。
2. **再整组淘汰**（`:150`）：`_group`（`:48`）把「assistant 带 tool_calls + 其后连续 tool 回复」绑成不可拆的组；`_pick_victim`（`:75`）保护 system、最后一条 user、最后一组，优先删最老的带工具结果的组。

**不变量**：不修改传入的消息（全部返回新字典）；装不下时宁可超一点（`:156-159`），「不能把对话结构删坏」。

### 3.5 系统提示词 — `agent/prompt.py`

`build_system_prompt()` :30，一个 f-string 四段：环境（root/OS/shell）→ 干活顺序 → 回答格式 → 容易踩的点。

**设计要点**：只讲**工具说明里没有的**东西（工具的参数说明在各自的 `description`/`parameters` 里，不重复）；独立成模块是为了「改策略不用碰循环」+ 可单测。

### 3.6 Trace：执行记录与观察者 — `agent/trace.py`

**要理解的概念**：Trace 是「这一轮发生了什么」的结构化记录，也是 CLI 输出和 SSE 流**共同的数据源**。

| 东西 | 位置 |
| --- | --- |
| `TraceEvent`（工具调用一条） | `agent/trace.py:37` |
| `Trace.record_tool_call` | `agent/trace.py:112` |
| `Trace.record_llm_call`（统计另存 `llm_calls`） | `agent/trace.py:147` |
| 观察者通知 `_notify` | `agent/trace.py:156` |

**三条不变量**：

1. **通知不进 events**（`:79-84`）：events 里永远只有「跑完的工具调用」，观察者是旁路。
2. **观察者抛错被吞**（`:156-165`）：客户端断连不能把任务带崩。
3. **Trace 层不认识工具、不碰正文**（`:117-125`）：`locations` 是调用方提好传进来的。

**谁写它**：工具事件由 `tools/executor.py:108` 写；LLM 事件由 `llm/client.py` 的回调写（`:262-296`）。
**谁读它**：CLI 走 `main.py:157` `print_trace`；HTTP 走 `api/sse.py` 的 `TraceStreamer`。

### 3.7 取消与整轮超时 — `agent/cancel.py` + 检查点

**要理解的概念**：协作式取消 = 不杀线程、不硬杀子进程，只在**检查点**问一句「被取消了吗」，到了就主动停。

| 东西 | 位置 |
| --- | --- |
| `CancellationContext`（一次 Run 一个） | `agent/cancel.py:50` |
| `cancel()`（幂等，先到原因算数） | `agent/cancel.py:74` |
| `raise_if_cancelled()` | `agent/cancel.py:100` |
| 超时：构造时算 deadline，检查时现算 | `agent/cancel.py:70`、`:106` |
| Agent 侧检查点入口 | `agent/agent.py:57` `_check_cancelled` |
| 散布的检查点 | `agent/agent.py:205, 282, 289, 295, 306, 311, 326, 424, 429, 438` |
| LLM 退避切成 0.1 秒小片查取消 | `llm/client.py:332-347` |

**关键设计**：

- `RunCancelled` **刻意不继承 `AgentError`**（`agent/cancel.py:36-42`）：取消是控制流不是错误，但走 `Agent.run` 的回滚路径；HTTP 层提前认出来映射成 cancelled/timed_out 而不是 5xx。
- 取消请求**不取会话锁**（`api/runs.py:136-154`）——取消永远不该被正在跑的那一轮卡住。
- 「正在跑的工具/模型调用不打断」（单工具自己的 timeout 管），一返回就停。

**自检**：`tests/test_cancellation.py`（纯 Agent 层）+ `tests/test_api_runs.py`（HTTP 映射）。

### 3.8 改完自动验证 + 修复循环 — `agent/verify.py` + `agent/repair.py` + `agent/history.py`

**要理解的概念**：模型说「改好了」不算数——`edit_file` 之后自动跑编译/构建/测试，把新老诊断摆一起比；没过就一轮轮修，修不动按规矩停。这是本项目最有「工程味」的部分。

**Verifier（验证）** `agent/verify.py`：

- 挑什么验证：`_choose` :293（上次那条命令还管用就重编；Python→run_tests；C/C++ 先找构建系统，没有才单文件 compile）。
- **两关制**：`Verification.ok` :139 = 编译过了 **且**（没跑测试 或 测试也过）。编译过了会自动再跑一次 run_tests。
- **不比行列号**（`_identity` :402）：按 file+level+code+message 比，行号挪位是常事，按行列号比会误报。
- **读不到退出码不敢说通过**（`_outcome` :473）——防「假通过」；`_has_tests` :569 先判「有没有测试」——防「假失败」。
- 验证命令全走 registry + executor（`:238`、`:270`），所以**验证在 Trace 里也看得到**。

**RepairLoop（修复）** `agent/repair.py`：六个停止条件（模块 docstring `:5-15` + 实现）：

1. 验证通过（`:311`）；2. 用满 5 轮（`:341`）；3. 文件一字节没变（digest，`:348`）；4. 同一次改动重复做（fingerprint，`:352`）；5. 连着两轮看到的一模一样（`:358`）；6. 模型连着两次只回话不动手（`:422`）。

核心状态是 `pending`（「有一次改完还没验过」的欠债模型）。`review()` :406 决定主循环下一步：CONTINUE / NUDGE（把模型按回去）/ REPORT / FINISH。

**RepairHistory（落盘）** `agent/history.py:49`：写 JSON 到 `repair_history/`，**绝不往外抛**（`:67`）；文件名时间戳+uuid 绝不覆盖；临时文件 + fsync + `os.replace` 原子落盘（`:90-118`）。

### 3.9 LLM 客户端 — `llm/client.py`

**重试只有这一处**，而且构造 SDK 时 **`max_retries=0` 关掉了 SDK 内建重试**（`:188-193`）。注释 `:179-187` 是一个重要的踩坑记录：不关的话两层重试会叠加（默认 2 会变成最坏 9 次），而且 SDK 的内部重试让 attempt 计数、llm_start/llm_end、取消检查点全都看不见。

其他要点：只重试明确临时的故障（429/5xx/连接，`:50-53`）；`call_llm` :370 是鸭子类型入口（测试替身不用改签名）。

### 3.10 安全边界三块

1. **路径牢笼** `tools/paths.py`：`resolve_in_root` :123——**先 resolve 再判断**（顺序反了等于没查，`..`、绝对路径、符号链接三种逃逸一起堵）；`check_root` :102 挡危险 root；敏感文件名单 `SENSITIVE_FILES` :59（`.env` 挡掉、`.env.example` 放行）。
2. **shell 路径白名单** `tools/paths.py:151` + `_SAFE_PATH` :148：命令类工具的路径只允许 `[A-Za-z0-9_./-]`，代价是带空格/中文的路径用不了——**有意的保守**。
3. **AST 求值** `tools/calculator.py:32`：变量/函数调用/属性访问全拒绝，`9**9**9` 拒绝（`:48`），常数只允许 int/float（挡 bool，`:38`）。

---

## 4. 数据与检索：`index/`（Code RAG）

### 4.1 四层索引怎么衔接

```
ProjectIndex（文件清单）→ SymbolIndex（类/函数/方法+行号）→ ChunkIndex（按符号边界切真代码）→ Retriever（打分排序）
   index/project.py:72      index/symbols.py:208            index/chunks.py:79          index/retrieval.py:207
```

**层与层的关系**：每层「持有下层 + 全量重建」。传下层的对象 = 调用方管；传 `root=` = 自己管；默认 `build()` 不碰别人的东西，`refresh()`（`force=True`）才级联。

**交接面**是 `SymbolIndex.read_verified` `index/symbols.py:348`：读回来先核对 digest，对不上整个文件跳过（记 `CHANGED`）。符号提取、切块、建调用图三处**共用这一份「文件还能不能用」的判断**——不能各说各话。

**digest 是失效的唯一钥匙**：`index/project.py:242` sha256 前 16 位（mtime 是秒级精度，不可靠）。

**每个文件都值得看的两个模式**：扫完才整体换上新的（中途失败旧索引仍完整，如 `index/project.py:114-117`）；一个坏文件不拖垮整次扫描（`_examine` :163 绝不抛）。

### 4.2 字面检索与打分 — `index/retrieval.py`

- 四种 scorer：符号名 / 路径 / 正文 / （8.7 加的）粒度奖励，`Scorer` 接口 :131，`DEFAULT_SCORERS` :201。
- **整名命中单独第一档**（`_order` :270）：排序键第一个分量是「有没有整名命中」——分数比的是同类证据多少，整名是证据种类不同。有测试拿硬塞高分钉着这条。
- 排序完全确定（同档内：分数→路径→起始行→符号名），同一句查询跑一百遍顺序一样。
- **不产生假命中**：2 字母英文词不参与路径/正文模糊扫描（`_long_enough` :309），中文 2 字即可。

### 4.3 embedding 与降级链 — `index/embedding.py` 等

**四种模式**（`config.py:66`，环境变量 `SEARCH_CHUNKS_EMBEDDING`）：`off`（默认）/ `fake`（离线假货）/ `local` / `multilingual`。

**语义这只是 scorer 列表里多出来的一条**（`hybrid_scorers` :629）——这就是「可插拔」的落点：检索格式不变，只加一个打分器。

**降级链**（自上而下，每一环都在文件里有注释）：
1. 名字不认识的模式 → 启动即报（`config.py:81-85`）；
2. `DegradingScorer` :592：第一次出错**永久关自己**、`error` 记第一现场、**刻意不重试**；
3. `LocalEmbedder` 懒加载（`index/local_embedding.py:100`）：构造不碰模型，没装依赖也能起服务，报错推迟到真要用；
4. `VectorCache` 读写失败只降级不炸（`index/vector_cache.py:261`）。

**注意**：建索引失败**不降级**（`tools/search_chunks.py:243-245`）——那是 baseline 也会遇到的整体失败，不假装降级成功。

### 4.4 向量缓存 — `index/vector_cache.py`

- **key 怎么算**：chunk = `id|digest`（:106，文件一改 digest 变、旧向量自然查不到）；查询 = `query:` + sha256（:111，**故意不存查询原文**——等于「用户问过什么」）；命名空间 = 模型身份 + 文本管线版本 + 窗口（`cache_for` :662）。
- **防坏四层**：JSON 坏当空（:261）；version/namespace 对不上不读不写**绝不覆盖别人的**（:267）；单条坏只丢单条（:280）；写盘临时文件+换名（:307）。
- **打扫的分工**：`prune` :181 清「id 还在但版本过时」的；`forget_ids` :214 清「id 自己没了」的；查询向量永不动。

### 4.5 评估体系 — `index/evaluation.py` + `index/evaluation_cases.json`

**43 条 case（37 正 + 6 反）**，`python -m index` 最多四组同台比（字面 / fake / local / multilingual）。

- 指标：`rank` 是唯一存下来的事实，Hit@1/3/5 从它现推（:159）——明细和汇总不会对不上；Hit@k 只统计正例（:212），反例单独统计。
- **case 放 JSON 不放 py**（:19-21）：项目把自己也索引，查询字符串写成 Python 常量会进 chunk 索引——「等于把答案提前埋进考卷」。
- 效果回归 `tests/test_evaluation_regression.py`：功能测试问「对不对」，它问「有没有变差」——把 baseline 数字（Hit@1 10/37 等）冻结成下限。

### 4.6 调用图 — `callgraph/`

```
base.py（数据结构+查询）  ←  python.py / cpp.py（提调用点）  ←  index.py（认主+组图+失效）
   :124 CallGraph              :25 / :70 extract_sites            :42 CallGraphIndex
```

- **两步分离**（`callgraph/index.py:1-11`）：提调用点是**语法**问题（各语言一份提取器），认主是**语义**问题（拿符号索引行区间判，各语言共用）。这个切分值得当设计手法学。
- Python 提取器最妙的坑（`callgraph/python.py:58-72`）：**装饰器和默认参数里的调用不记**——它们在**外层作用域**执行，却位于被定义函数的行范围里，按行认主会算错主人。「宁可漏一条，不记错的」。
- C/C++（`callgraph/cpp.py:127`）：只扫函数体（签名后第一个深度 0 的 `{` 之后）；`int x(3)` 这种括号初始化用启发式挡掉。
- **缓存失效**（`callgraph/index.py:143-151`）：`built` 不光看自己建没建，还比对 `symbols.generation`——谁重扫了符号索引，旧图自动作废。
- 查询侧：BFS 最短链（`callgraph/base.py:201`）；`_matches` :101 的不对称规则（带点必须精确，简单名命中所有同名）；`self.helper` 断链**是有意的**（`:215-218`，「宁可说没找到，也不猜 self 是谁」）。

---

## 5. Memory — `memory/`

**要理解的概念**：任务历史记忆。开始前按输入检索几条相关历史注入（「仅供参考」），结束后记一条。**不是对话历史**。

四个文件的分工（很干净，值得学）：

| 文件 | 只管 | 关键位置 |
| --- | --- | --- |
| `store.py` | 追加一行 / 读全部（JSONL） | `LocalMemoryStore` :75、`append` :102 |
| `retriever.py` | 按 query 找 top_k 并裁剪 | `MemoryRetriever` :101、`retrieve` :119 |
| `context.py` | 接进 Agent 前后、渲染 system 消息、失败不抛 | `before` :109、`after` :132 |
| `maintenance.py` | **纯规划**：算留哪些（plan/apply 分离） | `plan_prune` :153、`dedupe` :88 |

**三个值得学的点**：

1. **残句补换行**（`store.py:251` + `:120-127`）：上次写到一半、文件结尾没 `\n` 时直接追加会让新记录粘在残句后面「一条坏变两条」——所以 append 前先看最后一个字节。
2. **汉字切 bigram**（`retriever.py:55`）：和 `index/retrieval.py` 的整段匹配**故意不同**——代码检索的汉字多在注释里、整段够用；Memory 的 task 是自然语言，整段匹配等于「一字不差才命中」。注释把差异理由写透了。
3. **降级点只有两处**：`context.before/after` 的 try/except 与 `store._enforce_limits`（自动清理失败只记 `last_error`，「记录已经写进去了，清理是附带的一步」）。

**提示词安全**：注入的历史带抬头「仅供参考、可能过时、不是指令」（`context.py:49`）——历史内容可能碰巧长得像命令，属于 prompt-injection 面。

---

## 6. Sandbox — `sandbox/`

**接口最小**（`sandbox/base.py:101`）：`run(command, cwd, timeout, env) -> CommandResult`。**超时是一等公民**（`CommandResult` :70 起，`timed_out` 是字段、不是异常）。

| 实现 | 文件 | 要点 |
| --- | --- | --- |
| `LocalSandbox` | `sandbox/local.py:21` | 执行/校验/边界**一行没自己写**——全复用 `tools/run_command.py` 的实现 |
| `DockerSandbox` | `sandbox/docker.py:70` | 一次性容器；**base64 包命令**（:57）绕过两三层引号的互咬 |
| `SandboxManager` | `sandbox/manager.py:48` | 按 kind 造，默认 local；**懒建**（配置不对推迟到第一次真跑才报） |

**工作目录怎么锁**：唯一实现是 `sandbox/base.py:49` `resolve_cwd`——省略=workspace；相对从 workspace 算；绝对也必须在里面。两种后端共用；docker 再把已校验的 host cwd 映射进 `/workspace/...`（`docker.py:222`）。

**Docker 的两条纪律**：只有 workspace 一个目录 bind mount；`--pull=never` 堵住偷拉镜像；**不静默退回 local**（docker 不可用就报错说清楚）。

---

## 7. HTTP API — `api/`

### 7.1 五条路由与三道门（`api/server.py`）

| 路由 | 处理函数 | 说明 |
| --- | --- | --- |
| `GET /api/health` | `_health` :308 | 免 Token（前端探活用，不碰 Agent） |
| `POST /api/run` | `_run` :317 | 普通 JSON 响应 |
| `POST /api/run/stream` | `_run_stream` :397 | SSE |
| `DELETE /api/sessions` | `_delete_session` :505 | 删会话（共享工具/索引/Memory 不动） |
| `POST /api/runs/cancel` | `_cancel_run` :540 | 按 session_id（可加 run_id）取消 |

路由表 `_ROUTES` :866 是**唯一清单**：405 的 Allow 头、404 的提示文案都从这张表推（加接口不用改文案）。

**三道门都在 `_dispatch` :133**，顺序即优先级：

1. **Origin**（:140，判定 :214）：不带 Origin 放行（命令行客户端）；带了必须在允许列表里——**同源也挡**，挡的是「本机随便一个网页对本服务发 POST」（本服务的 run_command/edit_file 是真会动系统的）。
2. **Token**（:170）：`hmac.compare_digest` 常量时间比较（:253）。
3. **Content-Type**（:184）：POST/DELETE 必须 JSON，否则 415。

顺带一个细节：路由没命中时**先把 body 读干净**再回 405/404（:147-167）——不然长连接里的残字节会被当成下一个请求。

### 7.2 会话表与锁 — `api/sessions.py`

**两把锁各管各的**（本文件最重要的设计）：表层一把 `_lock` 管创建/查找/删除；每个 `Session` 自带一把 `lock`（:60）管「同一会话别同时跑两个任务」。不同会话不互堵，所以**服务级没有全局锁**。

- **握着锁的不清**（`_purge_locked` :165）：把正在跑的会话摘掉，它照跑完但客户端再也拿不到——注释写明「这里以前漏了」。
- **`touch()` 单独存在**（:141）：一轮 run 可能比 TTL 还长，不补刷的话任务跑完自己就「已过期」了。
- 不认识的 session_id → 404，**绝不顺手新建**。

### 7.3 Run 登记与取消 — `api/runs.py`

状态机：created → running → completed/failed/cancelled/timed_out（:34-42）。

- **排队时间也算进整轮预算**（:83-97）：create 时就把 `CancellationContext(timeout=...)` 建好。
- **cancel 不取会话锁**（:136）：纯粹的置位，不等任何人。
- **不永久占内存**：结束的 Run 进 `_finished` + deque，超 64 条从最早的丢。

### 7.4 SSE 帧与 Trace 翻译 — `api/sse.py`

- **帧怎么拼**（:34）：`event: 名字\ndata: 一行JSON\n\n`。靠 `json.dumps` 把字符串内换行转义掉，**一帧就是规规矩矩两行**——读的人不用处理 SSE 续行规则。
- **写失败不算事**（:94）：`send` 捕获一切（多半是客户端断开），置 `closed` 跳过后面——异常不许顺栈把这一轮任务变成失败，「它还在替别人干活」。
- **克制原则**：工具正文（ToolResult.content）压根不进 Trace，也进不了帧；参数值/错误摘要截断到 120 字符（`SUMMARY_LIMIT` :31）。
- **`locations`**（14.3 唯一新增）：`tool_result` 帧多带一个位置列表。它**是坐标不是正文**——由 `tools/locations.py` 从工具输出里按**白名单**提取（只认 6 个工具自己固定的输出格式，`_PARSERS` :177），`run_command`/`run_tests` 这类一律空（往任意命令输出里捞「文件:行号」会捞出一堆似是而非的东西）。提取点：`tools/executor.py:129`（成功才提、失败兜空）。

### 7.5 取消/超时怎么走到 HTTP

```
cancel 请求 → RunRegistry.cancel（不等锁）→ CancellationContext 置位
  → Agent 在检查点抛 RunCancelled → api/server.py:361 / :468 捕获
  → _stopped_status :816（reason==TIMEOUT → timed_out，否则 cancelled）
  → 200 + success=false + status（"取消不是错误"）
```

SSE 流里则是发一个同名事件（`cancelled`/`timed_out`）后**正常关流**，响应本身仍是 200。

**两个踩坑记录**（都值得读原文）：`timeout = 60.0` 类属性 :107（socketserver 原生捕获 socket.timeout，防慢速攻击，且发生在 Token 校验之前就能生效）；**幽灵 Run**（:441-447 注释）：开流和第一帧原先在 try 外，客户端「连上又立刻断开」会让 `finish()` 全轮不到，`_active` 里留下永远 running 的幽灵——修法是把它们挪进 try + 每个出口 finally 兜底。

---

## 8. 前端工作台 — `frontend/`

### 8.1 三层依赖（不许跨层）

```
组件层 components/  ←  state 层 state/  ←  api 层 api/（唯一碰网络）
```

组件只依赖 `state/transcript.ts` 的类型与派生函数 + `components/labels.ts`，**不直接碰网络或 storage**。

### 8.2 一次任务的前端数据流（背下来这一条）

```
Composer 提交         components/Composer.tsx:23
  → App 接线          App.tsx:125
  → useRun.start      state/useRun.ts:111（同一时间只跑一轮 :114）
  → client.streamRun  api/client.ts:106（fetch POST + ReadableStream 自己读）
  → 帧解码            api/sse.ts:43（跨块缓冲，按空行切帧）
  → 事件翻译          api/sse.ts:103（toRunEvent，认不出的返回 null 不打断流）
  → 写进记录          useRun.ts:170 → useTranscripts.update → transcript.applyFrame :228
  → 落盘              state/storage.ts:135（每次 update 立刻存）
  → 渲染              App.tsx:117 → RunList → RunCard.tsx:42
```

**为什么不用浏览器自带的 `EventSource`**（`api/sse.ts:1-15`）：它只能发 GET、不能带 `Authorization` 头——后端是 POST + 可选 Bearer Token，两条都做不到。

两个细节：`decode(value, {stream: true})`（`client.ts:134`）解决汉字被切在两次 read 之间；`transcript.applyFrame:239` 用 cursor 挡重复帧。

### 8.3 两套状态（互不干扰）

| 状态 | 谁管 | 存哪 |
| --- | --- | --- |
| 会话：有哪些、当前哪个 | `state/sessions.ts:49` `useSessions` | localStorage 两个键（:16-17） |
| 当前这一轮：busy/cancelling | `state/useRun.ts:62` `LiveRun` | 内存 ref（:77-84） |
| 每会话一份记录 | `state/useTranscripts.ts:46` + `transcript.ts` 纯函数 | localStorage 每会话一个键 |

两套只在 `App.tsx:34-45` 的接线处碰面——`useRun` 自己不认识「会话列表」。

### 8.4 localStorage 的边界（安全相关，最该背）

- **键只有三种**：`uac.frontend.sessions.v1`、`uac.frontend.active.v1`、`uac.frontend.transcript.v1.<session_id>`（一会话一份，坏一份只坏一份）。
- **存的**：任务原文、状态与时间、回答（截 2 万字）、工具/模型摘要、位置信息（单次最多 12 条 + 真实总数）。
- **不存的**（有测试钉着）：API Token（类型里根本没有 token 字段）、工具参数原文（可能带被读文件内容）、原始 SSE JSON、完整工具返回正文、prompt。
- **坏数据三层兜底**：JSON 解析失败 / 版本不符 / 字段类型不对 → 一律当没存过，页面不崩；解析失败**不删**（用户可能还想捞，`storage.ts:87`）。
- **刷新之后不会显示「运行中」**：每次读回都过 `markInterrupted`（`transcript.ts:307`）并**顺手写回**——不显示假的 Running。
- **写不进去**（配额满/隐私模式）：先清别的会话最老的 → 再砍自己最旧的几轮 → 都不行才 failed，页面上说一句「本机存不下记录了」（`storage.ts:135-159`）。

### 8.5 组件清单（按重要性）

| 组件 | 位置 | 一句话 |
| --- | --- | --- |
| `RunCard` | `components/RunCard.tsx:42` | 一轮的卡片：任务→状态→（折叠）过程→回答 |
| `Markdown` | `components/Markdown.tsx` | 回答渲染；**结构安全**：无 dangerouslySetInnerHTML、不装 rehype-raw、图片不自动加载 |
| `LocationList` | `components/LocationList.tsx:30` | 位置卡片：复制路径 / 复制 `path:line`；「在编辑器打开」**明确禁用**（不做假按钮） |
| `ChainView` | `components/ChainView.tsx:28` | 调用链 `A → B → C`，有环画到重复处、超 8 节点截断 |
| `RunList` | `components/RunList.tsx:22` | 记录区 + 「用户本来就在底部才跟随」的自动滚动 |
| `SessionSidebar` | `components/SessionSidebar.tsx:37` | 左栏：会话列表 / 新建 / 删除 / Token 输入 / 后端状态 |
| `PaneHead` | `components/PaneHead.tsx:33` | 右栏头：标题 + 状态徽标 + 实时统计 + 清空本机记录 |
| `labels.ts` | `components/labels.ts:12` | 所有界面文案与格式化函数的**单点维护处** |

### 8.6 前端测试怎么在不起浏览器的情况下测 DOM

- **自造 harness**（`src/test/harness.tsx:15-60`）：用 React 自带 act + createRoot 十几行解决，**故意不引 @testing-library**（少一个依赖树）。
- vitest 默认 node 环境，需要 DOM 的文件头上写 `// @vitest-environment jsdom`（`vite.config.ts:18-21`）。
- **需要契约时起真 `node:http` 假后端**（`api/client.test.ts:92-214`），还故意把 SSE 从第 40 字节切开写，逼前端拼帧。
- `sse.ts` 的回归用**真 fixture**（`api/__fixtures__/stream-real.sse`，真服务输出的原始字节，注释里有重抓命令）——后端帧格式改了，前端测试先红。

推荐阅读顺序：`sse.test.ts`（纯函数）→ `transcript.test.ts`（纯逻辑）→ `storage.test.ts`（真 localStorage + 坏数据）→ `client.test.ts`（真 HTTP 契约）→ `useRun.test.tsx`（集成）。

---

## 9. 测试体系（Python，`tests/`）

### 9.1 分层

| 层 | 代表文件 | 手法 |
| --- | --- | --- |
| 纯单元 | `test_tools` / `test_paths` / `test_context` / `test_trace` / `test_llm_client` | 不碰 IO，钉不变量与边界 |
| 工具端到端 | `test_read_file` / `test_edit_file` / `test_compile` / `test_build` … | 临时目录 + 真文件系统；假编译器（往 PATH 塞 .bat） |
| RAG 线 | `test_index*` / `test_symbols` / `test_chunks` / `test_retrieval` / `test_embedding*` | 临时目录；语义用 FakeEmbedder |
| 评估线 | `test_evaluation*` | 测「尺子」本身 |
| 效果回归 | `test_evaluation_regression` | 只问「有没有变差」，baseline 数字冻结成下限 |
| Agent 行为 | `test_agent` / `test_repair` / `test_verify` / `test_cancellation` | 替身驱动 |
| HTTP | `test_api` / `test_api_stream` / `test_api_security` / `test_api_sessions` | **真 socket 真 HTTP**（`ApiCase` 基类 `test_api.py:53`，`port=0` 让系统挑端口） |

### 9.2 替身设计学（本项目最值得抄的一课）

**共同原则：替身只替换「外部世界」（网络、模型、时钟、进程、Docker），不动被测逻辑；能用真的就用真的。**

| 替身 | 位置 | 手法 |
| --- | --- | --- |
| `FakeLLMClient` | `tests/test_agent.py:13` | 脚本化 replies：字符串=最终回答 / dict=要调工具 / 异常=这次抛错；最后一项会重复（方便测「停不下来」） |
| `ApiCase` | `tests/test_api.py:53` | 真服务真 socket；后台线程 serve；`addCleanup` 后进先出 |
| `FakeClock` | `tests/test_sessions.py` | sleep=advance，退避不用真等 |
| `RecordingSandbox` | `tests/test_sandbox_integration.py` | 先记账再转给真 LocalSandbox——既证真结果又证「确实走了 Sandbox」 |
| 假 g++ / 假 cmake | `tests/test_compile.py` / `test_build.py` | 往 PATH 最前塞脚本，**真的走 `shutil.which` 那道检查** |

### 9.3 断言风格

`tests/test_trace.py` 是范本：每个用例名是一句规格（`test_every_call_adds_exactly_one_event`）；大量钉**不变量**而不是实现细节——frozen dataclass 不可改、可变默认值不在实例间共享、arguments/locations 是快照不是引用；「记 Trace 不改变行为」有整组测试；观察者抛异常不许把成功调用变失败。

---

## 10. 横切设计原则（这个项目的「性格」）

学完各个模块后，回头用这些原则串一遍，你会发现它们反复出现：

1. **附带的事不能把正事带崩**——写历史（`agent/history.py:67`）、提位置（`tools/executor.py:129`）、Trace 通知（`agent/trace.py:156`）、Memory 读写（`memory/context.py:121`）、SSE 写帧（`api/sse.py:94`）五处同一条。
2. **边界只有一份实现**——行号口径（`tools/textfile.py:68`）、路径牢笼（`tools/paths.py:123`）、cwd 校验（`sandbox/base.py:49`）、文件新鲜度（`index/symbols.py:348`）。
3. **全量重建 + generation/digest 管失效**——四层索引、调用图都是这个模式；不做增量是**有意的取舍**（简单、可重跑、不会错）。
4. **假通过和假失败一样坏**——读不到退出码不算通过（`agent/verify.py:473`）；先判「有没有测试」再报失败（:569）；前端刷新后不许显示假的 Running。
5. **不可信输入要显式建模**——见 2.3 的表；每个入口都有测试钉着。
6. **宁可说不知道，也不猜**——行号提取不猜（`tools/locations.py`）、调用链断在 `self`（`callgraph/base.py:215`）、C/C++ 结构宁可漏不误报、Memory 检索不到就没有。
7. **降级而不是崩**——embedding 降级链、缓存坏文件当空、前端 localStorage 坏了当没存过。
8. **导入环有三种拆法**（都用了）：`TYPE_CHECKING`、函数内延迟 import、鸭子类型不提类名。
9. **启动即报错 vs 运行期降级，界限明确**——配置错启动就死（`config.py`）；数据/缓存/可选依赖坏了运行期降级。
10. **不做的功能要说明白为什么不做**——不该弹的横幅不弹、做不到的按钮直接禁用并写明原因（`LocationList.tsx:88`）、代理不配（`vite.config.ts:6-13`）。

---

## 11. 自检清单（能答上来就算学完了）

**主干**
1. 模型要了一个不存在的工具，从 LLM 返回到最终回答，这条错误经过了哪些代码？（答：`agent/agent.py:365` → `tools/executor.py:66` → `_failed` → 回灌 tool 消息）
2. 为什么 `run()` 必须回滚消息？不回滚会发生什么？
3. Trace 的 events 里为什么没有「开始调用」？（答：观察者是旁路，events 只记跑完的）

**工具与安全**
4. 为什么 `run_command` 不用 `subprocess.run(timeout=)`？
5. `resolve_in_root` 为什么必须先 resolve 再判断？少一步会怎样？
6. edit_file 的「原文唯一」指什么？为什么空串 old_text 要直接拒绝？

**检索**
7. 四层索引各自的「输入 → 输出」是什么？`read_verified` 被哪三处共用？
8. 文件改动后，为什么旧向量「自然查不到」而不需要显式清理？
9. 查询向量为什么不存原文？

**HTTP / 前端**
10. 为什么浏览器前端必须配 `API_CORS_ORIGINS`，同源也挡？
11. 取消一个正在跑的任务，从点击 Cancel 到页面显示「已取消」，中间经过哪些环节？
12. 刷新页面后，为什么「运行中」会变成「已中断」？为什么这是对的？

**测试**
13. `FakeLLMClient` 的 replies 列表里，放一个异常实例是什么意思？
14. 效果回归和功能测试的区别是什么？

---

## 12. 附录

### 12.1 命令速查

```bash
python main.py                                  # 命令行对话
python -m api                                   # 起 HTTP 服务
python -m index                                 # 跑 RAG 评估（43 条 case）
python -m index --hybrid                        # 加一组语义对比
.venv/Scripts/python.exe -m unittest tests.test_agent -v    # 单文件测试
run_tests.bat -v                                # 全量 + verbose
cd frontend && npm run typecheck && npm test && npm run build
```

### 12.2 关键数字（封版时）

- Python 测试：2474 条，13 skip（7 Docker + 4 符号链接 + 2 真模型）；60 个测试文件。
- 前端测试：7 个文件、164 条 `it()`；`npm run build` 产物 JS 416 kB（gzip 130 kB）。
- 评估集：43 条 case（37 正 + 6 反）；字面 baseline Hit@1 10/37、Hit@3 12/37、Hit@5 12/37。
- 运行时常量：`DEFAULT_MAX_STEPS = 15`（`agent/agent.py:45`）、`DEFAULT_MAX_REPAIR_ROUNDS = 5`（`agent/repair.py:65`）、`MAX_TOTAL_CHARS = 100_000`（`agent/context.py:18`）、`MAX_LOCATIONS = 50`（`tools/locations.py:28`）。

### 12.3 推荐精读的五个文件（如果只读五个）

1. `agent/agent.py` —— 全项目的心脏，注释密度最高的地方之一。
2. `tools/executor.py` —— 「横切关注点收在一层」的范本，很短。
3. `agent/repair.py` —— 最复杂也最有工程味的部分；先读模块 docstring 那 15 行。
4. `api/server.py` —— 标准库 HTTP 服务的完整范本（三道门、状态码地图、两个踩坑记录）。
5. `frontend/src/state/transcript.ts` —— 纯函数记录规则，前端所有不变量的所在地。

### 12.4 延伸阅读顺序

- 想深入重试/统计：`llm/client.py` → `tests/test_llm_retry.py`（ScriptedAPI + FakeClock）。
- 想深入 SSE：`api/sse.py` → `api/server.py:397`（`_run_stream`）→ `tests/test_api_stream.py` → 前端 `sse.ts` + `sse.test.ts`。
- 想深入索引：`index/project.py` → `symbols.py` → `chunks.py` → `retrieval.py`（按依赖顺序读，每层先读模块 docstring）。
- 想深入诊断解析：`tools/diagnostics.py` → `agent/verify.py` 的 `_identity`/`_outcome` → `tests/test_diagnostics.py`。
- 设计与取舍的原文：`docs/NOTES.md`（后端 374 条）；前端设计说明在 `README.md` 的「启动前端」一节。
