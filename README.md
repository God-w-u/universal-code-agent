# Universal Code Analysis Agent

通用代码分析智能体：能自主分析代码项目、调用工具、执行编译 / 测试，并根据结果继续推理。**不用任何智能体框架**（没有 LangChain / LangGraph），Agent Runtime、检索、调用图、Memory、沙箱全部手写，依赖只有 `openai` + `python-dotenv` 两个。

核心能力一览：

- **Agent 循环**：tool-calling 循环（`agent/agent.py`）+ 发请求前的上下文裁剪（`agent/context.py`）；
- **Code RAG**：四层索引（文件 → 符号 → 代码块 → 字面检索），可选 embedding 语义检索（`index/`）；
- **调用图**：谁调用了谁、函数到函数的调用链（`callgraph/`）；
- **Memory**：任务历史落本地 JSONL，下一轮按输入检索、作为「仅供参考」注入（`memory/`）；
- **Sandbox**：命令执行的统一接口，local / docker 两种后端（`sandbox/`）；
- **改完自动验证 + 修复循环**：C/C++ 先编译 / 构建再跑测试，验不过接着修，修不动按规矩停（`agent/verify.py`、`agent/repair.py`）；
- **HTTP API**：标准库实现的 JSON + SSE 接口，带会话、协作式取消、整轮超时（`api/`）；
- **工程化**：一键测试脚本（`run_tests.bat` / `run_tests.py`）+ 静态检查（ruff / mypy，见「静态检查」一节）+ GitHub Actions CI（`.github/workflows/ci.yml`）。

## 目录

- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [配置（.env）](#配置env)
- [一键测试](#一键测试)
- [静态检查](#静态检查)
- [CI](#ci)
- [启动 API](#启动-api)
- [命令行对话](#命令行对话)
- [能力概览](#能力概览)
- [Docker Sandbox 的前置条件和测试 skip 原因](#docker-sandbox-的前置条件和测试-skip-原因)
- [Embedding：可选依赖与降级](#embedding可选依赖与降级)
- [已知限制](#已知限制)
- [更多文档](#更多文档)

## 项目结构

```
universal-code-agent/
├── agent/           # Agent 主体
│   ├── agent.py     # Agent：会话历史 + tool-calling 循环（LLM / 工具 / Memory 的接线处）
│   ├── context.py   # ContextManager：发给 LLM 前的上下文裁剪（单条截断 + 整轮丢弃）
│   ├── prompt.py    # 系统提示词：环境、干活顺序、引用规范
│   ├── trace.py     # Trace / TraceEvent：一次 run() 的工具调用记录（也是 SSE 的数据源）
│   ├── cancel.py    # 一次 Run 的取消开关：协作式（含整轮超时），不杀线程
│   ├── verify.py    # 改完自动验：编译 / 构建过了再跑测试，和上次诊断比
│   ├── repair.py    # 验不过接着修：记轮次、盯指纹、修不动就停
│   └── history.py   # 每轮 run() 的修复过程写一份 JSON 到 repair_history/
├── index/           # Code RAG（8.1 起）
│   ├── languages.py / project.py / symbols.py / chunks.py   # 四层索引：文件 → 符号 → 代码块
│   ├── retrieval.py # Retriever：字面打分（符号 / 路径 / 正文）+ 排序
│   ├── embedding.py # Embedder 接口 / FakeEmbedder / EmbeddingScorer / DegradingScorer
│   ├── compact.py / segments.py   # 喂给模型的紧凑文本 + 按窗口切段
│   ├── vector_cache.py # 向量缓存（JSON，key 带 digest，坏文件只降级）
│   ├── local_embedding.py / providers.py  # 本地模型封装（可选依赖）+ 工厂
│   ├── evaluation.py / evaluation_cases.json / __main__.py  # 43 条评估 case 与 CLI 入口
│   └── ...          # 各索引都支持显式 refresh()（13.6）
├── callgraph/       # 调用图：python.py / cpp.py 提调用点，index.py 建图，base.py 数据结构
├── memory/          # Memory：base / store（JSONL）/ retriever / context / maintenance
├── sandbox/         # 命令执行后端：base（接口）/ local.py / docker.py / manager.py
├── llm/             # LLM 客户端（openai SDK 封装 + 重试 + 用量统计）
├── tools/           # 12 个工具：list_files / read_file / search_code / code_summary /
│                    #   run_command / run_tests / compile / build / edit_file /
│                    #   calculator / search_code_chunks / call_graph
├── api/             # HTTP 服务（python -m api）：server / sessions / runs / sse
├── tests/           # 全部单元测试（离线，用替身对象）；fixtures/cpp 是 C++ 评估语料
├── docs/
│   ├── PHASES.md    # 开发阶段记录（9.5 ~ 6.7 的原文存档）
│   └── NOTES.md     # 设计说明（逐条：每个行为背后的取舍与实测结论）
├── .github/workflows/ci.yml  # CI：Linux 跑静态检查 + 全量 unittest，Windows 跑一键入口
├── run_tests.py / run_tests.bat  # 一键测试（用项目 .venv，不依赖 PATH）
├── main.py          # 命令行入口（python main.py）
├── config.py        # 配置读取（环境变量 / .env）
├── pyproject.toml   # 只放 ruff / mypy 的配置（不是打包文件）
├── requirements.txt # 运行时依赖：openai + python-dotenv，就这两个
├── requirements-dev.txt  # 上面那份 + 测试专用的 httpx + 静态检查的 ruff / mypy
├── .env.example     # 配置模板（只有空值，没有真 Key）
└── repair_history/  # 运行时生成：每轮对话一份修复记录，平时不存在
```

## 环境安装

要求 **Python 3.10+**（开发与 CI 用 3.12）。

```bash
cd universal-code-agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

依赖只有两个（LLM 客户端 + 读 `.env`），没有 Web 框架、向量库、Agent 框架。

**要跑测试 / 静态检查就装 `requirements-dev.txt`**（= 上面那份 + 测试专用的 `httpx` + 静态检查的 `ruff` / `mypy`，CI 装的就是这一份）：

```bash
pip install -r requirements-dev.txt
```

`httpx` 只被测试用到：`tests/test_llm_retry.py` 拿 `httpx.Request` / `httpx.Response` 构造 OpenAI SDK 的异常对象，`tests/test_api_llm.py` 又从它 import。它是 `openai` 的间接依赖，但新版 `openai` 拉的是 `httpx2`、不再带 `httpx`，不能指望「装了 openai 就有」，所以显式声明。`ruff` / `mypy` 只在开发和 CI 里跑，运行时代码一行都不 import 它们。**运行时都不需要**，`requirements.txt` 里没有。

可选依赖（**不装也能跑全部测试**，测试自带替身对象）：

| 能力 | 需要装什么 | 什么时候才用到 |
| --- | --- | --- |
| 语义检索 | `pip install sentence-transformers` | 只在 `SEARCH_CHUNKS_EMBEDDING=local / multilingual` 且真的调用 `search_code_chunks` 时加载模型 |
| 容器化执行 | 本机有 Docker CLI + 运行中的 daemon + 现成镜像 | 只在显式用 DockerSandbox 时（见「Docker Sandbox」一节） |

## 配置（.env）

把 `.env.example` 复制为 `.env` 填写。**`.env` 含密钥，已被 `.gitignore` 忽略，不要提交**；仓库里只有全是空值的模板。系统环境变量优先于 `.env`。

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 是 | API Key（只进 `.env` / 环境变量，不写进代码） |
| `OPENAI_MODEL` | 是 | 模型名，按服务商填 |
| `OPENAI_BASE_URL` | 否 | OpenAI 兼容服务地址，留空用官方 |
| `OPENAI_TIMEOUT` | 否 | 单次请求超时秒数，默认 60 |
| `LLM_MAX_RETRIES` | 否 | 临时错误（429 / 5xx / 连接不上 / 超时）最多重试几次，默认 2；0 = 一次就抛 |
| `LLM_RETRY_BACKOFF` | 否 | 重试基础退避秒数，默认 0.5，指数涨（0.5、1.0、…）；0 = 不等 |
| `SEARCH_CHUNKS_EMBEDDING` | 否 | `search_code_chunks` 的检索模式：`off`（默认）/ `fake` / `local` / `multilingual` |
| `AGENT_MEMORY` | 否 | Memory 开关：`on`（默认）/ `off` |
| `AGENT_MEMORY_PATH` | 否 | Memory 数据文件位置，默认 `~/.cache/universal-code-agent/memory.jsonl` |
| `AGENT_MEMORY_MAX_RECORDS` | 否 | 最多留几条历史，默认 500，超了丢最旧 |
| `AGENT_MEMORY_MAX_BYTES` | 否 | Memory 文件字节上限，默认 2 MiB，超了丢最旧 |
| `API_SESSION_TTL` | 否 | API 会话多久没人用就过期（秒），默认 3600 |
| `API_MAX_SESSIONS` | 否 | 最多同时留几个会话，默认 32，超了丢最久没用的 |
| `API_CORS_ORIGINS` | 否 | 允许跨源的前端 Origin（逗号分隔），默认空；不接受 `*` |
| `API_TOKEN` | 否 | 设了之后除 `/api/health` 外的接口都要 `Authorization: Bearer <token>` |
| `AGENT_RUN_TIMEOUT` | 否 | 一次 Run 的总预算秒数（LLM + 工具 + 修复共享），默认 0 = 不限时 |

值不合法（拼错模式名、负数、`*` Origin）在**启动时**就报 `ConfigError` 并列出可选项，不会等第一次用到才发现。

## 一键测试

三种入口，做的是同一件事——用项目 `.venv` 里的 Python 跑全量 unittest：

```bash
run_tests.bat                     # Windows：双击或命令行都行，完全不依赖 PATH
python run_tests.py               # 任何 Python 解释器：自动切到 .venv 里的那个
.venv/bin/python run_tests.py     # macOS / Linux（或 Windows 的 .venv\Scripts\python.exe）
```

等价于直接敲：

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -t .
```

脚本只多做了三件事：找 `.venv` 解释器（找不到就报错说清楚怎么建，**绝不用系统 Python 凑合**）、把控制台编码设成 UTF-8（中文测试输出在 GBK 控制台会炸）、最后打印退出码（0 = 全过，非 0 = 有失败）。多加的参数原样转给 unittest，例如 `run_tests.bat -v`。

**测试不联网、不需要 API Key、不加载任何真实模型、不碰 Docker**：全部用替身对象；需要这些能力的用例自己 skip。当前实测（2026-10-05，Windows 11 / Python 3.12.10）：

```
Ran 2360 tests in 109.813s
OK (skipped=13)
```

13 个 skip 的构成：7 个是 Docker 冒烟测试（本机没 Docker 或没有现成镜像）、4 个是符号链接测试（Windows 没开开发者模式，创建不了符号链接）、2 个是真模型评估回归（默认关闭，见「Embedding」一节）。CI 上（Linux）符号链接那 4 个会真跑。

RAG 效果回归单独跑：

```bash
.venv/Scripts/python.exe -m unittest tests.test_evaluation_regression -v
# 真模型那两组默认 skip；装了 sentence-transformers 且想跑：RAG_REAL_MODELS=1
```

## 静态检查

两个轻量工具，配置都在 `pyproject.toml`（那个文件**只放检查配置**，不是打包文件）：

```bash
.venv/Scripts/python.exe -m ruff check .   # 未用导入 / 未定义名字 / 明显语法与写法问题
.venv/Scripts/python.exe -m mypy           # 类型检查（只查生产代码，不查测试）
```

范围是刻意收窄的，为的是「检查拦得住真问题，又不逼着全库大改」：

- **ruff 只开 `E4` / `E7` / `E9` / `F`**（未用导入与变量、未定义名字、语法错误这一类）。工具的默认规则集要宽得多，开了会冒出上百条风格建议——那属于「改法见仁见智」，不是本轮目标。`target-version = "py310"` 顺带把「只有 3.12 能跑的写法」拦下来过。
- **mypy 配 `ignore_missing_imports`**（`openai`、`sentence-transformers` 都不带类型桩），`python_version = "3.10"`；`files` 只列生产代码——测试里的替身对象天生类型不整齐，查它只会逼出一堆无意义的标注。另外关掉 `override` 一类错误：`Tool.execute(**kwargs) -> str` 这个基类签名本来就是宽口径，12 个工具各自收窄参数是**有意设计**，不是错误。
- **`assert ... is not None` 是有意留的**：几处 helper 返回 `(值, 错因)` 这种一元组（`read_verified()`、`_read_task()` 等），「错因是 None 就代表值一定在」这个约定写在 docstring 里；mypy 不会顺着解包推出来，所以检查完错因补一句 `assert`——它同时也是给读代码的人的说明。运行期开销可以忽略。
- **测试不查类型**，只跑 ruff 的 `F` 系（未用导入这类真正的笔误）。评价 case、评分逻辑、阈值本轮一个字没动。

当前实测（2026-10-05，ruff 0.16.10 / mypy 2.4.0）：`ruff check .` → `All checks passed!`；`mypy` → `Success: no issues found in 73 source files`。

有几处 ruff 会更宽的规则集下报警、**这里明确保留不改**的写法，都是有意的：

| 位置 | 写法 | 为什么留着 |
| --- | --- | --- |
| `agent/history.py`、`tools/edit_file.py` | `tempfile.NamedTemporaryFile(delete=False)` 先拿名字、再 `with handle:` 写 | 原子写盘的标准姿势（写临时文件 → fsync → `os.replace`）；宽规则集会把「没在 with 里创建」当成泄漏，是误报 |
| `agent/trace.py`、`llm/client.py` | 观察者回调外面 `except Exception: pass` | 观察者（SSE 连接、日志）的错不该把一次成功的调用变成失败，代码里有注释说明 |
| `tools/vsenv.py` | `subprocess.run` 不传 `check=` | 探测类命令的返回码就是结果本身，不靠异常 |

## CI

`.github/workflows/ci.yml`，push 和 PR 都触发，两个 job 各跑一遍**全量 unittest**：

| job | 环境 | 步骤 |
| --- | --- | --- |
| `tests` | ubuntu-latest / Python 3.12 | 静态检查（ruff + mypy）→ `python run_tests.py`（顺带验证「自动切到 .venv」这条路径） |
| `windows` | windows-latest / Python 3.12 | `.\run_tests.bat`（顺带验证 `.bat` 入口在干净机器上能用） |

两个 job 都只做三件事：建 `.venv` → `pip install -r requirements-dev.txt` → 跑检查 / 测试。静态检查只放在 Linux job：它和平台无关，Windows job 的职责就是把 `.bat` 那一条路走通。**不需要 API Key、Docker daemon、真实 embedding 模型**，也不下载模型；任何一步失败退出码就非 0，job 直接红。不跳过任何测试来迎合 CI。

## 启动 API

```bash
python -m api                 # 默认 127.0.0.1:8000，只服务本机
python -m api --port 9000     # 换端口；--host 0.0.0.0 等于对局域网开放，先配 API_TOKEN
```

启动横幅会把当前安全配置打出来（会话 TTL / 允许的 Origin / Token 开没开 / 整轮超时），**不打 Token 本身**。

### 端点

| 方法 | 路径 | 请求体 | 响应 |
| --- | --- | --- | --- |
| GET | `/api/health` | 无 | `{"status": "ok"}`（不需要 Token，前端探活用） |
| POST | `/api/run` | `{"task": "...", "session_id": "..."（可省）}` | `{"success": true, "answer": "...", "session_id": "...", "run_id": "..."}` |
| POST | `/api/run/stream` | 同上 | SSE 事件流（`text/event-stream`） |
| POST | `/api/runs/cancel` | `{"session_id": "...", "run_id": "..."（可省）}` | 把正在跑的那一轮停下来（200 + `success=false` + `status`） |
| DELETE | `/api/sessions` | `{"session_id": "..."}` | 删掉这个会话 |

```bash
curl http://127.0.0.1:8000/api/health

curl -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Agent 的 tool-calling 循环在哪个文件里？"}'

# 接着用上一次返回的 session_id 追问（同一本会话历史）
curl -X POST http://127.0.0.1:8000/api/run \
  -H "Content-Type: application/json" \
  -d '{"task": "那个循环最多跑几步？", "session_id": "<上次的 session_id>"}'

# SSE：-N 关掉 curl 的缓冲，事件边跑边到
curl -N -X POST http://127.0.0.1:8000/api/run/stream \
  -H "Content-Type: application/json" \
  -d '{"task": "这个项目怎么做代码检索？"}'
```

### SSE 事件

| 事件 | 内容 |
| --- | --- |
| `start` | task / session_id / run_id |
| `llm_start` | model / attempt / max_attempts（prompt 一个字不推） |
| `llm_end` | 成没成 / 耗时 / 试了几次 / token 用量 / 错误摘要 |
| `tool_start` | 工具名 + 参数摘要（值截断到 120 字符，**不推正文**） |
| `tool_result` | 成没成 / 耗时 / 错误摘要（**不推工具返回正文**） |
| `completed` | success / answer / session_id / run_id（最终回答就在这） |
| `cancelled` / `timed_out` | 这一轮被停下的终态（同样 200，不是 5xx） |
| `error` | 这一轮失败（模型调用失败 / Agent 失败 / 意外异常） |

### 会话

`session_id` 由服务端在响应里发：请求不带它就新建一个会话，带了就接着用那本会话历史；认不出来的 id 回 404，不会悄悄换一只。同一个会话的两次 run 排队跑（会话锁），不同会话各跑各的，服务级没有一把大锁。会话超过 `API_SESSION_TTL` 没人用就过期清理，超过 `API_MAX_SESSIONS` 丢最久没用的那个。`DELETE /api/sessions` 删的是会话那本账，共享的工具 / 索引 / Memory 一个都不动。

### 取消与整轮超时

每次 run 请求服务端发一个 `run_id`，并建一个独立的取消开关。`POST /api/runs/cancel` 按 `session_id`（可加 `run_id`）把那一轮停下来——**协作式**：Agent 在检查点上发现取消才停，不杀线程、不硬杀子进程；正在跑的那个工具让它自然结束。被停下的这一轮回 200 + `success=false` + `status`（`cancelled` / `timed_out`）——服务没出错，是这一轮被停了。cancel 请求只碰登记表和开关，**不取会话锁**，所以卡不住正在跑的那一轮。

整轮超时用的是同一个开关：`AGENT_RUN_TIMEOUT` 设正整数后，从请求受理到结束（LLM + 工具 + 修复共享）总预算用完，下一次检查点按「运行超时」停下。它是协作式的，单个工具自己的 `timeout` 照旧生效——工具超时只是工具结果，整轮超时才是这一轮被停下。

### 访问控制

按顺序检查三样，默认（本机开发）全部放行：

- **Origin**：带 `Origin` 头的请求必须落在 `API_CORS_ORIGINS` 里，否则 403——挡的是「本机随便一个网页对 localhost 发简单请求」（本服务的 `run_command` / `edit_file` 是真会动系统的）。命令行客户端不带 Origin，不受影响；允许的 Origin 会拿到 CORS 响应头，`OPTIONS` 预检单独处理。
- **Token**：配了 `API_TOKEN` 后，除 `/api/health` 外都要带 `Authorization: Bearer <token>`，缺 / 错回 401。比较用 `hmac.compare_digest`（定时安全）。
- **Content-Type**：POST / DELETE 必须声明 `application/json`，否则 415。

错误出口都是 JSON：400 参数不对、401 没 Token、403 Origin 不允许、404 路径 / 会话 / Run 不存在、405 方法不对、413 请求体太大（1 MiB）、415 不是 JSON、500 Agent 失败、502 模型调用失败。**取消和超时不是错误**。

## 命令行对话

```bash
python main.py
```

输入问题回车即可，输入 `exit` / `quit` 退出；同一进程内的对话保留上下文，可以直接追问。每轮回答之前先打这一轮的工具调用记录（Trace）。启动横幅会打出当前模型、全部工具名、embedding 模式、Memory 状态，一眼看清这次跑的是什么配置。

## 能力概览

### 代码检索（RAG）

四层链：`ProjectIndex`（全量扫项目，每文件一条记录）→ `SymbolIndex`（类 / 函数 / 方法，Python 走 `ast`、C/C++ 走逐行扫描）→ `ChunkIndex`（按符号边界切真实代码）→ `Retriever`（字面打分：符号名 / 路径 / 正文，排序确定，同一句查询跑一百遍顺序一样）。工具 `search_code_chunks` 把它接进 Agent：**只给位置不给正文**（`相对路径:起止行号 符号 [类型] 分数 命中原因`），模型拿行号自己去 `read_file` 读当前代码。

- **embedding 可选**：`off`（默认，纯字面）/ `fake`（离线假货，演示）/ `local` / `multilingual`（本地真模型）。语义分只是 scorer 列表里多出来的第四条，返回格式不变。
- **显式刷新**：文件增删改之后调 `SearchChunksTool.refresh()`（或对应索引的 `refresh()`）——全量重建，删掉的文件不残留，语义那一侧的过时向量（含「id 都没了」的）同步清掉，下次检索用最新代码。不做后台文件监听。
- **评估**：自带 43 条 case（37 正 + 6 反，Python 30 / C++ 7，中文 20 / 英文 17，词法 12 / 语义 25），`python -m index [--hybrid|--local|--multilingual]` 最多四组同台比。`tests/test_evaluation_regression.py` 把它冻结成效果回归：字面 baseline 实测 Hit@1 10/37、Hit@3 12/37、Hit@5 12/37（2026-10-05），下限留了余量，掉了才报警。

### 调用图

`callgraph/` 从符号索引上建图：Python 用 `ast` 提调用点，C/C++ 逐行扫。工具 `call_graph` 支持三种查法：`callers`（谁调用了它）、`callees`（它调用了谁）、`path`（从 A 怎么走到 B）——同样只给名字和位置，不给正文。图是懒建的、建一次缓存；符号索引被刷新过，缓存图自动作废、下次查询重建（`CallGraphTool.refresh()` 可显式重建）。

### Memory

任务历史落本地 JSONL（默认 `~/.cache/universal-code-agent/memory.jsonl`，项目之外）。每轮任务：开始前按输入做轻量 token 匹配检索（不接 embedding），把有限的几条裁好、带上「仅供参考」抬头注入；结束后记一条（task / summary / outcome / 时间）。两个方向都**降级不炸**：检索失败、写不进去，都只影响 Memory 自己，任务照常。上限（默认 500 条 / 2 MiB）到了自动丢最旧的，去重按「三元组完全相同只留最新」。`AGENT_MEMORY=off` 整个关掉。

### Sandbox

命令执行的统一接口（`sandbox/base.py`）：`command / cwd / workspace / timeout / stdout / stderr / exit_code / 超时状态`，两种后端——`LocalSandbox`（宿主机，默认）和 `DockerSandbox`（每次执行一个一次性容器）。`run_command / run_tests / compile / build` 四个执行类工具都从这里走，工作目录锁在工具 root 以内。装配处默认 local；换后端是显式决定：`SandboxManager(kind="docker", image=...)` 或给工具传 `sandbox=`。

### 修复循环

`edit_file` 改完代码自动接着验证：C/C++ 先编译 / 构建，过了再跑 `run_tests`；Python 直接跑测试。验不过就进修复循环——模型想收尾会被按回去接着改，直到修好或按五条规矩停（用满轮数 / 改动重复 / 连着两轮一样 / 文件没变 / 只回话不动手）；不自动回滚，停在哪就是哪，最终回答里附 Repair Summary。每轮 run() 的过程写一份 JSON 到 `repair_history/`（写失败只记警告，不影响任务）。

## Docker Sandbox 的前置条件和测试 skip 原因

用 DockerSandbox 需要三样**都**满足：

1. 本机装了 Docker CLI（`docker` 在 PATH 上）；
2. daemon 正在运行（`docker version` 能通）；
3. 要用的镜像**本地已经有了**——这个实现**不自动 pull**（`--pull=never`），没有就报错说清楚，也不会偷偷换回宿主机执行。

设计上是每次执行一个 `docker run --rm` 的一次性容器，workspace 以 bind mount 挂在容器的 `/workspace`，只传调用方显式给的 `-e` 环境变量；超时被我们打断的容器由 `docker rm -f` 点名杀掉（正常结束由 `--rm` 收，两条路都收干净）。

测试（`tests/test_docker_sandbox.py`）的行为：导入时探测一次——没有 docker / daemon 没起 / 候选镜像（`ubuntu:24.04`、`python:3.12-slim` 等）一个都没有，整个 DockerSmokeTest **跳过**（当前 7 个 skip 就是它），提示语是「本机没有可用的 Docker 或现成镜像（不 pull 任何东西）」。本机有镜像时可以用 `UAC_TEST_DOCKER_IMAGE=你的镜像` 指定它。**这条线一个字节都不联网。**

## Embedding：可选依赖与降级

`SEARCH_CHUNKS_EMBEDDING` 四档：

| 值 | 需要什么 | 说明 |
| --- | --- | --- |
| `off`（默认） | 什么都不用 | 纯字面检索，行为和没有 embedding 之前一字不差 |
| `fake` | 什么都不用 | 词表哈希凑的假向量，演示 / 联调整条语义管线用，不背语义质量 |
| `local` | `pip install sentence-transformers` | 本地英文小模型（默认 `all-MiniLM-L6-v2`），英文查询 |
| `multilingual` | 同上 | 本地多语言模型，中文查询用这个 |

降级与懒加载的规矩：

- **模型懒加载**：构造工具、注册工具、Agent 启动都不碰模型；第一次真正调用 `search_code_chunks` 才加载（`off` 全程不加载、不联网）。
- **坏了自己退**：没装库、模型下不动、缓存文件损坏、打分中途抛异常——一律被 `DegradingScorer` 兜住，退回字面检索，工具照常出结果，错因记在 `tool.embedding_error` 里。**一次为限，失败不重试**。
- **向量缓存**：`~/.cache/universal-code-agent` 下按「模型 + 文本拼法 + 分段策略」命名空间存 JSON；chunk 的 key 带 digest，换模型 / 改文件 / 换策略都自动失效；缓存读写失败只降级（回到「没有缓存」，慢一点而已）。缓存目录在项目之外，不会被代码索引扫到，也不提交任何向量数据。
- **测试不需要它**：全部 embedding 测试用注入的假模型，不装 sentence-transformers 也能全绿；真模型评估回归默认跳过，`RAG_REAL_MODELS=1` 才跑（装不上 / 下不动一律 skip，不判失败）。

## 已知限制

- **`run_command` 超时杀的是 shell**：被 shell 启动的子进程可能还活着（Windows 上尤其明显，`cmd.exe` 死了下面的 `python` 还在跑），超时那一路也拿不到任何输出。彻底解决要进程组 / Job Object，还没做。
- **C/C++ 结构提取是逐行近似**：宏展开、宏生成的名字、复杂模板、函数指针、`operator[]` 这类认不出来——宁可漏也不报错；函数里的局部类型也不报。
- **索引是显式刷新**：不做后台文件监听、不做增量 AST；刷新是全量重建，大仓库上比增量慢。刷新与并发查询的极端交错下，理论上某个已删块的向量可能被重新写回缓存（无害孤儿条目，索引本身不会用它）。
- **效果回归的下限是快照**：字面 baseline 在 43 条 case 上 Hit@1 只有 10/37——语义那 25 条字面打分够不着是数据集**故意**的；真模型在反例上的假阳性（当前实测 3/6）是「语义检索改写『查不到』」的既有现象，没有改评分逻辑去凑数。
- **Memory 是轻量 token 匹配**：不接 embedding，中文按字、英文按词；检索不到就是没有，不猜同义词。
- **对话历史**：命令行进程退出即丢；API 会话有 TTL，过期即清。都没有跨进程持久化（Memory 存的是任务记录，不是对话）。
- **Windows 上符号链接测试会 skip**：没开开发者模式 / 没权限时创建不了符号链接，4 个用例跳过（代码路径有直接覆盖，Linux CI 上会真跑）。
- **Docker sandbox 目前不在默认装配里**：工具默认走 LocalSandbox，换容器要显式给 `kind="docker"` + 镜像；环境变量开关还没有。
- **API 是单机定位**：标准库 `ThreadingHTTPServer`，没有 TLS、没有多进程；对局域网开放前先配 `API_TOKEN` 和 `API_CORS_ORIGINS`。

## 更多文档

- [docs/PHASES.md](docs/PHASES.md)——开发阶段记录（每个阶段做了什么、量出了什么，原文存档）；
- [docs/NOTES.md](docs/NOTES.md)——设计说明逐条：每个行为背后的取舍与实测结论（工具边界、上下文裁剪、超时、诊断解析、提示词演化……）。
