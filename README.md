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
- **Agent 工作台**：React + TypeScript 单页前端（`frontend/`），SSE 实时过程 —— 工具卡片 / 代码位置 / 调用链 / 运行时间线，会话记录只存在本机浏览器里；
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
- [启动前端（React）](#启动前端react)
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
│                    #   locations.py：工具输出 → 位置信息（给工作台用，本身不是工具）
├── api/             # HTTP 服务（python -m api）：server / sessions / runs / sse
├── frontend/        # React + TypeScript 前端（Vite）：会话 + Agent 工作台 + SSE 实时过程
│                    #   工具卡片 / 代码位置 / 调用链 / 运行时间线（14.3）
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
Ran 2474 tests in 112.860s
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

CI 目前**只覆盖 Python 那一半**：前端的三条（`npm run typecheck` / `npm test` / `npm run build`）要本地手动跑，见上面「前端自己的检查」。

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
| `tool_result` | 成没成 / 耗时 / 错误摘要 / `locations`（代码位置，见下；**不推工具返回正文**） |
| `completed` | success / answer / session_id / run_id（最终回答就在这） |
| `cancelled` / `timed_out` | 这一轮被停下的终态（同样 200，不是 5xx） |
| `error` | 这一轮失败（模型调用失败 / Agent 失败 / 意外异常） |

`tool_result` 里那个 `locations` 是**位置，不是内容**：每条只有 `path` / `line` / `end_line` / `symbol` / `kind` 五个字段，从工具**自己固定的**输出格式里按白名单提出来（`tools/locations.py`），一次最多 50 条。它做的是「把 `search_code` 找到的 `文件:行号` 变成前端上能看、能复制的位置卡片」这件事，**工具返回正文一个字都不进这一帧**——命中行的代码、RAG 的分数与命中原因、编译器的诊断原文，全都留在服务端。白名单目前是 `search_code` / `search_code_chunks` / `call_graph` / `compile` / `build` / `edit_file`；别的工具（`run_command` / `run_tests` / `calculator` …）一律给空数组——往任意输出里捞「文件:行号」会捞出一堆似是而非的东西，**不猜、不倒推**。

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

## 启动前端（React）

单页 Web 前端（Agent 工作台）：左边是会话列表（新建 / 当前高亮 / 删除），右边是当前会话 —— 顶部标题 + 状态，中间按「用户任务 → 运行状态 → 工具/模型调用（默认折叠，点开是位置卡片 / 调用链 / 时间线）→ 最终回答」列出来，底部是任务输入框 + 运行 / Cancel。接口还是原来那五条、没有新增：14.3 唯一的后端改动是 `tool_result` 帧多带一个 `locations`（见上面「SSE 事件」），**只加字段、不加接口**，Agent 核心没动。

```bash
# 1) 后端（一个终端）。先把前端的地址加进允许列表，在 .env 里写：
#      API_CORS_ORIGINS=http://localhost:5173
python -m api                              # 默认 127.0.0.1:8000

# 2) 前端（另一个终端）
cd frontend
npm install                                # 第一次才要；拉过新代码之后也要跑一次
npm run dev                                # http://localhost:5173
```

然后浏览器打开 `http://localhost:5173`：写一句任务 → 运行，做没做完都看得到；跑着的时候可以 Cancel。

前端依赖不多：运行时只有 `react` / `react-dom` 和渲染回答用的 `react-markdown` / `remark-gfm`；开发期是 `vite` / `typescript` / `vitest` / `jsdom`。版本锁在 `frontend/package-lock.json`（由 `npm install` 维护，不手改）。

### 后端地址怎么配（VITE_API_BASE_URL）

前端的环境变量在 `frontend/.env`（从 `frontend/.env.example` 复制，**不要提交**）：

| `VITE_API_BASE_URL` | 什么时候用 |
| --- | --- |
| 不写这行 | 用后端默认地址 `http://127.0.0.1:8000`（本地开发就是它） |
| `VITE_API_BASE_URL=`（空值） | 同源 `/api/...`：前端和后端由同一个域名 + 端口提供时 |
| `VITE_API_BASE_URL=http://别的主机:8000` | 后端在别处 |

**不管选哪种，都要把页面地址加进后端的 `API_CORS_ORIGINS`**（例如 `http://localhost:5173`），改完重启后端。这不是「跨源才要配」：浏览器发 POST 时一定会带 `Origin` 头，而后端的设计是「带了 `Origin` 就必须在允许列表里」（`api/server.py` 的 `_origin_allowed`），**同源也照样挡**——这一条正是它防「本机别的网页对本服务发请求」的办法（本服务的 `run_command` / `edit_file` 是真会动系统的）。

没配会怎样：页面左下角直接显示「来源不被允许」，并把要加的那一行原样写出来（探活也会吃 403，所以进页面就能看到，不用等到运行）。忘了配就点运行的话，错误提示里同样带着这句地址。

顺带说一句：`npm run dev` **没有**配 `/api` 转发。转发并不能绕开后端的来源检查（浏览器发 POST 一定带 `Origin`，转发是原样交给后端的），只会让「不用配 CORS」这个错印象传下去。

后端配了 `API_TOKEN` 的话，在页面左下角填 Token（只存在内存里，刷新就没了，不写进浏览器存储）。

### 前端自己的检查

```bash
cd frontend
npm run typecheck   # tsc：浏览器代码 / vite.config / 测试 三份配置分别查
npm test            # vitest：SSE 解析 + 记录生成 + 落盘 + 工具 / 位置 / 调用链卡片 + Markdown 安全 + 接口联调
npm run build       # 类型检查 + 生产构建到 frontend/dist
```

测试不联网、不需要 API Key：假后端是本机回环上的一个 `node:http` 服务器，`useRun` 那几条把网络层换成脚本，其余全是真的（真的记录、真的 localStorage、真的 jsdom）。

### 记录（Transcript）存在哪

刷新页面、切会话都不丢；**只存在你自己的浏览器里**（localStorage），不上传、后端也不知道有这份东西。

| 存什么 | 说明 |
| --- | --- |
| 每个会话一份，键是 `uac.frontend.transcript.v1.<session_id>` | 一份坏掉不牵连别的；删会话就是删这一个键 |
| 任务原文、每一轮的状态与时间、最终回答（Markdown 原文） | 回答最多留 20000 字，超了截断并写明「本地只留了前一段」 |
| 工具 / 模型调用的**摘要**：名字、成功失败、用时、tokens、错误那句话；工具行另带**分类**（检索 / 读文件 / 改文件 / 调用图 / 跑命令 / 编译构建 / 计算 / 其它）、一行「在看什么」，以及**位置**（`路径:行号` / 符号 / 类型） | 每轮最多留 200 条过程记录；超了只丢过程，**次数统计照样准**。位置每次调用最多留 12 条，另有总数（界面说「还有 N 处」） |
| 每个会话最多 20 轮 | 再往前的丢掉（localStorage 不是数据库） |

**不存的**（有意为之）：

- **API Token**——它只活在页面内存里，刷新就没了；写进 localStorage 是被测试钉死不允许的（`useRun.test.tsx` 里有一条专门断言盘上既没有 Token 也没有含 `token` 的键）。
- **原始 SSE JSON**——事件读进来就翻译成记录，字节不落地。
- **工具的调用参数和被读文件的内容**——参数一个键都不落地（记录里根本没有 `arguments` 这个字段）。工具行上那一句「在看什么」是从参数里**派生**出来的白名单字段（`read_file` 的路径、`run_command` 的命令、`call_graph` 的函数名……），只此一行；参数里 `old_text` / `new_text` / `content` 这类可能带着整份源码的值，连同一行没白名单的键，全扔掉。位置信息同理：只留 `path` / `line` / `symbol` / `kind`，**行号旁边的代码一个字不留**。Prompt 也一概不存。

几条要说明的行为：

- **刷新之后不会显示「运行中」**：读回来的每一份记录都会把「还在跑」的那些轮标成**已中断**，并写明「后端可能还在跑——断线不等于取消」。想要结果就重新问一次；不显示假的 Running，是因为那会让人一直等一个永远不会来的结果。
- **删除会话会把本机那份记录一起删掉**；「清空记录」只清本机这份，后端那个会话（它的历史）不动。
- **本地存的坏了就当没存过**：JSON 解析不了、版本对不上、字段类型不对，一律退回空记录，页面不会因此崩；手改过的数据也不会被当成正常记录显示。
- **浏览器存不下时（配额满 / 隐私模式）**：先清掉更早的会话腾地方，还不行就砍自己最旧的几轮。真的写不进去时页面顶上会说一句「本机存不下记录了」——这一轮照样跑完，只是这次的结果留不到下次打开。

### Markdown 与安全

Agent 的最终回答按 Markdown 渲染（标题、列表、表格、代码块），用的是 `react-markdown` + `remark-gfm`，安全性靠**结构**而不是靠过滤字符串：

- 全程转成 React 元素，**没有 `dangerouslySetInnerHTML`**；
- **不装 `rehype-raw`**（那是唯一会把 Markdown 里的 HTML 当 HTML 渲染的插件）。所以回答里写 `<img src=x onerror=...>`、`<script>...` 只会**原样显示成一串字**；`a < b` 这种比较符也照样是文本；
- 链接地址由 `react-markdown` 的默认转换器过一遍：`javascript:` / `data:` 这类协议会被换成空串；
- 回答里的图片**不自动加载**——只显示成「[图片] 说明」的链接（不然一句话就能拿你的 IP 去外面报个信）；
- 外链一律新窗口打开并带 `rel="noreferrer noopener"`。

上面每一条都有测试钉着（`src/components/Markdown.test.tsx` 往渲染里灌 XSS payload，然后翻 DOM 断言没有 `<script>` / `<img>` / `on*` 属性）。哪天有人为了「支持 HTML」加上 `rehype-raw` 或改成 `dangerouslySetInnerHTML`，那些用例会红。

### 前端这一层做了什么

- **SSE 优先**：默认走 `POST /api/run/stream` 边跑边显示，九种事件（`start` / `tool_start` / `tool_result` / `llm_start` / `llm_end` / `completed` / `cancelled` / `timed_out` / `error`）全都处理，终态一定落到界面上。用 fetch + ReadableStream 自己读流——浏览器的 `EventSource` 发不了 POST、也带不了 `Authorization` 头，用不了。
- **只显示该显示的**：工具行是 `🔧 工具名 / 成功 / 用时`，模型行是 `LLM / 模型名 / attempt 2/3 / 用时 / tokens`，出错时多一句错误摘要。**不显示** prompt、API Key、工具参数原文和 Python 异常栈——后端发来的 `error` 里如果是不认识的那种（`异常类型: 消息`），压成一句白话再显示（原文进浏览器 console）。
- **一轮的样子**：用户任务 → 运行状态（跑着的时候显示 run_id 短号、当前在干什么、工具几次、模型几次、已跑多久、最近一条事件）→ 过程事件（**默认折叠**，点开才看）→ 最终回答。历史轮次整块折叠，长日志不会把回答淹掉。
- **代码分析看得见**（14.3）：工具行按**类型**分（检索 / 读文件 / 改文件 / 调用图 / 跑命令 / 编译构建 / 计算 / 其它），还带着状态、用时和一句「在看什么」（做的是什么，不是参数原文）。认不出来的工具**不崩**：归到「其它」，失败就按失败显示。点开一条工具事件，有位置信息就是**位置卡片**（`路径:行号`、符号、类型），带**复制路径** / **复制 `path:行号`** 两个按钮；`call_graph` 走 `path` 查法时画成 `A → B → C` 的**调用链**（有环就画到重复处为止，超过 8 个节点截断并写明「还有 N 跳」，不画图库、纯 DOM）；检索类工具给命中条数。位置只显示后端给了的——**没有位置就什么都不显示，绝不按别的行倒推一个行号**。
- **不对做不到的事装作能做到**：位置卡片里的「在编辑器打开」按钮是**灰的**，写明原因（浏览器打不开本地文件，要编辑器支持才行）。不做一个点了没反应的假按钮。
- **时间线**：每条过程事件左边一列时间戳（`09:31:02`），按事件顺序排。这个时刻是**浏览器收到那一帧的本地时间**——SSE 帧里本来就没有时间字段，所以不编一个「后端时间」出来充数。历史轮次同样看得到，一行的厚度，不占版面。
- **状态分两处**：会话（有哪些、哪个在用）和当前这一轮 Run（跑没跑、什么终态）是两套状态，互不干扰。
- **取消用真 run_id**：后端在 `start` 事件里发的那个，前端不自己编；它到之前 Cancel 是灰的。后端收下取消之后如果迟迟不发终态事件（连接卡住），10 秒宽限期一到就按「已取消」记下来并写明原因——界面不会永远转圈。
- **流断了不冒充停止**：断线不等于取消（后端的约定），所以只说「结果看不到了」，不说「已停止」。
- **同一轮的事件不会记两遍**：每一轮记着自己吃进去多少帧，重复的帧直接丢掉；已经终态的那一轮再收到什么都不改动。
- **错误说清楚下一步**：连不上 / 来源不被允许（403，左下角直接把要加进 `API_CORS_ORIGINS` 的那一行写出来）/ 400、404 用后端自己的话（说的就是这次请求哪儿不对）/ 5xx 压成一句中文（后端的 Python 异常原文只进浏览器 console）。网络层失败也会在本机记一条 failed，不会「什么都没发生」。
- **会话记在前端**：后端没有「列出会话」的接口、也没把会话落盘，所以前端用 localStorage 记着用过哪些 `session_id`（id / 标题 / 时间）以及每个会话的记录（见上面「记录（Transcript）存在哪」）；后端说它不存在了（404）就把这条划掉，重新发一次会开新会话。
- **不放密钥**：Token 只在内存里，刷新就没了，**不写进 localStorage**（有测试钉着）；它只以 `Authorization` 头随请求发给后端。渲染走 React 元素，全项目没有 `dangerouslySetInnerHTML`。

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
- **工作台的位置是「后端给什么显示什么」**：只有白名单工具（`search_code` / `search_code_chunks` / `call_graph` / `compile` / `build` / `edit_file`）命中固定输出格式才有位置卡片；`run_command` 这类工具即使输出里出现「文件:行号」也不显示（宁可没有，也不猜）。**「在编辑器打开」做不到**——浏览器打不开本地文件，那个按钮是灰的，只能复制路径。
- **调用链是有损的**：只有 `call_graph` 的 `path` 查法画成 `A → B → C` 的链，一次最多 8 个节点，有环画到重复处为止；`callers` / `callees` 显示成位置列表，不是图。是纯 DOM 拼的，不是图形库、也不是完整调用图视图。
- **API 是单机定位**：标准库 `ThreadingHTTPServer`，没有 TLS、没有多进程；对局域网开放前先配 `API_TOKEN` 和 `API_CORS_ORIGINS`。

## 更多文档

- [docs/PHASES.md](docs/PHASES.md)——开发阶段记录（每个阶段做了什么、量出了什么，原文存档）；
- [docs/NOTES.md](docs/NOTES.md)——设计说明逐条：每个行为背后的取舍与实测结论（工具边界、上下文裁剪、超时、诊断解析、提示词演化……）。
