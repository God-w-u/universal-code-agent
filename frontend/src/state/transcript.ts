/**
 * 一个会话里的记录：**按「轮（run）」分组**。一轮 = 用户发一次任务到 Agent
 * 给出终态，过程事件（工具调用、模型调用）挂在它自己那一轮下面。
 *
 * 为什么按轮分组：14.1 是一条平铺的流水账，回看时「哪些工具是这一轮调用的」
 * 全靠猜。分组之后，历史轮次可以整块折叠、整块浏览，正在跑的那一轮单独展开。
 *
 * 全是纯函数：不改输入，只返回新对象。组件只管画，规则都在这里，测试里喂几条
 * 事件就能验（见 transcript.test.ts）。时间一律由调用方传 `now` 进来——这样
 * 测试不用跟真实时钟赛跑。
 *
 * **配对靠「最近一条还没结束的」**：后端的 tool_start / tool_result 只带工具名
 * 和结果，不带调用 id（协议不改）。Agent 是一轮一轮串着跑的，同一个工具不会
 * 同时跑两次，所以「最近一条还在跑的同名工具」就是它；llm_start / llm_end 同理。
 *
 * **重复帧靠 cursor 挡**：一轮记着自己吃进去多少帧（cursor = 最大帧号）。
 * 同一帧再来一次（seq ≤ cursor）直接丢掉。工具调用参数、模型 prompt 一概不进
 * 记录——协议里本来就没有 prompt，工具参数也被后端截断过，但里面可能带着被读
 * 文件的内容，所以**只在内存里看一眼就扔，不落地**。
 *
 * 14.3 给工具事件添了两样**派生**的东西（都不是参数原文）：
 * - `intent`：一行「在干什么」，只从白名单的那一个参数键拼出来（见 toolIntent），
 *   命令行、新旧正文、表达式一概不取；
 * - `locations`：这次调用落在哪个文件第几行，来自 tool_result 的 locations
 *   字段——后端只从认得输出格式的工具里提出位置，正文一个字都不带（见
 *   tools/locations.py）。一条事件最多存 12 条，真有几条记在 locationCount 里。
 */

import type { RunEvent, RunStatus, ToolLocation } from "../api/types";

/** 终态：这一轮不会再变了。interrupted 是前端判的（见 markInterrupted）。 */
export type RunOutcome = "completed" | "cancelled" | "timed_out" | "failed" | "interrupted";

/** 一轮的状态：running + 五种终态。 */
export type RunPhase = "running" | RunOutcome;

/** 状态条上显示的东西：idle = 这个会话还没有跑过。 */
export type DisplayState = "idle" | RunPhase;

/**
 * 工具属于哪一类（14.3）。界面按这个换图标和说明文字：检索的、读文件的、
 * 改文件的、看调用图的、跑命令的、编译构建的、算数的，剩下的都算 other。
 *
 * 认不出来的工具**一律 other**，照样显示工具名和成败——不认识不等于不能显示。
 */
export type ToolFamily = "search" | "read" | "edit" | "graph" | "run" | "build" | "compute" | "other";

/** 工具名 → 类别。表里没有的一律 other（见 ToolFamily 的说明）。 */
const TOOL_FAMILIES: Record<string, ToolFamily> = {
  search_code: "search",
  search_code_chunks: "search",
  read_file: "read",
  list_files: "read",
  code_summary: "read",
  edit_file: "edit",
  call_graph: "graph",
  run_command: "run",
  run_tests: "run",
  compile: "build",
  build: "build",
  calculator: "compute",
};

export function toolFamily(name: string): ToolFamily {
  return TOOL_FAMILIES[name] ?? "other";
}

export interface ToolRecord {
  kind: "tool";
  id: string;
  tool: string;
  /** 哪一类工具（换图标、换说明用）；见 toolFamily。 */
  family: ToolFamily;
  /**
   * 这次调用在看什么（一行，比如「读 main.py」「关键词 "def "」）。
   *
   * **只从白名单参数里取**（见 toolIntent）：read_file 的 path、search 的
   * keyword……run_command 的命令行、edit_file 的新旧正文一概不取。取不到
   * 就是 null——宁可不写，也不把参数摊开。
   */
  intent: string | null;
  state: "running" | "ok" | "fail";
  durationSeconds: number | null;
  error: string | null;
  /** 这次调用落在哪几段（14.3）；提不出来就是空数组。 */
  locations: ToolLocation[];
  /** 一共收到多少条位置（locations 只是留下来的前几条，见 MAX_EVENT_LOCATIONS）。 */
  locationCount: number;
  /** 这些位置是不是**一条有序的调用链**（call_graph 的 path 模式）。 */
  chain: boolean;
  /** 这条是什么时候来的（毫秒时间戳）——「最近一次事件」要报时间。 */
  at: number;
}

export interface LlmRecord {
  kind: "llm";
  id: string;
  model: string;
  attempt: number;
  maxAttempts: number;
  state: "running" | "ok" | "fail";
  durationSeconds: number | null;
  totalTokens: number | null;
  error: string | null;
  at: number;
}

/** 挂在这一轮下面的一句话（取消结果、断流说明这类），不是工具/模型事件。 */
export interface NoticeRecord {
  kind: "notice";
  id: string;
  level: "info" | "warn" | "error";
  text: string;
  at: number;
}

export type RunEventRecord = ToolRecord | LlmRecord | NoticeRecord;

export interface RunRecord {
  /** 本地 id：run_id 还没到（或永远没到）时也有个身份，用来定位这一轮。 */
  key: string;
  /** 后端在 start 事件里发的 run_id；还没到就是 null。 */
  runId: string | null;
  task: string;
  startedAt: number;
  endedAt: number | null;
  phase: RunPhase;
  /** 最终回答（Markdown 原文，渲染时再转）。 */
  answer: string | null;
  /** 终态那句话：取消 / 超时 / 失败 / 中断的原因。 */
  message: string | null;
  events: RunEventRecord[];
  /** 工具调用次数（events 被截断后仍然准）。 */
  toolCount: number;
  /** 模型调用次数。 */
  llmCount: number;
  /** 一共来过多少条过程事件（events 只是保留下来的那部分）。 */
  eventCount: number;
  /** 已经吃进去的最大帧号，用来挡重复帧。 */
  cursor: number;
  /** events 是否因为太长被截过（界面上要说一声「较早的过程没留下」）。 */
  truncated: boolean;
}

export const TRANSCRIPT_VERSION = 1;

export interface Transcript {
  version: number;
  /** 从旧到新；新的在最后（和画出来的顺序一致）。 */
  runs: RunRecord[];
  updatedAt: number;
}

/** 单个会话最多留几轮：再往前的丢掉（localStorage 不是数据库）。 */
export const MAX_RUNS = 20;
/** 一轮最多留几条过程事件：只影响「过程」，toolCount / llmCount 照样准。 */
export const MAX_EVENTS = 200;
/** 最终回答最多存这么多字。 */
export const MAX_ANSWER = 20_000;
/** 终态说明 / 报错最多存这么多字（后端已经截过一轮，这里再兜一次）。 */
export const MAX_TEXT = 500;
/**
 * 一条工具事件最多**存**几条位置。
 *
 * 后端一帧最多给 50 条（tools/locations.py 的 MAX_LOCATIONS）；界面上一屏也
 * 看不了那么多，而 localStorage 是有配额的。所以只存前 12 条，真有几条记在
 * locationCount 里，界面上写「显示前 12 处，还有 N 处」——**不假装只有这些**。
 */
export const MAX_EVENT_LOCATIONS = 12;
/** 一行「在干什么」最多存这么多字（来源本身就被后端截过一轮）。 */
export const MAX_INTENT = 120;

const TRUNCATED_ANSWER = "\n\n…（回答太长，本地记录只留了前一段；完整内容在后端的会话里）";
const TRUNCATED_TEXT = "…（已截断）";

export function emptyTranscript(): Transcript {
  return { version: TRANSCRIPT_VERSION, runs: [], updatedAt: 0 };
}

// ---------------------------------------------------------------------------
// 改记录
// ---------------------------------------------------------------------------

// 事件的本地 id：自增就够（不用 crypto），测试里也好对
let idSeq = 0;

function nextEventId(): string {
  idSeq += 1;
  return `ev${idSeq}`;
}

/** 一轮的本地 key。带上时间戳：刷新之后接着跑，也不会和存着的那批撞。 */
export function nextRunKey(now: number): string {
  idSeq += 1;
  return `run-${now.toString(36)}-${idSeq.toString(36)}`;
}

/** 开一轮：任务先记上，状态是 running，等着 start 事件把 run_id 送来。 */
export function beginRun(t: Transcript, task: string, now: number, key: string): Transcript {
  const run: RunRecord = {
    key,
    runId: null,
    task: clampText(task, MAX_ANSWER),
    startedAt: now,
    endedAt: null,
    phase: "running",
    answer: null,
    message: null,
    events: [],
    toolCount: 0,
    llmCount: 0,
    eventCount: 0,
    cursor: 0,
    truncated: false,
  };
  // 新的在最后；超出的丢最旧的（slice(-N) 在少于 N 条时就是原样）
  return { ...t, runs: [...t.runs, run].slice(-MAX_RUNS), updatedAt: now };
}

/**
 * 一帧事件落进某一轮。
 *
 * 两种情况下原样返回（调用方可以据此跳过落盘）：
 * - 这一轮已经终态了：终态之后来的东西一律不算数（后端发完 completed 就关
 *   连接，读端再冒出什么都不该改这一轮的结果）；
 * - 这一帧吃过了（seq ≤ cursor）：同一次执行的事件流被应用两遍也只有一份。
 */
export function applyFrame(
  t: Transcript,
  key: string,
  event: RunEvent,
  seq: number,
  now: number,
): Transcript {
  const index = t.runs.findIndex((run) => run.key === key);
  if (index < 0) return t;
  const run = t.runs[index];
  if (run === undefined) return t;
  if (run.phase !== "running" || seq <= run.cursor) return t;

  const updated = applyEvent(run, event, seq, now);
  if (updated === run) return t;

  const runs = [...t.runs];
  runs[index] = updated;
  return { ...t, runs, updatedAt: now };
}

/** 给某一轮挂一句话（取消结果、断流说明）。终态之后不再挂。 */
export function addNotice(
  t: Transcript,
  key: string,
  text: string,
  level: NoticeRecord["level"],
  now: number,
): Transcript {
  const index = t.runs.findIndex((run) => run.key === key);
  if (index < 0) return t;
  const run = t.runs[index];
  if (run === undefined) return t;

  const notice: NoticeRecord = {
    kind: "notice",
    id: nextEventId(),
    level,
    text: clampText(text, MAX_TEXT),
    at: now,
  };
  const runs = [...t.runs];
  runs[index] = pushEvent(run, notice);
  return { ...t, runs, updatedAt: now };
}

/**
 * 本地判定的终态（没等到后端的终态事件时用）：网络断了、请求被拒、后端说
 * 这个会话没了……这一轮在前端这边结束了，但**不等于后端停了**。
 */
export function finishRun(
  t: Transcript,
  key: string,
  phase: RunOutcome,
  message: string,
  now: number,
): Transcript {
  const index = t.runs.findIndex((run) => run.key === key);
  if (index < 0) return t;
  const run = t.runs[index];
  if (run === undefined || run.phase !== "running") return t;

  const runs = [...t.runs];
  runs[index] = {
    ...run,
    phase,
    message: clampText(message, MAX_TEXT),
    endedAt: now,
    // 还在「运行中」的过程事件不可能有结果了：标成未知，不留一个转圈的圈
    events: run.events.map((event) => (event.kind === "notice" ? event : { ...event, state: "fail" as const })),
  };
  return { ...t, runs, updatedAt: now };
}

/**
 * 把「还在跑」的那些轮标成中断。**每次从 localStorage 读出来都要过一遍**：
 * 刷新之后没有任何一条流还活着，界面绝不能显示 Running——那是假象，用户会
 * 一直等一个永远不会来的结果。后端那一轮可能真的还在跑，所以措辞里说清楚。
 */
export function markInterrupted(t: Transcript, now: number): Transcript {
  if (!t.runs.some((run) => run.phase === "running")) return t;

  const runs = t.runs.map((run) =>
    run.phase === "running"
      ? {
          ...run,
          phase: "interrupted" as const,
          endedAt: now,
          message:
            "页面刷新（或关掉）时这一轮还没结束，记录停在中断处。后端可能还在跑——" +
            "断线不等于取消，要看结果得重新问一次",
          events: run.events.map((event) =>
            event.kind === "notice" ? event : { ...event, state: "fail" as const },
          ),
        }
      : run,
  );
  return { ...t, runs, updatedAt: now };
}

/** 清空：整份记录（那一轮后端会话不动——那是另一件事，见 README）。 */
export function clearTranscript(): Transcript {
  return emptyTranscript();
}

// ---------------------------------------------------------------------------
// 看记录
// ---------------------------------------------------------------------------

/** 正在跑的那一轮；没有就是 null。 */
export function currentRun(t: Transcript): RunRecord | null {
  for (let index = t.runs.length - 1; index >= 0; index -= 1) {
    const run = t.runs[index];
    if (run !== undefined && run.phase === "running") return run;
  }
  return null;
}

/** 最近一轮（不管跑没跑完）。 */
export function latestRun(t: Transcript): RunRecord | null {
  return t.runs.length === 0 ? null : (t.runs[t.runs.length - 1] ?? null);
}

/** 状态条上显示哪一个状态。 */
export function displayState(t: Transcript): DisplayState {
  const running = currentRun(t);
  if (running !== null) return "running";
  const last = latestRun(t);
  return last === null ? "idle" : last.phase;
}

export interface RunStats {
  tools: number;
  llms: number;
  /** 已跑时长（秒）：终态用 endedAt，还在跑就用传进来的 now。 */
  durationSeconds: number;
  /** 正在干什么（一行）。 */
  phaseText: string;
  /** 最近一条事件（一行）；一条都还没有就是空串。 */
  lastEventText: string;
  /** 最近一条事件是什么时候（毫秒时间戳）；没有就是 null。 */
  lastEventAt: number | null;
}

/**
 * 一轮的即时统计。界面上的「工具 3 次 / 模型 2 次 / 正在调用 xxx」都从这儿来，
 * 数字取自记录的计数（不是 events 的长度——events 可能被截断）。
 */
export function runStats(run: RunRecord, now: number): RunStats {
  const last = run.events.length === 0 ? null : (run.events[run.events.length - 1] ?? null);
  const end = run.endedAt ?? now;

  return {
    tools: run.toolCount,
    llms: run.llmCount,
    durationSeconds: Math.max(0, (end - run.startedAt) / 1000),
    phaseText: phaseTextOf(run, last),
    lastEventText: last === null ? "" : eventText(last),
    lastEventAt: last === null ? null : last.at,
  };
}

function phaseTextOf(run: RunRecord, last: RunEventRecord | null): string {
  if (run.phase !== "running") return outcomeText(run);
  if (last === null) return "已派给后端，等第一条事件";

  switch (last.kind) {
    case "tool":
      return last.state === "running" ? `正在调用工具 ${last.tool}` : `工具 ${last.tool} 已返回，等模型继续`;
    case "llm":
      return last.state === "running"
        ? `模型 ${last.model} 第 ${last.attempt}/${last.maxAttempts} 次调用中`
        : "模型已回复，Agent 正在处理";
    case "notice":
      return last.text;
  }
}

/** 终态的一句话（状态条上跟在状态后面）。 */
export function outcomeText(run: RunRecord): string {
  switch (run.phase) {
    case "running":
      return "运行中";
    case "completed":
      return "已完成";
    case "cancelled":
      return "已取消";
    case "timed_out":
      return "已超时";
    case "failed":
      return "失败";
    case "interrupted":
      return "已中断";
  }
}

/** 一条过程事件压成一行。 */
export function eventText(event: RunEventRecord): string {
  switch (event.kind) {
    case "tool":
      return `工具 ${event.tool}`;
    case "llm":
      return `模型 ${event.model}`;
    case "notice":
      return event.text;
  }
}

/** 终态事件 → 状态；过程事件返回 null。 */
export function terminalStatus(event: RunEvent): RunOutcome | null {
  switch (event.type) {
    case "completed":
      return "completed";
    case "stopped":
      return event.status;
    case "error":
      return "failed";
    default:
      return null;
  }
}

/**
 * error 事件的 message 能不能直接给用户看。
 *
 * 三种来源（api/server.py 的 _run_stream）：模型调用失败 / Agent 运行失败 /
 * 其余一切异常的兜底——兜底那条是 `异常类型: 消息`，直接摆出来就是把 Python
 * 的异常名和栈信息（可能带路径、甚至密钥）端到用户面前。规格要求不这样，
 * 所以只放行前两种（后端手写的中文说明），兜底压成一句白话；原文在 client.ts
 * 那一层已经写进 console 了。
 */
export function describeStreamError(message: string): string {
  const known = ["调用模型失败", "Agent 运行失败"];
  if (known.some((prefix) => message.startsWith(prefix))) return message;
  return "这一轮失败：后端抛了个没归类的错误（详情看后端日志和浏览器控制台）";
}

// ---------------------------------------------------------------------------
// 底下的零件
// ---------------------------------------------------------------------------

function applyEvent(run: RunRecord, event: RunEvent, seq: number, now: number): RunRecord {
  const at = { ...run, cursor: seq };

  switch (event.type) {
    case "start":
      // start 不长过程事件：任务在分组头上，run_id 补进记录（后端的真 id）
      return event.runId === "" ? at : { ...at, runId: event.runId };

    case "tool_start": {
      // read_file 的返回正文里没有路径，但它自己那个 path 参数就是位置
      // （只有路径、没有行号——它本来就没有行参数，不编）。结果回来时如果
      // 带位置，会把它换掉（见 mergeLocations）。
      const fromArguments = argsLocation(event.tool, event.arguments);
      return pushEvent({ ...at, toolCount: at.toolCount + 1 }, {
        kind: "tool",
        id: nextEventId(),
        tool: event.tool,
        family: toolFamily(event.tool),
        intent: toolIntent(event.tool, event.arguments),
        state: "running",
        durationSeconds: null,
        error: null,
        locations: fromArguments,
        locationCount: fromArguments.length,
        chain: isChainStart(event.tool, event.arguments),
        at: now,
      });
    }

    case "tool_result":
      return patchRunning(at, (record) => record.kind === "tool" && record.tool === event.tool, (record) =>
        record.kind === "tool"
          ? {
              ...record,
              state: event.success ? ("ok" as const) : ("fail" as const),
              durationSeconds: event.durationSeconds,
              error: event.error,
              ...mergeLocations(record, event.locations),
            }
          : record,
      ) ?? pushEvent(at, {
        // 没配对上（不该发生）也记一条结果：少一个「运行中」，好过把结果丢了
        kind: "tool",
        id: nextEventId(),
        tool: event.tool,
        family: toolFamily(event.tool),
        intent: null, // 没看到 tool_start，参数也就没有——不猜
        state: event.success ? "ok" : "fail",
        durationSeconds: event.durationSeconds,
        error: event.error,
        locations: event.locations.slice(0, MAX_EVENT_LOCATIONS),
        locationCount: event.locations.length,
        chain: false,
        at: now,
      });

    case "llm_start":
      return pushEvent({ ...at, llmCount: at.llmCount + 1 }, {
        kind: "llm",
        id: nextEventId(),
        model: event.model,
        attempt: event.attempt,
        maxAttempts: event.maxAttempts,
        state: "running",
        durationSeconds: null,
        totalTokens: null,
        error: null,
        at: now,
      });

    case "llm_end":
      return patchRunning(at, (record) => record.kind === "llm", (record) =>
        record.kind === "llm"
          ? {
              ...record,
              model: event.model === "" ? record.model : event.model,
              state: event.success ? ("ok" as const) : ("fail" as const),
              durationSeconds: event.durationSeconds,
              totalTokens: event.totalTokens,
              error: event.error,
            }
          : record,
      ) ?? pushEvent(at, {
        kind: "llm",
        id: nextEventId(),
        model: event.model,
        attempt: event.attemptCount,
        maxAttempts: event.attemptCount,
        state: event.success ? "ok" : "fail",
        durationSeconds: event.durationSeconds,
        totalTokens: event.totalTokens,
        error: event.error,
        at: now,
      });

    case "completed":
      return { ...at, phase: "completed", endedAt: now, answer: clampText(event.answer, MAX_ANSWER, TRUNCATED_ANSWER) };

    case "stopped":
      return { ...at, phase: event.status, endedAt: now, message: clampText(event.message, MAX_TEXT, TRUNCATED_TEXT) };

    case "error":
      return {
        ...at,
        phase: "failed",
        endedAt: now,
        message: clampText(describeStreamError(event.message), MAX_TEXT, TRUNCATED_TEXT),
      };
  }
}

/** 事件记到列表末尾；超过上限就丢最早的，并把 truncated 立起来。 */
function pushEvent(run: RunRecord, record: RunEventRecord): RunRecord {
  const events = [...run.events, record];
  const overflow = events.length > MAX_EVENTS;
  return {
    ...run,
    events: overflow ? events.slice(-MAX_EVENTS) : events,
    eventCount: run.eventCount + 1,
    truncated: run.truncated || overflow,
  };
}

/** 改最近一条还在跑、且满足条件的记录；没有就返回 null（调用方决定怎么办）。 */
function patchRunning(
  run: RunRecord,
  match: (record: RunEventRecord) => boolean,
  patch: (record: RunEventRecord) => RunEventRecord,
): RunRecord | null {
  for (let index = run.events.length - 1; index >= 0; index -= 1) {
    const record = run.events[index];
    if (record === undefined || record.kind === "notice") continue;
    if (record.state !== "running" || !match(record)) continue;

    const events = [...run.events];
    events[index] = patch(record);
    return { ...run, events };
  }
  return null;
}

/** 截断长文本。超了就砍掉并把话说明白——不假装记录是完整的。 */
function clampText(text: string, limit: number, suffix = ""): string {
  if (text.length <= limit) return text;
  return text.slice(0, limit) + suffix;
}

/**
 * 这次调用在看什么——**只认白名单里的那几个参数键**。
 *
 * 为什么不把参数摊开：后端给的 arguments 是截断过的摘要，但里面可能带着被
 * 读文件的内容（edit_file 的 new_text、run_command 的整条命令行、calculator
 * 的表达式）。那些进了记录就可能落进 localStorage。这里每个工具只取**一个
 * 指明对象**的键：找什么关键词、读哪个文件、看哪条调用链。
 *
 * 认不出来的工具返回 null：不显示「在干什么」，照样显示工具名和成败。
 */
export function toolIntent(tool: string, args: Record<string, string>): string | null {
  const one = (key: string, prefix: string) => {
    const value = arg(args, key);
    return value === null ? null : clampText(`${prefix}${value}`, MAX_INTENT);
  };

  switch (tool) {
    case "search_code":
      return one("keyword", "关键词 ");
    case "search_code_chunks":
      return one("query", "查询 ");
    case "read_file":
    case "edit_file":
    case "list_files":
    case "code_summary":
      return one("path", "");
    case "compile":
    case "build":
      return one("source", "");
    case "call_graph":
      return callGraphIntent(args);
    default:
      // run_command / run_tests / calculator 之类：参数里可能是整条命令、
      // 整段表达式，不取
      return null;
  }
}

function callGraphIntent(args: Record<string, string>): string | null {
  const name = arg(args, "function");
  if (name === null) return null;

  switch (args.direction) {
    case "callers":
      return clampText(`谁调用了 ${name}`, MAX_INTENT);
    case "callees":
      return clampText(`${name} 调用了谁`, MAX_INTENT);
    case "path": {
      const target = arg(args, "target");
      return clampText(target === null ? `${name} 的调用链` : `${name} → ${target} 的调用链`, MAX_INTENT);
    }
    default:
      return clampText(`调用关系 ${name}`, MAX_INTENT);
  }
}

/** 这一条的位置是不是一条**有序的链**（只有 call_graph 的 path 模式是）。 */
function isChainStart(tool: string, args: Record<string, string>): boolean {
  return tool === "call_graph" && args.direction === "path";
}

/**
 * 从参数里取一条「只有路径、没有行号」的位置。**只认 read_file**。
 *
 * 为什么只有它：它的返回正文是 `42|一行代码`，里面根本没有路径，不补一条
 * 位置的话这次调用在界面上就没有落点；而它的 path 参数正好就是那个落点。
 * 别的工具要么结果里自带位置（search / call_graph / compile / edit_file），
 * 要么参数指的是目录（list_files），要么参数根本不是位置（run_command）。
 *
 * 值被后端截断过（以 … 结尾）就整条丢掉：那种路径既显示不全也没法复制，
 * 摆出来只会让人以为文件名真的叫「…」。
 */
function argsLocation(tool: string, args: Record<string, string>): ToolLocation[] {
  if (tool !== "read_file") return [];
  const path = arg(args, "path");
  if (path === null) return [];
  return [{ path, line: 0, endLine: null, symbol: null, kind: null }];
}

/** 取一个参数值：没有 / 空的 / 被截断的（以 … 结尾）都当没有。 */
function arg(args: Record<string, string>, key: string): string | null {
  const value = args[key];
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (trimmed === "" || trimmed.endsWith("…")) return null;
  return trimmed;
}

/**
 * 结果回来时换上新位置。规则一句话：**结果里有就用结果的，没有就留着原来的**。
 *
 * 结果里的位置是后端从返回正文里提出来的，比参数推出来的准（比如 read_file
 * 这次要是没读到东西，正文里什么都没有，但参数推的那条路径仍然是对的，留着）。
 * 结果里真带了位置，条数按 locationCount 记全（locations 只存前 12 条）。
 */
function mergeLocations(
  record: ToolRecord,
  incoming: ToolLocation[],
): Pick<ToolRecord, "locations" | "locationCount"> {
  if (incoming.length === 0) {
    return { locations: record.locations, locationCount: record.locationCount };
  }
  return {
    locations: incoming.slice(0, MAX_EVENT_LOCATIONS),
    locationCount: incoming.length,
  };
}

/** RunStatus 里的那些取值，这里也认（后端响应里可能直接给）。 */
export function asRunPhase(value: RunStatus | null | undefined): RunPhase | null {
  switch (value) {
    case "running":
    case "completed":
    case "cancelled":
    case "timed_out":
    case "failed":
      return value;
    default:
      return null;
  }
}
