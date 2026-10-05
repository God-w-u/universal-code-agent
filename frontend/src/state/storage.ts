/**
 * transcript 的落盘：一个会话一个键。
 *
 * **为什么一个会话一个键**（而不是一个大 JSON 全装下）：坏一份只坏一份，
 * 删会话就是删一个键，写也只写没变的那部分。localStorage 本来就是「可能被
 * 别的东西写坏、可能写不进去」的地方（隐私模式、盘满、用户手改），按会话
 * 拆开之后，这些事最多影响一个会话。
 *
 * 存进去的**只有** transcript（任务、工具名、状态、时长、最终回答这些）。
 * Token 不在这个模块的类型里，也就不可能被写进去——那是 App 里的一份内存
 * 状态（见 README）。工具的调用参数也不存：后端虽然截断过，但那里面可能带
 * 着被读文件的内容（见 transcript.ts 的说明）。
 *
 * 读的时候一律当作「可能是坏的」：JSON 解析失败、版本对不上、字段类型不对
 * ——统统当没存过，返回空。**读出来的每一份都要过 markInterrupted**：刷新
 * 之后没有任何一条流还活着，界面上绝不能还显示「运行中」。
 */

import { MAX_EVENT_LOCATIONS, TRANSCRIPT_VERSION, markInterrupted, toolFamily } from "./transcript";
import type { RunEventRecord, RunPhase, RunRecord, ToolFamily, Transcript } from "./transcript";
import type { ToolLocation } from "../api/types";

const KEY_PREFIX = "uac.frontend.transcript.v1.";

/** 最多留几个会话的记录（超了按 updatedAt 丢最老的）。 */
export const MAX_STORED = 30;

/** 写入结果：ok = 写进去了；trimmed = 清掉旧的后才写进去；failed = 没写进去。 */
export type SaveResult = "ok" | "trimmed" | "failed";

function storageKey(sessionKey: string): string {
  return `${KEY_PREFIX}${sessionKey}`;
}

/** 有没有可用的 localStorage（没有就整个模块当哑巴，不该因此崩页面）。 */
function store(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null; // 有的浏览器在隐私模式下访问 localStorage 就抛
  }
}

// ---------------------------------------------------------------------------
// 读
// ---------------------------------------------------------------------------

/**
 * 把存着的所有会话记录读进来（进页面时调一次）。
 *
 * 返回的 map 里键是会话 key；读不出来的键直接不出现。每一份都过了
 * markInterrupted —— 见文件头上的说明。
 */
export function loadAllTranscripts(now = Date.now()): Record<string, Transcript> {
  const box = store();
  if (box === null) return {};

  const result: Record<string, Transcript> = {};
  for (const storage of listKeys(box)) {
    const sessionKey = storage.slice(KEY_PREFIX.length);
    const transcript = readKey(box, storage, now);
    if (transcript !== null) result[sessionKey] = transcript;
  }
  return result;
}

/** 读一个会话的记录；没有 / 坏了就是 null。 */
export function readTranscript(sessionKey: string, now = Date.now()): Transcript | null {
  const box = store();
  if (box === null) return null;
  return readKey(box, storageKey(sessionKey), now);
}

function readKey(box: Storage, storage: string, now: number): Transcript | null {
  let text: string | null = null;
  try {
    text = box.getItem(storage);
  } catch {
    return null;
  }
  if (text === null) return null;

  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    // 手改过 / 别的程序写坏的：当没存过。**不删**——万一用户还想自己捞回来
    return null;
  }

  const parsed = parseTranscript(raw);
  if (parsed === null) return null;

  const healed = markInterrupted(parsed, now);
  if (healed !== parsed) {
    // 刚标成「中断」的这份顺手写回去：不然盘上一直写着「运行中」，每次进页面
    // 都得再判一次，而且别的工具来看这份数据也会被那个假的 Running 骗到
    try {
      box.setItem(storage, JSON.stringify(healed));
    } catch {
      /* 写不进去就只在内存里是中断的，页面照样对 */
    }
  }
  return healed;
}

/** 存着记录的那些会话 key（顺序不保证）。 */
export function listStoredSessions(): string[] {
  const box = store();
  return box === null ? [] : listKeys(box).map((key) => key.slice(KEY_PREFIX.length));
}

function listKeys(box: Storage): string[] {
  const keys: string[] = [];
  try {
    for (let index = 0; index < box.length; index += 1) {
      const key = box.key(index);
      if (key !== null && key.startsWith(KEY_PREFIX)) keys.push(key);
    }
  } catch {
    return [];
  }
  return keys;
}

// ---------------------------------------------------------------------------
// 写
// ---------------------------------------------------------------------------

/**
 * 存一份记录。写不进去时**先清旧的、再砍自己的**，都试过还不行就返回 failed
 * ——调用方拿这个结果去提示用户（页面上说一句「本地存不下了」），而不是悄悄
 * 丢掉。localStorage 只是便利，失败不影响这一轮本身。
 */
export function saveTranscript(sessionKey: string, transcript: Transcript): SaveResult {
  const box = store();
  if (box === null) return "failed";

  let current = transcript;
  let trimmed = false;

  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      box.setItem(storageKey(sessionKey), JSON.stringify(current));
      return trimmed ? "trimmed" : "ok";
    } catch (cause) {
      if (!isQuotaError(cause)) return "failed";
      trimmed = true;

      // 先清别的会话里最老的（不动自己）
      if (evictOldest(MAX_STORED - 1, sessionKey) > 0) continue;

      // 再砍自己最老的几轮；只剩一轮时没得砍了
      if (current.runs.length <= 1) return "failed";
      current = { ...current, runs: current.runs.slice(Math.ceil(current.runs.length / 2)) };
    }
  }
  return "failed";
}

/** 删一个会话的记录（删会话时一起删）。 */
export function removeTranscript(sessionKey: string): void {
  const box = store();
  if (box === null) return;
  try {
    box.removeItem(storageKey(sessionKey));
  } catch {
    /* 删不掉也没什么可做的 */
  }
}

/** 全部删掉（「清空本机记录」用）。 */
export function removeAllTranscripts(): void {
  const box = store();
  if (box === null) return;
  for (const key of listKeys(box)) {
    try {
      box.removeItem(key);
    } catch {
      /* 同上 */
    }
  }
}

/**
 * 只留最近 `keep` 个会话的记录，其余的删掉。返回删了几个。
 * `except` 是「这个别删」——正在写的那个会话总得留着。
 */
function evictOldest(keep: number, except: string): number {
  const box = store();
  if (box === null) return 0;

  const entries: { sessionKey: string; updatedAt: number }[] = [];
  for (const key of listKeys(box)) {
    const sessionKey = key.slice(KEY_PREFIX.length);
    entries.push({ sessionKey, updatedAt: updatedAtOf(box, key) });
  }
  if (entries.length <= keep) return 0;

  // 最老的排前面；要删够 entries.length - keep 个
  entries.sort((a, b) => a.updatedAt - b.updatedAt);
  let removed = 0;
  for (const entry of entries) {
    if (entries.length - removed <= keep) break;
    if (entry.sessionKey === except) continue;
    try {
      box.removeItem(storageKey(entry.sessionKey));
      removed += 1;
    } catch {
      /* 删不掉就留着 */
    }
  }
  return removed;
}

/** 取一份记录的 updatedAt；读不出来当 0（最老，优先被清）。 */
function updatedAtOf(box: Storage, storage: string): number {
  try {
    const raw: unknown = JSON.parse(box.getItem(storage) ?? "null");
    if (typeof raw === "object" && raw !== null) {
      const value = (raw as Record<string, unknown>).updatedAt;
      if (typeof value === "number") return value;
    }
  } catch {
    /* 坏的当最老 */
  }
  return 0;
}

function isQuotaError(cause: unknown): boolean {
  if (!(cause instanceof Error)) return false;
  // 各家的名字 / 码不完全一样，宽一点无妨：反正下一步是「清旧的再试」
  return (
    cause.name === "QuotaExceededError" ||
    cause.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    (cause as { code?: unknown }).code === 22
  );
}

// ---------------------------------------------------------------------------
// 解析：存着的东西一律当坏的
// ---------------------------------------------------------------------------

function parseTranscript(raw: unknown): Transcript | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const record = raw as Record<string, unknown>;

  // 版本对不上：这份记录是别的版本写的，宁可当空——瞎猜字段只会更糟
  if (record.version !== TRANSCRIPT_VERSION) return null;
  if (!Array.isArray(record.runs)) return null;

  const runs: RunRecord[] = [];
  for (const item of record.runs) {
    const run = parseRun(item);
    if (run !== null) runs.push(run);
  }

  return {
    version: TRANSCRIPT_VERSION,
    runs,
    updatedAt: typeof record.updatedAt === "number" ? record.updatedAt : 0,
  };
}

const PHASES: RunPhase[] = ["running", "completed", "cancelled", "timed_out", "failed", "interrupted"];

function parseRun(raw: unknown): RunRecord | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const record = raw as Record<string, unknown>;

  const key = str(record.key);
  const task = str(record.task);
  if (key === null || task === null) return null; // 没有身份的一轮没法显示

  const events: RunEventRecord[] = [];
  if (Array.isArray(record.events)) {
    for (const item of record.events) {
      const event = parseEvent(item);
      if (event !== null) events.push(event);
    }
  }

  return {
    key,
    runId: str(record.runId),
    task,
    startedAt: num(record.startedAt),
    endedAt: record.endedAt === null ? null : num(record.endedAt),
    phase: PHASES.includes(record.phase as RunPhase) ? (record.phase as RunPhase) : "failed",
    answer: str(record.answer),
    message: str(record.message),
    events,
    toolCount: count(record.toolCount, events, "tool"),
    llmCount: count(record.llmCount, events, "llm"),
    eventCount: count(record.eventCount, events, null),
    cursor: num(record.cursor),
    truncated: record.truncated === true,
  };
}

function parseEvent(raw: unknown): RunEventRecord | null {
  if (typeof raw !== "object" || raw === null) return null;
  const record = raw as Record<string, unknown>;
  const id = str(record.id);
  if (id === null) return null;

  const at = num(record.at);
  const state = record.state;
  const runState = state === "ok" || state === "fail" || state === "running" ? state : "fail";

  if (record.kind === "tool") {
    const tool = str(record.tool);
    if (tool === null) return null;
    // family / intent / locations 都是 14.3 才有的：老记录里没有，认不出来
    // 就当没有——family 现算一个出来（工具名没变，算得出来），位置就算了
    const locations = parseLocations(record.locations);
    return {
      kind: "tool",
      id,
      tool,
      family: familyOf(record.family, tool),
      intent: str(record.intent),
      state: runState,
      durationSeconds: record.durationSeconds === null ? null : num(record.durationSeconds),
      error: str(record.error),
      locations,
      locationCount: countLocations(record.locationCount, locations),
      chain: record.chain === true,
      at,
    };
  }

  if (record.kind === "llm") {
    return {
      kind: "llm",
      id,
      model: str(record.model) ?? "",
      attempt: num(record.attempt),
      maxAttempts: num(record.maxAttempts),
      state: runState,
      durationSeconds: record.durationSeconds === null ? null : num(record.durationSeconds),
      totalTokens: record.totalTokens === null ? null : num(record.totalTokens),
      error: str(record.error),
      at,
    };
  }

  if (record.kind === "notice") {
    const text = str(record.text);
    if (text === null) return null;
    const level = record.level;
    return {
      kind: "notice",
      id,
      level: level === "info" || level === "warn" || level === "error" ? level : "info",
      text,
      at,
    };
  }

  return null; // 认不出来的记录丢掉，不猜
}

/**
 * 位置列表：存坏了 / 缺字段的那几条丢掉，行号不是正经数字就当 0。
 *
 * 一条位置最少要有路径（和后端那层的规矩一致，见 api/sse.ts 的 locationList）；
 * 参数形状不对就当这条没有。**不按别的字段倒推**。
 */
function parseLocations(value: unknown): ToolLocation[] {
  if (!Array.isArray(value)) return [];

  const result: ToolLocation[] = [];
  for (const item of value) {
    if (result.length >= MAX_EVENT_LOCATIONS) break;
    if (typeof item !== "object" || item === null) continue;
    const record = item as Record<string, unknown>;

    const path = str(record.path);
    if (path === null || path === "") continue;

    const line = num(record.line);
    result.push({
      path,
      line: line >= 1 ? line : 0,
      endLine: record.endLine === null ? null : num(record.endLine) || null,
      symbol: str(record.symbol),
      kind: str(record.kind),
    });
  }
  return result;
}

/** 条数：存着的不算数（可能被截过），字段本身可信就用字段的。 */
function countLocations(value: unknown, locations: ToolLocation[]): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= locations.length) {
    return value;
  }
  return locations.length;
}

/** 老记录没有 family：工具名还在，现算一个（表在 transcript.ts 里，一处维护）。 */
function familyOf(value: unknown, tool: string): ToolFamily {
  const families: ToolFamily[] = ["search", "read", "edit", "graph", "run", "build", "compute", "other"];
  return families.includes(value as ToolFamily) ? (value as ToolFamily) : toolFamily(tool);
}

function str(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** 计数缺失（老记录 / 被手改）时从保留的事件里数一个出来。 */
function count(value: unknown, events: RunEventRecord[], kind: "tool" | "llm" | null): number {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) return value;
  if (kind === null) return events.length;
  return events.filter((event) => event.kind === kind).length;
}
