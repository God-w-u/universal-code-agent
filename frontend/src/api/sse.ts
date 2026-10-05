/**
 * SSE 的解析：纯函数，不碰网络、不碰 React。
 *
 * **为什么不用浏览器的 EventSource**：它只能发 GET、不能带请求头，而后端的
 * `/api/run/stream` 是 POST + 可选 `Authorization: Bearer <token>`（api/server.py）。
 * 两条都做不到，所以走 fetch + ReadableStream 自己读流（那一段在 client.ts），
 * 解析就落在这儿。
 *
 * 分两层，各管各的：
 * - createFrameDecoder()：文本块 → 一帧一帧。要处理「一帧被切成两半」
 *   （网络分片），所以它带着跨块的缓冲；
 * - toRunEvent()：一帧 → 这个前端认识的事件，不认识的返回 null。
 *
 * 两层都是纯函数，测试里不用起服务器、不用真网络（见 sse.test.ts）。
 */

import type { RunEvent, ToolLocation } from "./types";

/**
 * 一次调用最多认这么多条位置。后端自己封顶 50（tools/locations.py 的
 * MAX_LOCATIONS），这里是**不信任对面**的那一层：万一以后后端忘了封，
 * 前端也不该被一帧塞进一万条位置。
 */
export const MAX_LOCATIONS = 50;
/** 路径 / 符号名 / 类型各自的长度上限：超了截图，别让一条位置撑破版面。 */
export const MAX_LOCATION_PATH = 200;
export const MAX_LOCATION_SYMBOL = 120;
export const MAX_LOCATION_KIND = 32;

/** 服务端发来的一帧：事件名 + data 原文（还没解析 JSON）。 */
export interface SseFrame {
  event: string;
  data: string;
}

/**
 * 造一个「喂文本块、吐完整帧」的解码器。
 *
 * 返回的函数是可以连着调的：上一块剩下的半个帧留在闭包里，下一块接着拼。
 * 流结束时就别再调了——残留的半帧按 SSE 的规矩就是丢弃（服务端每帧都写完整
 * 才 flush，读到半个帧说明连接是断的）。
 */
export function createFrameDecoder(): (chunk: string) => SseFrame[] {
  let buffer = "";

  return function push(chunk: string): SseFrame[] {
    // 换行统一成 \n。先吃成对的 \r\n，末尾落单的 \r 先留着——它可能是被
    // 切开的 \r\n 的前半个，下一块到了再判
    buffer += chunk.replace(/\r\n/g, "\n");
    const pendingCr = buffer.endsWith("\r");
    if (pendingCr) buffer = buffer.slice(0, -1);
    buffer = buffer.replace(/\r/g, "\n");
    if (pendingCr) buffer += "\r";

    const frames: SseFrame[] = [];
    // 空行 = 一帧结束。归一过之后分隔符只剩 \n\n 一种
    let end = buffer.indexOf("\n\n");
    while (end >= 0) {
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const frame = parseFrame(raw);
      if (frame !== null) frames.push(frame);
      end = buffer.indexOf("\n\n");
    }
    return frames;
  };
}

/**
 * 一帧的原文 → {event, data}；没有 data 的帧（纯注释 / 心跳）返回 null。
 *
 * 按 SSE 的规矩来：`字段: 值`，冒号后的一个空格是分隔用的、不算值；`data`
 * 出现多次就按行拼起来；认不出的字段（id / retry）忽略——不认识的东西不该
 * 让整条流读不下去。
 */
export function parseFrame(raw: string): SseFrame | null {
  let event = "";
  const data: string[] = [];

  for (const line of raw.split("\n")) {
    if (line === "" || line.startsWith(":")) continue; // 空行 / 注释

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }

  if (data.length === 0) return null;
  // 没写 event 的按 SSE 的默认事件名算，虽然后端每帧都写
  return { event: event || "message", data: data.join("\n") };
}

/**
 * 一帧 → 一条事件。三种情况返回 null（都当作「没有这条事件」，不打断流）：
 * - data 不是 JSON 对象；
 * - 事件名不认识（以后后端加了新事件，老前端不该崩）；
 * - 事件名认识但字段对不上（缺字段的按空值处理，不猜）。
 */
export function toRunEvent(frame: SseFrame): RunEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(frame.data);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const data = parsed;

  switch (frame.event) {
    case "start":
      return {
        type: "start",
        task: text(data.task),
        sessionId: text(data.session_id),
        runId: text(data.run_id),
      };

    case "tool_start":
      // arguments 是后端截断过的摘要对象（值已经转成字符串，见 api/sse.py 的
      // summarize），不是工具的原始参数——这里只拿来显示
      return {
        type: "tool_start",
        tool: text(data.tool),
        arguments: stringMap(data.arguments),
      };

    case "tool_result":
      return {
        type: "tool_result",
        tool: text(data.tool),
        success: data.success === true,
        durationSeconds: number(data.duration_seconds),
        error: optionalText(data.error),
        // 老后端（14.3 之前）不发这个键：没有就是空数组，不是错误
        locations: locationList(data.locations),
      };

    case "llm_start":
      return {
        type: "llm_start",
        model: text(data.model),
        attempt: number(data.attempt),
        maxAttempts: number(data.max_attempts),
      };

    case "llm_end":
      return {
        type: "llm_end",
        model: text(data.model),
        success: data.success === true,
        durationSeconds: number(data.duration_seconds),
        attemptCount: number(data.attempt_count),
        totalTokens: totalTokens(data.usage),
        error: optionalText(data.error),
      };

    case "completed":
      // 最终回答就在这条上：不用再发一次 /api/run 去要
      return {
        type: "completed",
        answer: text(data.answer),
        sessionId: text(data.session_id),
        runId: text(data.run_id),
      };

    case "cancelled":
    case "timed_out":
      return {
        type: "stopped",
        status: frame.event === "cancelled" ? "cancelled" : "timed_out",
        message: text(data.message),
        sessionId: text(data.session_id),
        runId: text(data.run_id),
      };

    case "error":
      return {
        type: "error",
        message: text(data.message),
        sessionId: text(data.session_id),
        runId: text(data.run_id),
      };

    default:
      return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** 取字符串字段；不是字符串就当空串（宁可少显示，也别把 "[object]" 摆上去）。 */
function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

/** 同上，但空串归成 null——「没有」和「空的」在显示上是一回事。 */
function optionalText(value: unknown): string | null {
  const result = text(value);
  return result === "" ? null : result;
}

/** 取数字字段；不是有限数字就是 0（时间、次数这些，0 比 NaN 好显示）。 */
function number(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** 参数摘要：值按字符串收（后端已经都转成字符串了），非字符串的再兜一层。 */
function stringMap(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {};
  const result: Record<string, string> = {};
  for (const [key, item] of Object.entries(value)) {
    result[key] = typeof item === "string" ? item : String(item ?? "");
  }
  return result;
}

/** usage 只取总 token 数：接口没给就是 null（不编数据）。 */
function totalTokens(value: unknown): number | null {
  if (!isRecord(value)) return null;
  const total = value.total_tokens;
  return typeof total === "number" && Number.isFinite(total) ? total : null;
}

/**
 * 位置列表：**认不出来的整条丢掉**，不猜、不倒推。
 *
 * 一条位置最小要有一份路径——后端那条白名单保证了这一点；路径字段丢了 /
 * 不是字符串 / 是空串的，说明这条本来就立不住，丢掉比编一个强。行号不是
 * 正经数字就当 0（「只知道文件，不知道行号」，见 ToolLocation 的说明）。
 */
function locationList(value: unknown): ToolLocation[] {
  if (!Array.isArray(value)) return [];

  const result: ToolLocation[] = [];
  for (const item of value) {
    if (result.length >= MAX_LOCATIONS) break;
    const location = oneLocation(item);
    if (location !== null) result.push(location);
  }
  return result;
}

function oneLocation(value: unknown): ToolLocation | null {
  if (!isRecord(value)) return null;
  const path = clamp(text(value.path), MAX_LOCATION_PATH);
  if (path === "") return null;

  // 先把行号归一（不是正经行号就是 0 = 没有行号），再拿它去量结束行——
  // 拿原始值去比会把 `line: -3` 那种错数据算出一个合法的结束行来
  const rawLine = number(value.line);
  const line = rawLine >= 1 ? Math.floor(rawLine) : 0;

  return {
    path,
    line,
    endLine: line === 0 ? null : endLine(value.end_line, line),
    symbol: optionalClamp(text(value.symbol), MAX_LOCATION_SYMBOL),
    kind: optionalClamp(text(value.kind), MAX_LOCATION_KIND),
  };
}

/** 结束行：不成立（不是数字 / 不大于起始行）就是 null——单行，不是范围。 */
function endLine(value: unknown, line: number): number | null {
  const end = number(value);
  return end > line ? Math.floor(end) : null;
}

/** 截断到上限；截了就带个省略号，让人知道它不完整。 */
function clamp(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
}

function optionalClamp(value: string, limit: number): string | null {
  const clamped = clamp(value, limit);
  return clamped === "" ? null : clamped;
}
