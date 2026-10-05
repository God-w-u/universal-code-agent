/**
 * 后端 API 的调用：**整个前端只有这一个文件碰网络**。
 *
 * 五条路由（api/server.py 的 _ROUTES），这里一一对应，一个字段都不改：
 *
 *   GET    /api/health          checkHealth()
 *   POST   /api/run             runTask()      —— 普通入口（跑完一次性拿到）
 *   POST   /api/run/stream      streamRun()    —— SSE，默认走这条
 *   POST   /api/runs/cancel     cancelRun()
 *   DELETE /api/sessions        deleteSession()
 *
 * **地址**：`VITE_API_BASE_URL`（见 .env.example）。不写 = 用后端默认的
 * http://127.0.0.1:8000；写成空串 = 同源相对路径（/api/...）。
 *
 * **浏览器前端一定要配后端的 API_CORS_ORIGINS**：浏览器发 POST 时必然带
 * Origin 头，而后端的设计是「带了 Origin 就必须在允许列表里」（api/server.py
 * 的 _origin_allowed），同源也一样挡——这正是它防「别的网页对本机发请求」的
 * 办法。所以这不是「跨源才要配」，是「用浏览器就要配」；配的是**页面地址**
 * （例如 http://localhost:5173），改完重启后端。
 *
 * **错误**：一律抛 ApiError，message 是**给用户看的中文**。后端 answer 里的
 * 原文只在 400 / 404 这类「说的是这次请求哪儿不对」时直接采用；5xx 的 answer
 * 里可能有 Python 异常的原文（api/server.py 的兜底就是 `类型: 消息`），压成
 * 一句中文，原文写进 console 供排查——页面上不摆异常栈。
 */

import { createFrameDecoder, toRunEvent } from "./sse";
import type { CancelResult, RunEvent, RunResult, RunStatus } from "./types";

/** python -m api 的默认地址（api/server.py 的 DEFAULT_HOST / DEFAULT_PORT）。 */
const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

// 「没配」和「配成空串」是两回事：前者用后端的默认地址，后者是要同源。
// import.meta.env 里没写就是 undefined，写了空值就是 ""。
const RAW_BASE = import.meta.env.VITE_API_BASE_URL as string | undefined;

/** 后端地址；空串 = 同源（请求发到本页面的 /api/...）。 */
export const API_BASE_URL = (RAW_BASE === undefined ? DEFAULT_BASE_URL : RAW_BASE)
  .trim()
  .replace(/\/+$/, "");

/** 给界面显示用的说法。 */
export const API_LABEL = API_BASE_URL === "" ? "同源 /api" : API_BASE_URL;

/** 一次调用失败。status 为 null 表示根本没连上（网络层就没成）。 */
export class ApiError extends Error {
  readonly status: number | null;

  constructor(status: number | null, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function isApiError(cause: unknown): cause is ApiError {
  return cause instanceof ApiError;
}

/** 用户主动中断（组件卸载 / 关页面）不是错误，调用方自己判断。 */
export function isAbort(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

// ---------------------------------------------------------------------------
// 五个接口
// ---------------------------------------------------------------------------

/**
 * 探活结果的三种情况。分开是有用的：403 说明后端**活着**，只是没把本页面
 * 加进 API_CORS_ORIGINS——用户该去改后端配置，而不是去查它为什么没启动。
 */
export type HealthState = "ok" | "blocked" | "down";

/** GET /api/health：这条不需要 Token（后端特意开放的）。 */
export async function checkHealth(signal?: AbortSignal): Promise<HealthState> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, { signal });
    // 来源检查在路由分发之前跑，所以探活也会吃 403——正好用来区分「没起来」
    if (response.status === 403) return "blocked";
    if (!response.ok) return "down";
    const body = await readJson(response);
    return body.status === "ok" ? "ok" : "down";
  } catch {
    // 连不上。不往上抛——状态条要显示「连不上」，而不是弹一个错误
    return "down";
  }
}

export interface StreamRunOptions {
  task: string;
  /** null = 不带 session_id，让后端新建一个会话。 */
  sessionId: string | null;
  token: string | null;
  signal: AbortSignal;
  onEvent: (event: RunEvent) => void;
}

/**
 * POST /api/run/stream：把这一轮跑的过程边跑边读。
 *
 * 函数在**流结束时**返回（正常结束或被中断），不返回结果——结果在事件里
 * （completed 带最终回答）。中途断开不抛错：让调用方看「有没有收到终态」
 * 来判断，比在这儿猜更准（断线和被取消在 HTTP 层都是「流没了」）。
 */
export async function streamRun(options: StreamRunOptions): Promise<void> {
  const response = await send(
    "/api/run/stream",
    bodyFor(options.task, options.sessionId),
    options.token,
    options.signal,
  );

  if (!response.ok) {
    throw await failure(response);
  }
  if (response.body === null) {
    throw new ApiError(null, "浏览器没给出响应流，这次执行的过程读不到（换一个浏览器试试）");
  }
  const contentType = response.headers.get("Content-Type") ?? "";
  if (!contentType.includes("text/event-stream")) {
    // 走到这儿通常是中间有代理把流改写了；当成「这次没结果」比解析 HTML 强
    throw new ApiError(null, "后端没有按事件流回应（可能被代理改写了），这次执行拿不到结果");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const push = createFrameDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // stream: true —— 一个汉字被切在两块之间时，下一块接着拼
    dispatch(push(decoder.decode(value, { stream: true })), options.onEvent);
  }
  // 收尾：解码器里可能还压着最后半个字符，冲出来再解一遍
  dispatch(push(decoder.decode()), options.onEvent);
}

/**
 * POST /api/run：普通模式，跑完一次性拿结果。
 *
 * 前端默认走流式；这条路是**兜底**：浏览器没有 fetch 流（ReadableStream）时
 * 才用（见 supportsStreaming）。那时这一轮的过程看不到，只有最终回答。
 */
export async function runTask(
  task: string,
  sessionId: string | null,
  token: string | null,
  signal: AbortSignal,
): Promise<RunResult> {
  const response = await send(
    "/api/run",
    bodyFor(task, sessionId),
    token,
    signal,
  );
  if (!response.ok) {
    throw await failure(response);
  }
  const body = await readJson(response);
  return {
    success: body.success === true,
    answer: typeof body.answer === "string" ? body.answer : "",
    sessionId: typeof body.session_id === "string" ? body.session_id : "",
    runId: typeof body.run_id === "string" ? body.run_id : "",
    // 被取消 / 超时才有 status；成功那条响应里没有这个字段
    status: asRunStatus(body.status),
  };
}

/**
 * POST /api/runs/cancel：把正在跑的那一轮停下来。
 *
 * run_id 用**流事件里发的那个真 id**（start 事件），前端不自己编。后端认
 * (session_id, run_id) 这一对；已经结束的 Run 再取消是幂等的（cancelled=false）。
 */
export async function cancelRun(
  sessionId: string,
  runId: string | null,
  token: string | null,
): Promise<CancelResult> {
  const payload: Record<string, string> = { session_id: sessionId };
  if (runId !== null) payload.run_id = runId;

  const response = await send("/api/runs/cancel", payload, token);
  if (!response.ok) {
    throw await failure(response);
  }
  const body = await readJson(response);
  return {
    cancelled: body.cancelled === true,
    runId: typeof body.run_id === "string" ? body.run_id : "",
    status: asRunStatus(body.status) ?? "running",
  };
}

/** DELETE /api/sessions：删掉这个会话（后端那本会话历史）。 */
export async function deleteSession(
  sessionId: string,
  token: string | null,
): Promise<void> {
  const response = await send(
    "/api/sessions",
    { session_id: sessionId },
    token,
    undefined,
    "DELETE",
  );
  if (!response.ok) {
    throw await failure(response);
  }
}

/** 浏览器支不支持「边收边读」的响应流。不支持就只能走普通模式。 */
export function supportsStreaming(): boolean {
  return (
    typeof ReadableStream !== "undefined" &&
    typeof TextDecoder !== "undefined" &&
    typeof Response !== "undefined" &&
    "body" in Response.prototype
  );
}

// ---------------------------------------------------------------------------
// 底下这些不关心是哪条路由
// ---------------------------------------------------------------------------

/** 能放进 HTTP 头的可打印 ASCII（见 send() 里那个检查）。 */
const ASCII_ONLY = /^[\x20-\x7e]+$/;

function bodyFor(task: string, sessionId: string | null): Record<string, string> {
  // 不给 session_id = 让后端新建一个会话（后端的行为，见 api/server.py）
  return sessionId === null ? { task } : { task, session_id: sessionId };
}

/** 发一个带 JSON 体的请求；连不上这一步就抛 ApiError。 */
async function send(
  path: string,
  payload: unknown,
  token: string | null,
  signal?: AbortSignal,
  method: "POST" | "DELETE" = "POST",
): Promise<Response> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  // Token 只存在这个页面里（内存），不落 localStorage、不进源码
  const value = token?.trim() ?? "";
  if (value !== "") {
    if (!ASCII_ONLY.test(value)) {
      // HTTP 头只能放 ASCII。不先拦下的话，fetch 会自己抛一个 TypeError，
      // 被下面兜成「连不上后端」——把复制粘贴的问题说成后端没起来，最误事
      throw new ApiError(
        null,
        "Token 里有非 ASCII 字符（HTTP 头只能放 ASCII）：检查是不是复制进了中文或全角空格",
      );
    }
    headers.Authorization = `Bearer ${value}`;
  }

  try {
    return await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: JSON.stringify(payload),
      signal,
    });
  } catch (cause) {
    if (isAbort(cause)) throw cause; // 自己按的停，不是错
    throw new ApiError(
      null,
      `连不上后端（${API_LABEL}）：确认它已经在跑（python -m api），地址和端口没填错`,
    );
  }
}

/** 非 2xx：读一下 answer，拼成一句能给用户看的话。 */
async function failure(response: Response): Promise<ApiError> {
  const body = await readJson(response);
  const answer = typeof body.answer === "string" ? body.answer : "";
  // 后端自己说的话留在 console 里，排查用；页面上只给那句中文
  if (answer !== "") {
    console.warn(`[api] ${response.status} ${response.url}：${answer}`);
  }
  return new ApiError(response.status, messageFor(response.status, answer));
}

/** 状态码 → 给用户看的中文。 */
function messageFor(status: number, answer: string): string {
  switch (status) {
    case 400:
      // 参数不对这一类，后端的话就是给调用方看的（「task 不能是空的」）
      return answer !== "" ? answer : "请求被后端拒绝了（400）：检查任务内容";
    case 401:
      return "后端要 Token：请在左下角填 API Token（只存在这个页面里，不保存）";
    case 403:
      // 把「要加哪一行」说死，用户直接复制。window 的判断是为了 node 下的单测
      // （这个文件在浏览器里跑，但测试环境没有 window）
      return typeof window === "undefined"
        ? "后端不认这个来源（403）：把本页面的地址加进后端的 API_CORS_ORIGINS"
        : `后端不认这个来源（403）：把 ${window.location.origin} 加进后端的 API_CORS_ORIGINS（改完重启后端）`;
    case 404:
      return answer !== "" ? answer : "没有这个会话或接口（404）";
    case 405:
      return "请求方法不对（405）";
    case 413:
      return "任务太长了，超过后端 1 MiB 的请求体上限（413）";
    case 415:
      return "请求头不对（415）：后端要求 Content-Type: application/json";
    case 500:
      return "后端处理这一轮时出错了（500），详情在后端的日志里";
    case 502:
      return "模型调用失败（502）：检查 API Key、网络和 OPENAI_BASE_URL 后重试";
    default:
      return `请求失败（HTTP ${status}）`;
  }
}

/** 读响应体；不是 JSON 就当空对象（后端保证是 JSON，但代理插一脚就未必）。 */
async function readJson(response: Response): Promise<Record<string, unknown>> {
  try {
    const parsed: unknown = await response.json();
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

function dispatch(frames: { event: string; data: string }[], onEvent: (e: RunEvent) => void): void {
  for (const frame of frames) {
    const event = toRunEvent(frame);
    if (event !== null) onEvent(event);
  }
}

function asRunStatus(value: unknown): RunStatus | null {
  const allowed: RunStatus[] = [
    "idle",
    "running",
    "completed",
    "cancelled",
    "timed_out",
    "failed",
  ];
  return typeof value === "string" && (allowed as string[]).includes(value)
    ? (value as RunStatus)
    : null;
}
