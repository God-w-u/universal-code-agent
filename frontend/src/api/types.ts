/**
 * 前端看到的「后端说的话」：SSE 事件和响应体。
 *
 * 这些形状是**后端定的**（api/sse.py 的 frame() + api/server.py 的发帧处），
 * 前端只做翻译，不改协议：字段名（session_id / run_id / duration_seconds）
 * 保持后端的蛇形，转换只在这里做一次（见 sse.ts 的 toRunEvent）。
 */

/** 一轮 Run 的状态。前四个来自后端，idle 是「这个页面还没有正在跑的」。 */
export type RunStatus =
  | "idle"
  | "running"
  | "completed"
  | "cancelled"
  | "timed_out"
  | "failed";

/**
 * 一条代码位置：这次工具调用落在哪个文件的哪一段（14.3）。
 *
 * 后端只从**认得输出格式**的那几个工具里提（tools/locations.py 的白名单），
 * 而且只提位置——命中的那行代码、诊断消息这些正文一个字都不带。所以这里能
 * 显示的也只有「哪个文件第几行、是什么符号」。
 *
 * `line` 为 0 表示**只知道文件、不知道行号**（比如后端给的位置行格式变了）。
 * 界面上遇到 0 就只显示路径——不编一个行号出来。
 */
export interface ToolLocation {
  path: string;
  /** 从 1 数起；0 = 没有行号。 */
  line: number;
  /** 一段范围的结束行；单行就是 null。 */
  endLine: number | null;
  /** 符号名（函数 / 类 / 方法）；拿不到就是 null。 */
  symbol: string | null;
  /** 符号类型（function / method / class…）；拿不到就是 null。 */
  kind: string | null;
}

/**
 * 一条流事件（翻译过的）。
 *
 * 后端一共发 9 种事件名，这里归成 7 种形状：
 * - cancelled / timed_out 只差一个终态，合成 stopped（status 里区分）；
 * - 其余一一对应。
 */
export type RunEvent =
  | { type: "start"; task: string; sessionId: string; runId: string }
  | { type: "tool_start"; tool: string; arguments: Record<string, string> }
  | {
      type: "tool_result";
      tool: string;
      success: boolean;
      durationSeconds: number;
      error: string | null;
      /** 这次调用落在哪几段（14.3）；提不出来就是空数组。 */
      locations: ToolLocation[];
    }
  | {
      type: "llm_start";
      model: string;
      attempt: number;
      maxAttempts: number;
    }
  | {
      type: "llm_end";
      model: string;
      success: boolean;
      durationSeconds: number;
      attemptCount: number;
      totalTokens: number | null;
      error: string | null;
    }
  | { type: "completed"; answer: string; sessionId: string; runId: string }
  | {
      type: "stopped";
      status: "cancelled" | "timed_out";
      message: string;
      sessionId: string;
      runId: string;
    }
  | {
      type: "error";
      message: string;
      sessionId: string;
      runId: string;
    };

/** POST /api/run 的响应（普通模式：跑完一次性拿到）。 */
export interface RunResult {
  success: boolean;
  answer: string;
  sessionId: string;
  runId: string;
  /** 只有被取消 / 超时才带；正常成功是 null。 */
  status: RunStatus | null;
}

/** POST /api/runs/cancel 的响应。 */
export interface CancelResult {
  /** 本次是不是真按下了取消；已经结束的 Run 会回来 false（幂等）。 */
  cancelled: boolean;
  runId: string;
  status: RunStatus;
}
