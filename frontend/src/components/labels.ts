/**
 * 界面上出现的字和格式，集中在这儿：状态名、时长、时间。
 *
 * 状态用**中英并排**：中文是给用户看的，英文（Running / Completed…）是后端和
 * README 里的那套词，演示时对得上号。两边不一致是最容易让人怀疑「是不是卡住了」
 * 的地方，所以宁可把两个都写上。
 */

import type { DisplayState, RunPhase, ToolFamily } from "../state/transcript";

/** 工具类别在界面上的说法（见 transcript.ts 的 toolFamily）。 */
export const FAMILY_TEXT: Record<ToolFamily, string> = {
  search: "检索",
  read: "读文件",
  edit: "改文件",
  graph: "调用图",
  run: "跑命令",
  build: "编译构建",
  compute: "计算",
  other: "其它",
};

export const STATE_TEXT: Record<DisplayState, string> = {
  idle: "空闲 Idle",
  running: "运行中 Running",
  completed: "已完成 Completed",
  cancelled: "已取消 Cancelled",
  timed_out: "已超时 Timed out",
  failed: "失败 Failed",
  interrupted: "已中断 Interrupted",
};

/** 一组过程事件里单条的三种状态。 */
export const EVENT_STATE_TEXT: Record<"running" | "ok" | "fail", string> = {
  running: "进行中",
  ok: "成功",
  fail: "失败",
};

/** 终态的一句话（横幅、分组头上用）。 */
export const PHASE_TEXT: Record<RunPhase, string> = {
  running: "运行中",
  completed: "已完成",
  cancelled: "已取消",
  timed_out: "已超时",
  failed: "失败",
  interrupted: "已中断",
};

/** 12.34 → "12.3s"；90 → "1m30s"。 */
export function formatSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m${rest.toString().padStart(2, "0")}s`;
}

/** 1523 → "1.5k"。 */
export function formatTokens(tokens: number | null): string {
  if (tokens === null || !Number.isFinite(tokens) || tokens <= 0) return "";
  if (tokens < 1000) return `${tokens} tok`;
  return `${(tokens / 1000).toFixed(1)}k tok`;
}

/** 时间戳 → "20:31:05"。 */
export function formatClock(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) return "";
  return new Date(timestamp).toLocaleTimeString("zh-CN", { hour12: false });
}
