/**
 * 右边那一片的头：当前会话 + 这一轮的状态。
 *
 * 状态分两处显示：
 * - 徽标：这一轮的终态（Running / Completed / Cancelled / Timed out / Failed /
 *   Interrupted），一眼看结论；
 * - 跑着的时候多一行实时信息：run_id 短号、当前阶段、工具/模型各几次、用时、
 *   最近一条事件——回答的都是「它现在在干嘛，还要等多久」。
 *
 * 清空记录是**本机**的动作：只清这个会话在浏览器里存的过程和回答，后端那个
 * 会话和它的历史不动（要删后端那个会话请用左边列表的 ×）。跑着的时候不让清
 * ——正记着的东西被抽走，用户会以为整个会话坏了。
 */

import type { DisplayState, RunRecord } from "../state/transcript";
import { runStats } from "../state/transcript";
import { shortId } from "../state/sessions";
import { STATE_TEXT, formatSeconds } from "./labels";

export interface PaneHeadProps {
  title: string;
  /** null = 这个会话还没有 session id（后端会在第一轮把 id 发回来）。 */
  sessionId: string | null;
  state: DisplayState;
  /** 正在跑的那一轮；没在跑就是 null。 */
  current: RunRecord | null;
  /** 「现在」（毫秒）。 */
  now: number;
  busy: boolean;
  onClear: () => void;
}

export function PaneHead({ title, sessionId, state, current, now, busy, onClear }: PaneHeadProps) {
  const stats = current === null ? null : runStats(current, now);

  return (
    <header className="pane-head">
      <div className="who-and-what">
        <div className="title">{title}</div>
        <div className="sub" title={sessionId ?? ""}>
          {sessionId === null
            ? "还没有 session id —— 第一轮跑起来时后端会发一个"
            : `session ${shortId(sessionId)}`}
        </div>
      </div>

      <div className="head-right">
        {stats === null || current === null ? null : (
          <div className="live-stats" title={stats.lastEventText}>
            {current.runId === null ? null : <span>run {shortId(current.runId)}</span>}
            <span>{stats.phaseText}</span>
            <span className="live-sep">·</span>
            <span>工具 {stats.tools}</span>
            <span>模型 {stats.llms}</span>
            <span>{formatSeconds(stats.durationSeconds)}</span>
          </div>
        )}

        <div className={`badge badge-${state}`}>{STATE_TEXT[state]}</div>

        <button
          type="button"
          className="ghost"
          disabled={busy}
          title="清掉本机记的过程和回答（后端那个会话不动）"
          onClick={onClear}
        >
          清空记录
        </button>
      </div>
    </header>
  );
}
