/**
 * 一轮 Run 的卡片：任务 → 状态 → （可折叠的）过程 → 回答。
 *
 * 视觉层级是这一版的重点：**回答要在最显眼的地方，过程默认收起来**。
 * 一轮跑下来可能几十条工具/模型事件，全摊开就把结论淹了；所以：
 *
 *   任务（这一轮要干什么）
 *   ├─ 状态行：时间 / run_id 短号 / 状态 / 用时 / 工具几次 / 模型几次
 *   ├─ 执行过程（默认折叠；正在跑的这一轮默认展开，跑完自动收起）
 *   └─ 回答（Markdown，字号大一号）
 *
 * 正在跑的那一轮额外显示「当前阶段」和最近一条事件，用来回答「它现在在干嘛」。
 *
 * 过程那一列（14.3 起）是一条**时间线**：每行开头是本地收到的时间，工具行还
 * 带类别（检索 / 读文件 / 调用图……）和一行「在干什么」，有代码位置的行可以
 * 点开看 `path:line`（见 LocationList）或调用链（见 ChainView）。位置是后端
 * 从工具返回正文里提出来的**坐标**，正文本身一直没进过浏览器。
 */

import { useEffect, useState } from "react";

import type { RunEventRecord, RunRecord, ToolRecord } from "../state/transcript";
import { runStats } from "../state/transcript";
import { ChainView } from "./ChainView";
import { LocationList } from "./LocationList";
import { Markdown } from "./Markdown";
import {
  EVENT_STATE_TEXT,
  FAMILY_TEXT,
  PHASE_TEXT,
  formatClock,
  formatSeconds,
  formatTokens,
} from "./labels";

export interface RunCardProps {
  run: RunRecord;
  /** 「现在」（毫秒）：还在跑的那一轮要拿它算已跑时长。 */
  now: number;
}

export function RunCard({ run, now }: RunCardProps) {
  const running = run.phase === "running";
  const stats = runStats(run, now);

  // 默认：正在跑的展开（看过程是跑起来的目的），已经结束的收起（看结论）
  const [open, setOpen] = useState(running);
  useEffect(() => {
    // 跑完了就收起来，把版面让给回答；用户自己点开的另说（下一次渲染不再改）
    if (!running) setOpen(false);
  }, [running]);

  return (
    <article className={`run run-${run.phase}`}>
      <header className="run-head">
        <div className="run-task">{run.task}</div>
        <div className="run-meta">
          <span>{formatClock(run.startedAt)}</span>
          {run.runId === null ? null : (
            <span className="run-id" title={run.runId}>
              run {run.runId.slice(0, 8)}
            </span>
          )}
          <span className={`run-phase phase-${run.phase}`}>{PHASE_TEXT[run.phase]}</span>
          <span>{formatSeconds(stats.durationSeconds)}</span>
          {stats.tools > 0 ? <span>工具 {stats.tools}</span> : null}
          {stats.llms > 0 ? <span>模型 {stats.llms}</span> : null}
        </div>
        {running ? (
          <div className="run-live">
            <span className="dot dot-live" />
            {stats.phaseText}
            {stats.lastEventText === "" ? null : (
              <span className="run-live-last">
                （最近一次：{stats.lastEventText}
                {stats.lastEventAt === null ? "" : `，${agoText(stats.lastEventAt, now)}`}）
              </span>
            )}
          </div>
        ) : null}
      </header>

      {run.events.length === 0 ? null : (
        <details
          className="run-events"
          open={open}
          onToggle={(event) => setOpen(event.currentTarget.open)}
        >
          <summary>
            执行过程 {run.eventCount} 条
            {run.truncated ? "（较早的没留下）" : ""}
          </summary>
          <ol>
            {run.events.map((event) => (
              <EventLine key={event.id} event={event} />
            ))}
          </ol>
        </details>
      )}

      {run.answer === null ? null : <Markdown text={run.answer} />}
      {run.message === null ? null : <p className="run-message">{run.message}</p>}
    </article>
  );
}

/**
 * 过程里的一行。
 *
 * **时间线**：每一行都以本地收到的时间打头（后端没给每条事件的时间戳，见
 * README 的说明），顺序就是收到的顺序——不会跳来跳去。带代码位置的那些行
 * 折起来：平时只占一行，点开才铺开位置 / 调用链，免得几十条事件把版面占满。
 */
function EventLine({ event }: { event: RunEventRecord }) {
  if (event.kind === "notice") {
    return (
      <li className={`event event-notice level-${event.level}`}>
        <span className="event-time">{formatClock(event.at)}</span>
        <span className="event-text">{event.text}</span>
      </li>
    );
  }

  if (event.kind === "tool") {
    return <ToolLine event={event} />;
  }

  return (
    <li className="event">
      <span className="event-time">{formatClock(event.at)}</span>
      <span className="event-icon" aria-hidden="true">
        LLM
      </span>
      <span className="event-name">{event.model === "" ? "模型" : event.model}</span>
      <span className="event-meta">
        attempt {event.attempt}/{event.maxAttempts}
      </span>
      <span className={`event-state state-${event.state}`}>{EVENT_STATE_TEXT[event.state]}</span>
      {event.durationSeconds === null ? null : (
        <span className="event-meta">{formatSeconds(event.durationSeconds)}</span>
      )}
      {formatTokens(event.totalTokens) === "" ? null : (
        <span className="event-meta">{formatTokens(event.totalTokens)}</span>
      )}
      {event.error === null || event.error === "" ? null : (
        <span className="event-error">{event.error}</span>
      )}
    </li>
  );
}

/** 一条工具调用：一行摘要 +（有位置时）可展开的位置 / 调用链。 */
function ToolLine({ event }: { event: ToolRecord }) {
  const summary = (
    <>
      <span className="event-icon" aria-hidden="true">
        🔧
      </span>
      <span className="event-name">{event.tool}</span>
      <span className="event-family">{FAMILY_TEXT[event.family]}</span>
      <span className={`event-state state-${event.state}`}>{EVENT_STATE_TEXT[event.state]}</span>
      {event.durationSeconds === null ? null : (
        <span className="event-meta">{formatSeconds(event.durationSeconds)}</span>
      )}
      {event.intent === null ? null : (
        <span className="event-intent" title={event.intent}>
          {event.intent}
        </span>
      )}
      {event.locationCount === 0 ? null : (
        <span className="event-hits">
          {event.chain ? `${event.locationCount} 个节点` : `命中 ${event.locationCount} 处`}
        </span>
      )}
      {event.error === null || event.error === "" ? null : (
        <span className="event-error">{event.error}</span>
      )}
    </>
  );

  if (event.locations.length === 0) {
    return (
      <li className="event">
        <span className="event-time">{formatClock(event.at)}</span>
        {summary}
      </li>
    );
  }

  return (
    <li className="event">
      <span className="event-time">{formatClock(event.at)}</span>
      <details className="event-detail">
        <summary>{summary}</summary>
        {event.chain ? (
          <ChainView locations={event.locations} total={event.locationCount} />
        ) : (
          <LocationList locations={event.locations} total={event.locationCount} />
        )}
      </details>
    </li>
  );
}

/** 「12 秒前」这种说法，超过一分钟就说分钟。 */
function agoText(at: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - at) / 1000));
  if (seconds < 1) return "刚刚";
  if (seconds < 60) return `${seconds} 秒前`;
  return `${Math.floor(seconds / 60)} 分钟前`;
}
