/**
 * 跑一轮：把一次执行的事件流写进「当前会话那份 transcript」里。
 *
 * 14.1 是「一轮结束就清屏」，这里改成**记录留住**：
 * - 开跑时在当前会话的记录里开一轮（beginRun）；拿到 session id 之前先记在
 *   草稿名下，id 一到就整份搬到真名下（useTranscripts.move）；
 * - 每一帧事件都过一遍 applyFrame（配对、终态、挡重复帧都在那个纯函数里），
 *   改完立刻落盘——**终态一定是存下来的**，不是等某个卸载时机去补；
 * - 前端自己判定的结束（断流、请求被拒、会话没了）走 finishRun 记成 failed：
 *   「没收到终态」和「后端说完成了」是两回事，不能混。
 *
 * **同一时间只跑一轮**：running 时 start() 直接返回（按钮也是灰的）。
 *
 * **断线不等于取消**（api/server.py 的约定）：页面关掉、网络断了，后端那一轮
 * 照跑；前端只是看不到了，所以措辞里绝不出现「已停止」。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  cancelRun,
  isAbort,
  isApiError,
  runTask,
  streamRun,
  supportsStreaming,
} from "../api/client";
import type { RunEvent, RunResult } from "../api/types";
import { shortId } from "./sessions";
import { addNotice, applyFrame, beginRun, finishRun, nextRunKey } from "./transcript";
import type { RunOutcome, Transcript } from "./transcript";
import { DRAFT_KEY } from "./useTranscripts";
import type { TranscriptsApi } from "./useTranscripts";

/**
 * 取消之后等后端终态事件的宽限期（毫秒）。后端收下取消后会走到下一个检查点，
 * 然后发 cancelled 把这一轮收尾；超过这个时间还没等到，说明那条连接卡住了，
 * 就按「已取消」记——总好过界面上永远转着圈。
 */
const CANCEL_GRACE_MS = 10_000;

export interface RunApi {
  busy: boolean;
  cancelling: boolean;
  start: (task: string, sessionId: string | null) => void;
  cancel: () => void;
}

export interface RunOptions {
  /** API Token；没有（后端没配）就是 null。只在内存里。 */
  token: string | null;
  transcripts: TranscriptsApi;
  /** 后端发来（或确认）了 session_id。 */
  onSessionId: (sessionId: string, task: string) => void;
  /** 后端说这个 session_id 它不认识了（过期 / 被删 / 重启过）。 */
  onSessionUnknown: (sessionId: string) => void;
  /** 需要用户看一眼的事（横幅）：连不上、这一轮失败、取消失败。 */
  onError: (message: string) => void;
}

interface LiveRun {
  /** 这一轮在记录里的 key；结束后置空（＝「现在没有在跑的」）。 */
  runKey: string | null;
  /** 这一轮记在哪份记录里（草稿还是真会话）。 */
  transcriptKey: string;
  runId: string | null;
  sessionId: string | null;
  cancelTimer: ReturnType<typeof setTimeout> | null;
}

export function useRun(options: RunOptions): RunApi {
  const [busy, setBusy] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  // 一次 run 里真正要紧的东西放 ref：异步流程里读它不会读到过期的闭包值
  const live = useRef<LiveRun>({
    runKey: null,
    transcriptKey: DRAFT_KEY,
    runId: null,
    sessionId: null,
    cancelTimer: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  // 回调 / Token 跟着渲染走：放在 ref 里，异步流程读到的永远是最新那份
  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  });

  const write = useCallback((change: (current: Transcript) => Transcript) => {
    optionsRef.current.transcripts.update(live.current.transcriptKey, change);
  }, []);

  const clearCancelTimer = useCallback(() => {
    if (live.current.cancelTimer !== null) {
      clearTimeout(live.current.cancelTimer);
      live.current.cancelTimer = null;
    }
  }, []);

  /** 这一轮在前端这边收尾了：按钮恢复、定时器清掉。记录那边另有各管各的。 */
  const endRun = useCallback(() => {
    live.current.runKey = null;
    clearCancelTimer();
    setBusy(false);
    setCancelling(false);
  }, [clearCancelTimer]);

  const start = useCallback(
    (rawTask: string, sessionId: string | null) => {
      const task = rawTask.trim();
      if (task === "" || live.current.runKey !== null) return;

      const { transcripts } = optionsRef.current;
      const now = Date.now();
      const runKey = nextRunKey(now);
      // 没有 session id = 新会话（草稿）：先记在草稿名下，id 到了再搬走
      const transcriptKey = sessionId ?? DRAFT_KEY;

      const controller = new AbortController();
      abortRef.current = controller;
      live.current = {
        runKey,
        transcriptKey,
        runId: null,
        sessionId,
        cancelTimer: null,
      };

      if (sessionId === null) transcripts.clear(DRAFT_KEY); // 新会话从头开始
      transcripts.update(transcriptKey, (current) => beginRun(current, task, now, runKey));
      setBusy(true);
      setCancelling(false);

      // 「见过终态没有」——流结束时靠它判断是正常收尾还是半路断了
      let settled = false;
      let seq = 0;

      /** 前端自己判定这一轮结束了（没等到后端的终态）。 */
      const fail = (message: string) => {
        settled = true;
        const key = live.current.runKey;
        if (key !== null) {
          write((current) => finishRun(current, key, "failed", message, Date.now()));
        }
        optionsRef.current.onError(message);
        endRun();
      };

      const onEvent = (event: RunEvent) => {
        seq += 1;
        const at = Date.now();

        if (event.type === "start") {
          if (event.sessionId !== "" && event.sessionId !== live.current.transcriptKey) {
            // 后端刚给了这个会话的 id：草稿那一份搬到真名下，后面的帧也记那边
            const from = live.current.transcriptKey;
            optionsRef.current.transcripts.move(from, event.sessionId);
            live.current.transcriptKey = event.sessionId;
          }
          if (event.sessionId !== "") {
            live.current.sessionId = event.sessionId;
            optionsRef.current.onSessionId(event.sessionId, task);
          }
          if (event.runId !== "") live.current.runId = event.runId;
        }

        write((current) => applyFrame(current, live.current.runKey ?? runKey, event, seq, at));

        // 终态之后这一轮就定住了：applyFrame 自己也不再往这一轮里记东西
        if (!settled && isTerminal(event)) {
          settled = true;
          endRun();
        }
      };

      void (async () => {
        try {
          if (supportsStreaming()) {
            await streamRun({
              task,
              sessionId,
              token: optionsRef.current.token,
              signal: controller.signal,
              onEvent,
            });
          } else {
            // 兜底：浏览器没有响应流，这一轮的过程看不到，只有最终结果
            const result = await runTask(task, sessionId, optionsRef.current.token, controller.signal);
            for (const event of asEvents(result, task)) onEvent(event);
          }

          if (!settled) {
            // 流自己结束了、终态却一条都没来：断线。这一轮在后端那边可能还在跑
            fail(
              "执行流在结果到达前断开了。这一轮可能还在后端跑着（断线不等于取消），结果这次看不到了",
            );
          }
        } catch (cause) {
          if (isAbort(cause)) return; // 页面卸载时自己掐的，不是错
          // 终态已经收到了，后面读流再出什么事都不改变这一轮的结果：后端发完
          // completed 就关连接，连接关得不干净时读端会抛——那不是「这一轮失败」
          if (settled) return;
          const gone = live.current.sessionId;
          if (isApiError(cause) && cause.status === 404 && gone !== null) {
            // 这个会话后端不认识了：从列表里划掉，并说清楚怎么继续
            optionsRef.current.onSessionUnknown(gone);
            fail("这个会话在后端已经没有了（过期或被清理）。已从列表里移除，重新发一次会开一个新会话");
          } else {
            fail(describe(cause));
          }
        }
      })();
    },
    [endRun, write],
  );

  const cancel = useCallback(() => {
    const { runId: currentRunId, sessionId, runKey, transcriptKey } = live.current;
    if (runKey === null || sessionId === null) return;

    setCancelling(true);
    void (async () => {
      try {
        // run_id 用后端发的那个真 id；还没收到（start 事件没到）就只报会话，
        // 让后端自己找这个会话当前那一轮
        const result = await cancelRun(sessionId, currentRunId, optionsRef.current.token);
        optionsRef.current.transcripts.update(transcriptKey, (current) =>
          addNotice(
            current,
            runKey,
            result.cancelled
              ? `已发出取消：Agent 会在下一个检查点停下（run ${shortId(result.runId)}）`
              : "这一轮已经结束了，取消来晚了一步",
            "info",
            Date.now(),
          ),
        );

        if (result.cancelled) {
          // 取消收下了、终态却迟迟不来：宽限期一到就按「已取消」记，别一直转圈
          clearCancelTimer();
          live.current.cancelTimer = setTimeout(() => {
            if (live.current.runKey !== runKey) return; // 已经自己收尾了
            write((current) =>
              finishRun(
                current,
                runKey,
                "cancelled",
                "已取消：后端收下了取消请求，但之后没再发终态事件（连接可能卡住了）",
                Date.now(),
              ),
            );
            endRun();
          }, CANCEL_GRACE_MS);
        }
      } catch (cause) {
        const reason = isApiError(cause) ? cause.message : "取消请求没发出去";
        optionsRef.current.transcripts.update(transcriptKey, (current) =>
          addNotice(current, runKey, `取消没成功：${reason}（这一轮还在跑）`, "warn", Date.now()),
        );
        optionsRef.current.onError(`取消失败：${reason}（这一轮还在跑）`);
      } finally {
        setCancelling(false);
      }
    })();
  }, [clearCancelTimer, endRun, write]);

  // 关页面 / 换组件树：把这条流掐了。**后端那一轮照跑**——断线不等于取消，
  // 真要停得走 cancel()（那才是给后端发信号）
  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (live.current.cancelTimer !== null) clearTimeout(live.current.cancelTimer);
    },
    [],
  );

  return { busy, cancelling, start, cancel };
}

function isTerminal(event: RunEvent): boolean {
  return event.type === "completed" || event.type === "stopped" || event.type === "error";
}

/** 终态事件 → 这一轮的终态（只给测试和阅读用；写记录是 applyFrame 的事）。 */
export function terminalPhase(event: RunEvent): RunOutcome | null {
  if (event.type === "completed") return "completed";
  if (event.type === "stopped") return event.status;
  if (event.type === "error") return "failed";
  return null;
}

/** 普通模式的响应 → 和流式一模一样的事件，后面就只有一条渲染路径。 */
function asEvents(result: RunResult, task: string): RunEvent[] {
  const events: RunEvent[] = [];
  const { sessionId, runId } = result;

  if (sessionId !== "" && runId !== "") {
    // 先补一条 start：会话 id、run_id 都在这条上，和流式那边对齐
    events.push({ type: "start", task, sessionId, runId });
  }

  if (result.success) {
    events.push({ type: "completed", answer: result.answer, sessionId, runId });
  } else if (result.status === "cancelled" || result.status === "timed_out") {
    events.push({
      type: "stopped",
      status: result.status,
      message: result.answer,
      sessionId,
      runId,
    });
  } else {
    events.push({ type: "error", message: result.answer, sessionId, runId });
  }
  return events;
}

/** 异常 → 给用户看的一句话。ApiError 里已经是中文了。 */
function describe(cause: unknown): string {
  if (cause instanceof ApiError) return cause.message;
  if (cause instanceof Error) {
    // 不把异常的 name / message 摆到页面上（可能是浏览器的英文原文）
    console.error("[run] 未归类的错误", cause);
  }
  return "执行时出了意外（详情看浏览器控制台和后端日志）";
}
