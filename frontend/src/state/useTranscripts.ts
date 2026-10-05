/**
 * 会话记录的本子：一份会话一份 transcript，改了就落盘（见 storage.ts）。
 *
 * 为什么用一个 map 全端着（而不是「当前会话」一份）：切会话本来就是换个
 * 名字取同一本账，切回去要立刻是原样（包括折叠状态之外的一切）。全端着
 * 也就省掉了「切走时先存、切回时再读」那一串容易漏掉的时机。
 *
 * **草稿（还没拿到 session id 的新会话）只在内存里**：它连个名字都没有，
 * 存下来也只会在下次进页面时变成一条谁也认不出的记录；而后端一旦发来 id
 * （start 事件），这一轮就已经记在真名下了（见 useRun.ts）。
 */

import { useCallback, useRef, useState } from "react";

import {
  MAX_RUNS,
  TRANSCRIPT_VERSION,
  clearTranscript,
  emptyTranscript,
} from "./transcript";
import type { Transcript } from "./transcript";
import { loadAllTranscripts, removeTranscript, saveTranscript } from "./storage";
import type { SaveResult } from "./storage";

/** 草稿会话（后端还没发 id）在这本账里的名字。 */
export const DRAFT_KEY = "draft";

export interface TranscriptsApi {
  /** 取一份记录；没有就是空记录（共享的常量，只读）。 */
  get: (key: string) => Transcript;
  /** 改一份记录：没变化（返回同一个对象）就不落盘、也不触发渲染。 */
  update: (key: string, change: (current: Transcript) => Transcript) => void;
  /** 删一份记录（删会话时一起删）。 */
  remove: (key: string) => void;
  /**
   * 把一份记录换个名字（草稿 → 后端刚发的 session_id）。目标已存在就并过去
   * ——同一次运行的事件不能因为换了名字而丢一半。
   */
  move: (from: string, to: string) => void;
  /** 清空一份记录（「清空记录」按钮；后端那个会话不动）。 */
  clear: (key: string) => void;
  /** 最近一次落盘的结果：failed 时页面上说一句，别让用户以为都存上了。 */
  saveState: SaveResult | "none";
}

export function useTranscripts(): TranscriptsApi {
  const [map, setMap] = useState<Record<string, Transcript>>(() => loadAllTranscripts());
  const [saveState, setSaveState] = useState<SaveResult | "none">("none");

  // update 可能在同一轮渲染里被连着调（一轮跑起来会连着落好几条事件），
  // 只认 state 会读到旧值，所以再拿一个 ref 当「最新的那份」
  const latest = useRef(map);

  const get = useCallback((key: string): Transcript => {
    return map[key] ?? emptyTranscript();
  }, [map]);

  const update = useCallback((key: string, change: (current: Transcript) => Transcript) => {
    const before = latest.current[key] ?? emptyTranscript();
    const after = change(before);
    if (after === before) return; // 没有变化：不写盘、不渲染

    const next = { ...latest.current, [key]: after };
    latest.current = next;
    setMap(next);

    if (key === DRAFT_KEY) return; // 草稿不落盘

    const result = saveTranscript(key, after);
    setSaveState(result);
  }, []);

  const move = useCallback((from: string, to: string) => {
    if (from === to) return;
    const source = latest.current[from];
    if (source === undefined) return;

    const target = latest.current[to] ?? emptyTranscript();
    const seen = new Set(target.runs.map((run) => run.key));
    const merged: Transcript = {
      version: TRANSCRIPT_VERSION,
      // 目标在前、搬过来的在后（顺过来的时序），同一个 key 不重复收
      runs: [...target.runs, ...source.runs.filter((run) => !seen.has(run.key))].slice(-MAX_RUNS),
      updatedAt: source.updatedAt,
    };

    const next = { ...latest.current, [to]: merged };
    delete next[from];
    latest.current = next;
    setMap(next);

    if (from !== DRAFT_KEY) removeTranscript(from);
    setSaveState(saveTranscript(to, merged));
  }, []);

  const remove = useCallback((key: string) => {
    const next = { ...latest.current };
    delete next[key];
    latest.current = next;
    setMap(next);
    if (key !== DRAFT_KEY) removeTranscript(key);
  }, []);

  const clear = useCallback(
    (key: string) => {
      update(key, () => clearTranscript());
    },
    [update],
  );

  return { get, update, remove, move, clear, saveState };
}
