/**
 * 记录区：一轮一轮往下排，滚到底。
 *
 * 自动滚动只在「用户本来就在底部」时才跟着走（手机上翻看历史别被拽回去的
 * 那个道理）——判断方式就是离底部还有多远，见 onScroll。
 */

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

import type { RunRecord } from "../state/transcript";
import { RunCard } from "./RunCard";

export interface RunListProps {
  runs: RunRecord[];
  /** 「现在」（毫秒）：还在跑的那一轮拿它算时长。 */
  now: number;
  /** 一条记录都没有时显示什么（说什么由 App 决定：可能后端还没起来）。 */
  empty: ReactNode;
}

export function RunList({ runs, now, empty }: RunListProps) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const box = boxRef.current;
    if (box === null || !pinnedRef.current) return;
    box.scrollTop = box.scrollHeight;
  }, [runs]);

  const onScroll = () => {
    const box = boxRef.current;
    if (box === null) return;
    pinnedRef.current = box.scrollHeight - box.scrollTop - box.clientHeight < 48;
  };

  return (
    <div className="runs" ref={boxRef} onScroll={onScroll}>
      {runs.length === 0 ? (
        <div className="empty">{empty}</div>
      ) : (
        runs.map((run) => <RunCard key={run.key} run={run} now={now} />)
      )}
    </div>
  );
}
