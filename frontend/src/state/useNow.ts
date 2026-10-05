/**
 * 一个「现在几点」的节拍器：**只在跑着的时候跳**。
 *
 * 「已跑 8.4 秒」这种数字要动起来才有「它还活着」的感觉；但只要没有在跑的
 * 那一轮，界面上的数字全是算完定死的（终态用 endedAt），每秒重渲染一次纯属
 * 白烧电。所以由调用方说「现在有没有在跑」。
 */

import { useEffect, useState } from "react";

export function useNow(active: boolean, intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [active, intervalMs]);

  return now;
}
