/**
 * 后端探活：GET /api/health。
 *
 * 只回答「后端还活着吗」——它不查 Agent 状态、不需要 Token（后端特意开放的，
 * 见 api/server.py）。三种结果分开显示：
 *
 *   down    连不上：后端没起来，或者地址/端口不对
 *   blocked 连得上但被 403：后端活着，只是没把本页面加进 API_CORS_ORIGINS
 *
 * 「连不上」和「来源不被允许」要做的事完全不一样，糊成一句话就等于让用户去猜。
 *
 * 时机：进页面探一次、窗口重新获得焦点时探一次（切回来发现后端没了很常见）、
 * 用户点「重试」时探一次。不做定时轮询——本机开发，没必要每秒问一遍。
 */

import { useCallback, useEffect, useState } from "react";

import { checkHealth } from "../api/client";
import type { HealthState } from "../api/client";

export type Health = "checking" | HealthState;

export interface HealthApi {
  state: Health;
  /**
   * 本页面在后端眼里的来源（http://localhost:5173 这样）。403 时页面上要
   * 把它原样显示出来让用户复制——「加进 API_CORS_ORIGINS」得有个具体值
   * 才有用。非浏览器环境（单测）给空串。
   */
  origin: string;
  /** 手动重探（页面上那个「重试」）。 */
  recheck: () => void;
}

export function useHealth(): HealthApi {
  const [state, setState] = useState<Health>("checking");

  const probe = useCallback((loud: boolean) => {
    if (loud) setState("checking");
    void checkHealth().then(setState);
    // 探活失败不往上抛：状态条显示结果，不弹错误框
  }, []);

  useEffect(() => {
    probe(false);
    const onFocus = () => probe(false);
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [probe]);

  const recheck = useCallback(() => probe(true), [probe]);

  return {
    state,
    origin: typeof window === "undefined" ? "" : window.location.origin,
    recheck,
  };
}
