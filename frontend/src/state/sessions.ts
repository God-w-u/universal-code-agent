/**
 * 会话列表：页面这一侧的账本（Session 状态，和 Run 状态分开管）。
 *
 * **为什么记在前端**：后端没有「列出所有会话」的接口（五条路由里只有创建
 * （跑第一轮时顺带）、使用、删、探活），也没有把会话落盘——重启后端，会话就
 * 全没了（api/sessions.py 的 TTL 过期 + 进程内表）。所以前端自己记着「我用过
 * 哪些 session_id」，刷新页面还能接着用；哪天真没了，后端会用 404 告诉我们
 * （见 useRun 的 onSessionUnknown），那时把这条从列表里划掉就是了。
 *
 * **存在 localStorage 里的只有这三样**：session_id、标题（第一句任务的前几个
 * 字）、创建时间。没有 Token、没有对话内容、没有 API Key——那些一概不落地。
 */

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "uac.frontend.sessions.v1";
const ACTIVE_KEY = "uac.frontend.active.v1";

export interface SessionSummary {
  /** 服务端发的 session_id（前端不生成）。 */
  id: string;
  title: string;
  createdAt: number;
}

/** 列表里显示的那截 id：完整的太长，认人只要前 8 位。 */
export function shortId(id: string): string {
  return id.length > 8 ? id.slice(0, 8) : id;
}

/** 用第一句任务当标题；任务可能很长很多行，取第一行前 24 个字。 */
export function titleFrom(task: string): string {
  const line = task.split("\n")[0]?.trim() ?? "";
  if (line === "") return "新会话";
  return line.length > 24 ? `${line.slice(0, 24)}…` : line;
}

export interface SessionsApi {
  items: SessionSummary[];
  /** null = 还没跑过的新会话（后端还没发 id 给我们）。 */
  activeId: string | null;
  select: (id: string | null) => void;
  /** 记下一个后端刚发的 session_id（第一轮跑起来时调）。 */
  remember: (id: string, task: string) => void;
  /** 从列表里划掉一条（删掉了 / 后端说它不存在了）。 */
  forget: (id: string) => void;
}

export function useSessions(): SessionsApi {
  const [items, setItems] = useState<SessionSummary[]>(loadItems);
  const [activeId, setActiveId] = useState<string | null>(loadActive);

  // 每次变化都写回去：刷新之后至少列表和「上次在用哪个」还在
  useEffect(() => {
    save(STORAGE_KEY, items);
  }, [items]);
  useEffect(() => {
    save(ACTIVE_KEY, activeId);
  }, [activeId]);

  const select = useCallback((id: string | null) => {
    setActiveId(id);
  }, []);

  const remember = useCallback((id: string, task: string) => {
    setItems((prev) =>
      prev.some((item) => item.id === id)
        ? prev
        : [{ id, title: titleFrom(task), createdAt: Date.now() }, ...prev],
    );
    setActiveId(id);
  }, []);

  const forget = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
    setActiveId((current) => (current === id ? null : current));
  }, []);

  return { items, activeId, select, remember, forget };
}

// ---------------------------------------------------------------------------
// localStorage：读写都要当它是坏的
// ---------------------------------------------------------------------------

function loadItems(): SessionSummary[] {
  const raw = read(STORAGE_KEY);
  if (!Array.isArray(raw)) return [];
  const items: SessionSummary[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    const record = item as Record<string, unknown>;
    const id = record.id;
    if (typeof id !== "string" || id === "") continue; // 半条记录不要
    items.push({
      id,
      title: typeof record.title === "string" ? record.title : shortId(id),
      createdAt: typeof record.createdAt === "number" ? record.createdAt : 0,
    });
  }
  return items;
}

function loadActive(): string | null {
  const raw = read(ACTIVE_KEY);
  return typeof raw === "string" && raw !== "" ? raw : null;
}

/** 读一个键；解析不了（别的插件写的、旧版本写的）就当作没存过。 */
function read(key: string): unknown {
  try {
    const text = window.localStorage.getItem(key);
    return text === null ? null : JSON.parse(text);
  } catch {
    return null;
  }
}

/** 写一个键；写不进去（隐私模式、盘满）也不该让页面崩——这只是便利。 */
function save(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* 记不住就算了，会话本身在后端 */
  }
}
