/**
 * 左边那一列：会话列表 + 新建 + 删除，底下是 Token 和后端状态。
 *
 * 会话 id 是后端发的，这里只显示短版本（前 8 位）——完整的那串在标题的
 * tooltip 里，真要复制去用命令行也能拿到。
 *
 * 跑着的时候整个列表是灰的：一轮 Run 认的是当时那个会话，跑到一半切走会让
 * 「这一轮算谁的」说不清（见 state/useRun.ts 的说明）。
 */

import type { Health } from "../state/useHealth";
import type { SessionSummary } from "../state/sessions";
import { shortId } from "../state/sessions";

export interface SessionSidebarProps {
  items: SessionSummary[];
  activeId: string | null;
  busy: boolean;
  health: Health;
  /** 本页面的来源，403 时显示出来让用户复制（见 state/useHealth.ts）。 */
  origin: string;
  apiLabel: string;
  token: string;
  onSelect: (id: string | null) => void;
  onDelete: (id: string) => void;
  onTokenChange: (token: string) => void;
  onRecheck: () => void;
}

const HEALTH_TEXT: Record<Health, string> = {
  checking: "检查中…",
  ok: "已连接",
  blocked: "来源不被允许",
  down: "连不上",
};

export function SessionSidebar({
  items,
  activeId,
  busy,
  health,
  origin,
  apiLabel,
  token,
  onSelect,
  onDelete,
  onTokenChange,
  onRecheck,
}: SessionSidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <span>会话</span>
        <button type="button" disabled={busy} onClick={() => onSelect(null)}>
          ＋ 新建
        </button>
      </div>

      <ul className="sessions">
        <li
          className={`session ${activeId === null ? "active" : ""}`}
          onClick={busy ? undefined : () => onSelect(null)}
        >
          <div className="session-title">新会话</div>
          <div className="session-id">还没开始（第一次运行后由后端给 id）</div>
        </li>

        {items.map((item) => (
          <li
            key={item.id}
            className={`session ${item.id === activeId ? "active" : ""}`}
            onClick={busy ? undefined : () => onSelect(item.id)}
          >
            <div className="session-title" title={item.id}>
              {item.title}
            </div>
            <div className="session-id">{shortId(item.id)}</div>
            <button
              type="button"
              className="delete"
              title="删掉这个会话（后端那本会话历史）"
              disabled={busy}
              onClick={(event) => {
                event.stopPropagation(); // 别把「删除」点成「切换」
                if (window.confirm(`删掉会话「${item.title}」？后端的会话历史会一起没。`)) {
                  onDelete(item.id);
                }
              }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>

      <div className="sidebar-foot">
        <label className="token">
          <span>API Token（选填）</span>
          <input
            type="password"
            value={token}
            placeholder="后端配了 API_TOKEN 才要填"
            autoComplete="off"
            onChange={(event) => onTokenChange(event.target.value)}
          />
        </label>
        <p className="hint">
          只存在这个页面里（刷新就没了），不写进代码、不进浏览器存储。
        </p>

        <div className="health">
          <span className={`dot dot-${health}`} />
          <span>{HEALTH_TEXT[health]}</span>
          {health === "down" || health === "blocked" ? (
            <button type="button" onClick={onRecheck}>
              重试
            </button>
          ) : null}
        </div>
        {health === "blocked" ? (
          // 后端活着但挡了来源：把要加的那一行直接摆出来，改完重启后端
          <p className="hint">
            把 <code>{origin}</code> 加进后端的 API_CORS_ORIGINS，再重启后端
          </p>
        ) : null}
        <p className="hint" title={apiLabel}>
          后端：{apiLabel}
        </p>
      </div>
    </aside>
  );
}
