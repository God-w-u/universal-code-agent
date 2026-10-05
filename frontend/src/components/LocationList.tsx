/**
 * 代码位置列表：一行一条 `path:line`（带符号和类型），每行两个复制按钮。
 *
 * 三条分寸都摆在这儿：
 *
 * - **没有「在编辑器里打开」**：浏览器打不开本地文件——没有宿主程序配合就
 *   做不到。所以这里放的是一个**明确说做不到**的禁用按钮（title 里写清楚），
 *   而不是一个点了没反应的按钮：后者会让人以为是自己点错了。
 * - **路径长了截断显示，完整值在 title 里**，复制复制的也是完整值——截断只
 *   是版面的事，不该影响拿到手的东西。
 * - **没有行号（line = 0）就只显示路径**：后端说了「只知道文件、不知道行号」，
 *   这里不替它编一个。（read_file 就只有路径——它本来没有行参数。）
 */

import { useEffect, useState } from "react";

import type { ToolLocation } from "../api/types";

/** 复制成功那句话显示多久（毫秒）。 */
const COPIED_MS = 1_500;
/** 一行最多显示多少个字符的路径（完整值在 title 和复制结果里）。 */
const SHOWN_PATH = 56;

export interface LocationListProps {
  locations: ToolLocation[];
  /** 后端一共给了几条（locations 可能已经被截过，见 MAX_EVENT_LOCATIONS）。 */
  total: number;
}

export function LocationList({ locations, total }: LocationListProps) {
  const hidden = Math.max(0, total - locations.length);

  return (
    <div className="locations">
      <ul className="location-list">
        {locations.map((location, index) => (
          <LocationRow key={`${location.path}:${location.line}:${index}`} location={location} />
        ))}
      </ul>
      {hidden > 0 ? <p className="location-note">还有 {hidden} 处没列出来</p> : null}
    </div>
  );
}

function LocationRow({ location }: { location: ToolLocation }) {
  const full = whereText(location);
  const [copied, setCopied] = useState<"path" | "where" | null>(null);

  // 「已复制」自己会消失：隔一会儿让按钮回到原样，不用点第二次
  useEffect(() => {
    if (copied === null) return;
    const timer = setTimeout(() => setCopied(null), COPIED_MS);
    return () => clearTimeout(timer);
  }, [copied]);

  const copy = (kind: "path" | "where", value: string) => {
    void writeClipboard(value).then((ok) => setCopied(ok ? kind : null));
  };

  return (
    <li className="location">
      <span className="location-where" title={full}>
        {shortPath(location)}
      </span>
      {location.symbol === null ? null : (
        <span className="location-symbol" title={location.symbol}>
          {location.symbol}
        </span>
      )}
      {location.kind === null ? null : <span className="location-kind">{location.kind}</span>}
      <span className="location-actions">
        <button
          type="button"
          className="chip"
          onClick={() => copy("path", location.path)}
          title={`复制路径：${location.path}`}
        >
          {copied === "path" ? "已复制" : "复制路径"}
        </button>
        <button
          type="button"
          className="chip"
          onClick={() => copy("where", full)}
          title={`复制位置：${full}`}
        >
          {copied === "where" ? "已复制" : full}
        </button>
        <button
          type="button"
          className="chip chip-off"
          disabled
          title="浏览器打不开本地文件：要编辑器支持（比如把工作台嵌进 IDE）才能做到这一步"
        >
          在编辑器打开
        </button>
      </span>
    </li>
  );
}

/**
 * 写剪贴板。**失败不算错**：不是安全上下文、用户拒了权限、老浏览器没有这个
 * API，都可能失败——返回 false，按钮就不显示「已复制」，别骗人。
 */
async function writeClipboard(value: string): Promise<boolean> {
  try {
    const clipboard = navigator.clipboard;
    if (clipboard === undefined) return false;
    await clipboard.writeText(value);
    return true;
  } catch {
    return false;
  }
}

/** 显示用的短路径：长了就留头留尾，中间用 … 省掉（完整值在 title 里）。 */
function shortPath(location: ToolLocation): string {
  const where = whereText(location);
  if (where.length <= SHOWN_PATH) return where;

  const keep = Math.floor((SHOWN_PATH - 1) / 2);
  return `${where.slice(0, keep)}…${where.slice(where.length - keep)}`;
}

/** `path:line`；line 为 0 时只给路径，末尾也不留一个孤零零的冒号。 */
function whereText(location: ToolLocation): string {
  if (location.line <= 0) return location.path;
  if (location.endLine === null) return `${location.path}:${location.line}`;
  return `${location.path}:${location.line}-${location.endLine}`;
}
