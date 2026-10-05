/**
 * 调用链：`A → B → C`，每一跳下面挂它的位置（`path:line`）。
 *
 * **只给有序的链用**：后端的 call_graph 有 callers / callees / path 三种查法，
 * 只有 path 模式的结果是一条链（1 是起点，最后一条是终点）。callers / callees
 * 是**一份清单**，拿它画箭头就等于把「谁调用了它」说成「一条链」——那是错的，
 * 所以那两种走 LocationList。
 *
 * 两条安全处理（链是后端算出来的，但前端不能假设它一定干净）：
 * - **环**：链上出现重复的节点就停在重复处，并说一句「链里有环」——不无限画；
 * - **太长**：只画前 MAX_NODES 个，剩下的写「还有 N 个」——超长链不该把版面撑爆。
 *
 * 纯 DOM，没有图形库：一条链就是一列 div。
 */

import type { ToolLocation } from "../api/types";

/** 一条链最多画几个节点：再多就折起来说一声。 */
export const MAX_CHAIN_NODES = 8;

export interface ChainViewProps {
  /** 按链的顺序排好的位置（后端就是这么给的）。 */
  locations: ToolLocation[];
  /** 后端一共给了几跳（locations 可能已经被截过）。 */
  total: number;
}

export function ChainView({ locations, total }: ChainViewProps) {
  const { nodes, cyclic } = cutCycle(locations);
  const shown = nodes.slice(0, MAX_CHAIN_NODES);
  const hidden = Math.max(0, total - shown.length);

  return (
    <div className="chain">
      <ol className="chain-list">
        {shown.map((node, index) => (
          <li className="chain-node" key={`${node.path}:${node.line}:${index}`}>
            {index === 0 ? null : (
              <span className="chain-arrow" aria-hidden="true">
                ↓
              </span>
            )}
            <span className="chain-name">{node.symbol ?? node.path}</span>
            {node.kind === null ? null : <span className="chain-kind">{node.kind}</span>}
            <span className="chain-where">{whereText(node)}</span>
          </li>
        ))}
      </ol>
      {cyclic ? <p className="chain-note">链里有环（同一个符号又出现了一次），画到重复处为止</p> : null}
      {hidden > 0 ? <p className="chain-note">还有 {hidden} 跳没画出来</p> : null}
    </div>
  );
}

/**
 * 链上出现重复节点就砍掉后面那截（含重复的那个），并说一声。
 *
 * 「重复」按符号名算（没有符号名就按 `path:line`）。BFS 算出来的最短链本来
 * 不该有环，但这条数据是从后端一路传过来的，前端只负责画得安全。
 */
function cutCycle(locations: ToolLocation[]): { nodes: ToolLocation[]; cyclic: boolean } {
  const seen = new Set<string>();
  const nodes: ToolLocation[] = [];

  for (const location of locations) {
    const key = nodeKey(location);
    if (seen.has(key)) return { nodes, cyclic: true };
    seen.add(key);
    nodes.push(location);
  }
  return { nodes, cyclic: false };
}

function nodeKey(location: ToolLocation): string {
  return location.symbol ?? `${location.path}:${location.line}`;
}

/** `path:line`；没有行号（line = 0）时只给路径——不编一个行号出来。 */
function whereText(location: ToolLocation): string {
  if (location.line <= 0) return location.path;
  if (location.endLine === null) return `${location.path}:${location.line}`;
  return `${location.path}:${location.line}-${location.endLine}`;
}
