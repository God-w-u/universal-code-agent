/**
 * 底部那一条：写任务、运行、取消。
 *
 * 运行中不给再按一次（后端会把同一个会话的两轮排队，但用户按了没反应更像坏了）。
 * 取消要等 run_id 到了才给按——那个 id 是后端在 start 事件里发的，前端不自己
 * 编一个（编出来的取消请求后端也认不出来）。
 */

import { useState } from "react";

export interface ComposerProps {
  busy: boolean;
  cancelling: boolean;
  /** run_id 到了没有：没有就没法精确取消。 */
  canCancel: boolean;
  onRun: (task: string) => void;
  onCancel: () => void;
}

export function Composer({ busy, cancelling, canCancel, onRun, onCancel }: ComposerProps) {
  const [task, setTask] = useState("");

  const submit = () => {
    if (busy || task.trim() === "") return;
    onRun(task);
    setTask("");
  };

  return (
    <div className="composer">
      <textarea
        value={task}
        rows={3}
        placeholder="要它做什么？例如：这个项目怎么做代码检索的？（Ctrl/⌘ + Enter 运行）"
        onChange={(event) => setTask(event.target.value)}
        onKeyDown={(event) => {
          // 回车留给换行（任务经常是多行的），Ctrl / ⌘ + Enter 才运行
          if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            submit();
          }
        }}
      />
      <div className="composer-actions">
        <button
          type="button"
          className="primary"
          disabled={busy || task.trim() === ""}
          onClick={submit}
        >
          {busy ? "Running…" : "运行"}
        </button>
        {busy ? (
          <button
            type="button"
            disabled={!canCancel || cancelling}
            title={canCancel ? "让 Agent 在下一个检查点停下" : "等这一轮的身份（run_id）到了才能取消"}
            onClick={onCancel}
          >
            {cancelling ? "取消中…" : "Cancel"}
          </button>
        ) : null}
      </div>
    </div>
  );
}
