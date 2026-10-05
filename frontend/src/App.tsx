/**
 * 页面：左边会话列表，右边当前会话的记录 + 输入框。
 *
 * 这里只做「接线」：会话列表在 state/sessions.ts，记录在 state/useTranscripts.ts
 * （落盘在 state/storage.ts），这一轮在 state/useRun.ts，网络在 api/client.ts。
 *
 * **切会话不再清屏**：每个会话有自己那份 transcript（14.2 起落盘），切回去
 * 还是原样——这就是 14.2 和 14.1 最大的差别。
 *
 * Token 只放在这一层的 useState 里（内存），不写 localStorage、不进源码——
 * 后端没配 API_TOKEN 的话根本用不上它；刷新页面也不会把它捡回来。
 */

import { useState } from "react";
import type { ReactNode } from "react";

import { API_LABEL, deleteSession, isApiError } from "./api/client";
import { Banner } from "./components/Banner";
import { Composer } from "./components/Composer";
import { PaneHead } from "./components/PaneHead";
import { RunList } from "./components/RunList";
import { SessionSidebar } from "./components/SessionSidebar";
import { shortId, useSessions } from "./state/sessions";
import { currentRun, displayState } from "./state/transcript";
import { useHealth } from "./state/useHealth";
import { useNow } from "./state/useNow";
import { useRun } from "./state/useRun";
import { DRAFT_KEY, useTranscripts } from "./state/useTranscripts";

export default function App() {
  const [token, setToken] = useState("");
  const [banner, setBanner] = useState<string | null>(null);

  const health = useHealth();
  const sessions = useSessions();
  const transcripts = useTranscripts();

  const trimmedToken = token.trim();
  const run = useRun({
    token: trimmedToken === "" ? null : trimmedToken,
    transcripts,
    onSessionId: sessions.remember,
    onSessionUnknown: sessions.forget,
    onError: setBanner,
  });

  const active = sessions.items.find((item) => item.id === sessions.activeId) ?? null;
  // 没有会话 id 时（还没跑过 / 刚点「新建」）记在草稿名下，id 一到就搬过去
  const activeKey = active?.id ?? DRAFT_KEY;
  const transcript = transcripts.get(activeKey);
  const current = currentRun(transcript);
  const now = useNow(run.busy);

  const clearActive = () => {
    const what = active === null ? "这个新会话" : `会话「${active.title}」`;
    if (!window.confirm(`清空${what}在本机记的过程和回答？\n（后端那个会话不动：它的历史还在后端，删会话请用列表里的 ×）`)) {
      return;
    }
    transcripts.clear(activeKey);
  };

  const remove = async (id: string) => {
    try {
      await deleteSession(id, trimmedToken === "" ? null : trimmedToken);
      sessions.forget(id);
      transcripts.remove(id); // 会话没了，本机那份记录也一起清掉
      setBanner(`会话已删除：${shortId(id)}`);
    } catch (cause) {
      if (isApiError(cause) && cause.status === 404) {
        // 后端本来就没有这条了（过期 / 被清理）：目的已经达到，照删不误
        sessions.forget(id);
        transcripts.remove(id);
        setBanner("这个会话在后端已经没有了（过期或被清理），已从列表和本机记录里移除");
        return;
      }
      const reason = isApiError(cause) ? cause.message : "请求没发出去";
      setBanner(`删除会话失败：${reason}`);
    }
  };

  return (
    <div className="app">
      <SessionSidebar
        items={sessions.items}
        activeId={sessions.activeId}
        busy={run.busy}
        health={health.state}
        origin={health.origin}
        apiLabel={API_LABEL}
        token={token}
        onSelect={sessions.select}
        onDelete={(id) => void remove(id)}
        onTokenChange={setToken}
        onRecheck={health.recheck}
      />

      <main className="pane">
        <PaneHead
          title={active === null ? "新会话" : active.title}
          sessionId={active?.id ?? null}
          state={displayState(transcript)}
          current={current}
          now={now}
          busy={run.busy}
          onClear={clearActive}
        />

        {banner === null ? null : <Banner text={banner} onDismiss={() => setBanner(null)} />}

        <StatusStrips
          health={health.state}
          origin={health.origin}
          saveState={transcripts.saveState}
          onRecheck={health.recheck}
        />

        <RunList runs={transcript.runs} now={now} empty={emptyHint(health.state)} />

        <Composer
          busy={run.busy}
          cancelling={run.cancelling}
          // 后端还没发 run_id 时没法取消（取消要报 run_id 或 session_id）；
          // 草稿会话（连 session id 都没有）更是无从取消
          canCancel={current?.runId != null && active !== null}
          onRun={(task) => run.start(task, active?.id ?? null)}
          onCancel={run.cancel}
        />
      </main>
    </div>
  );
}

/** 后端 / 本机存储有问题时顶上那几条提示；都好就什么都不显示。 */
function StatusStrips({
  health,
  origin,
  saveState,
  onRecheck,
}: {
  health: "checking" | "ok" | "blocked" | "down";
  origin: string;
  saveState: "none" | "ok" | "trimmed" | "failed";
  onRecheck: () => void;
}) {
  return (
    <>
      {health === "down" ? (
        <p className="strip strip-warn">
          后端连不上（{API_LABEL}）：确认它已经在跑（<code>python -m api</code>），然后
          <button type="button" onClick={onRecheck}>
            重试
          </button>
        </p>
      ) : null}
      {health === "blocked" ? (
        <p className="strip strip-warn">
          后端不认这个来源（403）：把 <code>{origin}</code> 加进后端的 <code>API_CORS_ORIGINS</code>
          再重启它，然后
          <button type="button" onClick={onRecheck}>
            重试
          </button>
        </p>
      ) : null}
      {saveState === "failed" ? (
        <p className="strip strip-warn">
          本机存不下记录了（浏览器存储满或不可用）：过程和回答这次不会留到下次打开，后端那边不受影响。
        </p>
      ) : null}
      {saveState === "trimmed" ? (
        <p className="strip strip-info">浏览器存储有点紧：为了腾地方，已经清掉一些更早的记录。</p>
      ) : null}
    </>
  );
}

/** 一条记录都没有时说什么（分「后端还没就绪」和「就绪了，等第一个任务」两种）。 */
function emptyHint(health: "checking" | "ok" | "blocked" | "down"): ReactNode {
  if (health === "down" || health === "blocked") {
    return <p>后端还没就绪 —— 先处理上面那条提示，然后发任务试试。</p>;
  }
  return (
    <>
      <p className="empty-title">这个会话还没有记录</p>
      <p>
        在下面写一句任务（例如「读一下 README 的第一段，说说这个项目是干什么的」），
        点「运行」。工具调用和模型调用会实时记在这儿，最后给出回答。
      </p>
      <p className="empty-note">
        记录按会话存在浏览器里：切到别的会话再切回来，或者刷新页面，都还在。
      </p>
    </>
  );
}
