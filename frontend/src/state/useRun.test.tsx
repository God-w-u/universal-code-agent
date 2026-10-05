// @vitest-environment jsdom
/**
 * useRun 的联调测试：**用真的记录、真的落盘**（useTranscripts + localStorage），
 * 只把 api/client 那一层换成脚本——网络那一段本身在 client.test.ts 里用真
 * HTTP 测过了（那里测的是「字节怎么读」），这里测的是「事件流怎么变成记录」：
 *
 * - 一整轮流下来，记录落盘、草稿搬到真会话、终态四种都留得住；
 * - 断流 / 网络错误 / 取消，前端自己判定的结束也有一条诚实的记录；
 * - 关页面再打开（卸载再挂载）＝刷新：绝不能显示「运行中」（interrupted）；
 * - 同一轮的帧再喂一遍，事件不翻倍；
 * - Token 只在内存里，绝不进 localStorage。
 *
 * 用假的时钟跑取消那条（宽限期是 10 秒，真等会把测试拖死）。
 */

import { act, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import type { CancelResult, RunEvent } from "../api/types";
import { flush, mountHook } from "../test/harness";
import type { HookView } from "../test/harness";
import { applyFrame } from "./transcript";
import { useRun } from "./useRun";
import type { RunApi } from "./useRun";
import { useTranscripts } from "./useTranscripts";
import type { TranscriptsApi } from "./useTranscripts";

interface StreamOptions {
  task: string;
  sessionId: string | null;
  token: string | null;
  signal: AbortSignal;
  onEvent: (event: RunEvent) => void;
}

/** 测试给假 client 塞的脚本（vi.mock 会被提升，所以用 vi.hoisted 起头）。 */
const fake = vi.hoisted(() => ({
  streamRun: null as null | ((options: StreamOptions) => Promise<void>),
  cancelRun: null as null | ((
    sessionId: string,
    runId: string | null,
    token: string | null,
  ) => Promise<CancelResult>),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual, // ApiError / isApiError / isAbort 这些是真的
    supportsStreaming: () => true,
    streamRun: (options: StreamOptions) => {
      if (fake.streamRun === null) throw new Error("这条测试没给 streamRun 脚本");
      return fake.streamRun(options);
    },
    cancelRun: (sessionId: string, runId: string | null, token: string | null) => {
      if (fake.cancelRun === null) throw new Error("这条测试没给 cancelRun 脚本");
      return fake.cancelRun(sessionId, runId, token);
    },
  };
});

const PREFIX = "uac.frontend.transcript.v1.";
const TOKEN = "sk-绝密-0123456789";

const START = (sessionId = "s1", runId = "run-1"): RunEvent => ({
  type: "start",
  task: "任务甲",
  sessionId,
  runId,
});

const TOOL_START: RunEvent = { type: "tool_start", tool: "read_file", arguments: { path: "a.cpp" } };
const TOOL_RESULT: RunEvent = {
  type: "tool_result",
  tool: "read_file",
  success: true,
  durationSeconds: 0.2,
  error: null,
  locations: [],
};
const LLM_START: RunEvent = { type: "llm_start", model: "m-1", attempt: 1, maxAttempts: 2 };
const LLM_END: RunEvent = {
  type: "llm_end",
  model: "m-1",
  success: true,
  durationSeconds: 1,
  attemptCount: 1,
  totalTokens: 9,
  error: null,
};
const COMPLETED = (answer = "算出来是 2"): RunEvent => ({
  type: "completed",
  answer,
  sessionId: "s1",
  runId: "run-1",
});

/** 按顺序把事件喂完就结束的流。 */
function scripted(events: RunEvent[]): (options: StreamOptions) => Promise<void> {
  return async (options) => {
    for (const event of events) options.onEvent(event);
  };
}

/** 喂完这些事件就**停在那儿**（模拟「这一轮还在后端跑着」）。 */
function stalled(events: RunEvent[]): (options: StreamOptions) => Promise<void> {
  return async (options) => {
    for (const event of events) options.onEvent(event);
    await new Promise<never>(() => undefined);
  };
}

function stored(key: string): { runs: { phase: string }[] } | null {
  const raw = window.localStorage.getItem(`${PREFIX}${key}`);
  return raw === null ? null : (JSON.parse(raw) as { runs: { phase: string }[] });
}

interface Harness {
  transcripts: TranscriptsApi;
  run: RunApi;
  errors: string[];
  unknown: string[];
  sessionId: string | null;
}

/** App 的那套接线，缩成一条：useTranscripts + useRun。 */
function useHarness(token: string | null): Harness {
  const transcripts = useTranscripts();
  const [errors, setErrors] = useState<string[]>([]);
  const [unknown, setUnknown] = useState<string[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const run = useRun({
    token,
    transcripts,
    onSessionId: setSessionId,
    onSessionUnknown: (id) => setUnknown((prev) => [...prev, id]),
    onError: (message) => setErrors((prev) => [...prev, message]),
  });

  return { transcripts, run, errors, unknown, sessionId };
}

/** 挂一个 harness 并跑一轮；返回视图（调用方自己卸载）。 */
async function runOnce(
  task: string,
  sessionId: string | null,
  token: string | null = null,
): Promise<HookView<Harness>> {
  const view = mountHook(() => useHarness(token));
  await act(async () => {
    view.current.run.start(task, sessionId);
  });
  await flush();
  return view;
}

beforeEach(() => {
  window.localStorage.clear();
  fake.streamRun = null;
  fake.cancelRun = null;
});

afterEach(() => {
  vi.useRealTimers();
});

describe("一整轮流下来", () => {
  it("记录落盘：草稿搬到真会话，回答和计数都在", async () => {
    fake.streamRun = scripted([START(), TOOL_START, TOOL_RESULT, LLM_START, LLM_END, COMPLETED("# 好了")]);

    const view = await runOnce("任务甲", null);

    const transcript = view.current.transcripts.get("s1");
    expect(transcript.runs).toHaveLength(1);
    expect(transcript.runs[0]).toMatchObject({
      runId: "run-1",
      phase: "completed",
      answer: "# 好了",
      toolCount: 1,
      llmCount: 1,
    });
    expect(transcript.runs[0]?.events.map((event) => event.kind)).toEqual(["tool", "llm"]);
    expect(view.current.sessionId).toBe("s1"); // 后端给的 id 记进了会话列表
    expect(view.current.run.busy).toBe(false);

    // 草稿只在内存里（它没有名字，存下来下次也没人认得）；真会话那份在盘上
    expect(window.localStorage.getItem(`${PREFIX}draft`)).toBeNull();
    expect(stored("s1")?.runs[0]?.phase).toBe("completed");

    await view.unmount();
  });

  it("任务里的工具参数不落盘（可能带着被读文件的内容）", async () => {
    fake.streamRun = scripted([
      START(),
      { type: "tool_start", tool: "read_file", arguments: { path: "机密/密钥.cpp", content: "绝密内容" } },
      TOOL_RESULT,
      COMPLETED(),
    ]);

    const view = await runOnce("任务甲", null);
    const raw = window.localStorage.getItem(`${PREFIX}s1`) ?? "";
    expect(raw).not.toContain("绝密内容"); // 参数里的内容
    expect(raw).not.toContain("arguments"); // 参数本身
    // 14.3 起会留一行派生出来的「在看什么」（read_file → 它的 path）和位置，
    // 那是指给用户看的东西，不是参数原文；内容仍然一个字都没有
    expect(raw).toContain("机密/密钥.cpp");
    await view.unmount();
  });

  it("跟着已有会话跑：直接记在那个会话名下", async () => {
    fake.streamRun = scripted([START(), COMPLETED()]);
    const view = await runOnce("任务甲", "s1");

    expect(view.current.transcripts.get("s1").runs[0]?.phase).toBe("completed");
    expect(stored("s1")?.runs[0]?.phase).toBe("completed");
    await view.unmount();
  });
});

describe("四种终态都留得住", () => {
  const cases: [string, RunEvent, string][] = [
    ["completed", COMPLETED(), "completed"],
    [
      "cancelled",
      { type: "stopped", status: "cancelled", message: "已取消", sessionId: "s1", runId: "run-1" },
      "cancelled",
    ],
    [
      "timed_out",
      { type: "stopped", status: "timed_out", message: "超时了", sessionId: "s1", runId: "run-1" },
      "timed_out",
    ],
    ["error", { type: "error", message: "调用模型失败：连不上", sessionId: "s1", runId: "run-1" }, "failed"],
  ];

  it.each(cases)("%s 落到内存和盘上都是 %s", async (_name, terminal, phase) => {
    fake.streamRun = scripted([START(), terminal]);
    const view = await runOnce("任务甲", "s1");

    expect(view.current.transcripts.get("s1").runs[0]?.phase).toBe(phase);
    expect(stored("s1")?.runs[0]?.phase).toBe(phase);
    expect(view.current.run.busy).toBe(false);
    await view.unmount();
  });
});

describe("前端自己判定的结束", () => {
  it("网络错误：记成 failed，并把话摆到横幅上", async () => {
    fake.streamRun = async () => {
      throw new ApiError(null, "连不上后端（http://127.0.0.1:8000）：确认它已经在跑（python -m api）");
    };

    const view = await runOnce("任务甲", "s1");
    const run = view.current.transcripts.get("s1").runs[0];

    expect(run?.phase).toBe("failed");
    expect(run?.message).toContain("连不上后端");
    expect(view.current.errors[0]).toContain("连不上后端");
    expect(stored("s1")?.runs[0]?.phase).toBe("failed");
    await view.unmount();
  });

  it("流断了、终态没来：说清楚「可能还在后端跑着」，不假装是完成", async () => {
    fake.streamRun = scripted([START(), TOOL_START]);

    const view = await runOnce("任务甲", "s1");
    const run = view.current.transcripts.get("s1").runs[0];

    expect(run?.phase).toBe("failed");
    expect(run?.message).toContain("断线不等于取消");
    expect(run?.events[0]).toMatchObject({ state: "fail" }); // 不留一个永远转圈的圈
    await view.unmount();
  });

  it("会话在后端没了（404）：从列表划掉，并说清怎么继续", async () => {
    fake.streamRun = async () => {
      throw new ApiError(404, "没有这个会话（404）");
    };
    fake.cancelRun = async () => ({ cancelled: false, runId: "", status: "idle" });

    const view = await runOnce("任务甲", "s1");
    expect(view.current.unknown).toEqual(["s1"]);
    expect(view.current.transcripts.get("s1").runs[0]?.message).toContain("已经没有了");
    await view.unmount();
  });
});

describe("刷新", () => {
  it("跑到一半关掉页面：再打开是「已中断」，不是「运行中」", async () => {
    fake.streamRun = stalled([START(), TOOL_START]);

    const first = await runOnce("任务甲", null);
    expect(first.current.transcripts.get("s1").runs[0]?.phase).toBe("running");

    await first.unmount(); // 关页面 / 刷新（那一轮在后端照跑，这里只是看不到了）

    const second = mountHook(() => useHarness(null)); // 重新进页面
    const restored = second.current.transcripts.get("s1").runs[0];

    expect(restored?.phase).toBe("interrupted");
    expect(restored?.message).toContain("断线不等于取消");
    expect(restored?.events[0]).toMatchObject({ state: "fail" });
    // 盘上那份也被治过了：别的工具来看这份数据也不会被骗
    expect(stored("s1")?.runs[0]?.phase).toBe("interrupted");

    await second.unmount();
  });

  it("刷新不会把旧的那一轮再跑一遍（只是读回来）", async () => {
    fake.streamRun = scripted([START(), COMPLETED()]);
    const first = await runOnce("任务甲", null);
    await first.unmount();

    let calls = 0;
    fake.streamRun = async () => {
      calls += 1;
    };

    const second = mountHook(() => useHarness(null));
    await flush();
    expect(second.current.transcripts.get("s1").runs[0]?.phase).toBe("completed");
    expect(calls).toBe(0);
    await second.unmount();
  });
});

describe("同一个 run 不重复追加事件", () => {
  it("进行中的那轮：同一段帧再喂一遍，计数和过程都不翻倍", async () => {
    const events = [START(), TOOL_START, TOOL_RESULT];
    fake.streamRun = stalled(events);

    const view = await runOnce("任务甲", "s1");
    const before = view.current.transcripts.get("s1");
    const key = before.runs[0]?.key ?? "";

    // 模拟「同一段流被应用了两遍」（帧号也从头来）
    await act(async () => {
      view.current.transcripts.update("s1", (current) =>
        events.reduce((acc, event, index) => applyFrame(acc, key, event, index + 1, Date.now()), current),
      );
    });

    const after = view.current.transcripts.get("s1");
    expect(after.runs[0]?.toolCount).toBe(1);
    expect(after.runs[0]?.events).toHaveLength(1);
    expect(after.runs[0]?.cursor).toBe(before.runs[0]?.cursor);
    await view.unmount();
  });

  it("已终态的那轮：再喂什么都不改了", async () => {
    fake.streamRun = scripted([START(), TOOL_START, TOOL_RESULT, COMPLETED()]);
    const view = await runOnce("任务甲", "s1");

    const before = view.current.transcripts.get("s1");
    const key = before.runs[0]?.key ?? "";

    await act(async () => {
      view.current.transcripts.update("s1", (current) =>
        applyFrame(current, key, { type: "tool_start", tool: "晚到的工具", arguments: {} }, 99, Date.now()),
      );
    });

    expect(view.current.transcripts.get("s1")).toEqual(before);
    await view.unmount();
  });
});

describe("取消", () => {
  it("后端收下取消后会发终态，取消这件事在记录里留一句话", async () => {
    fake.streamRun = stalled([START(), TOOL_START]);
    fake.cancelRun = async () => ({ cancelled: true, runId: "run-1", status: "running" });

    const view = await runOnce("任务甲", "s1");
    await act(async () => {
      view.current.run.cancel();
    });
    await flush();

    const run = view.current.transcripts.get("s1").runs[0];
    expect(run?.events.some((event) => event.kind === "notice")).toBe(true);
    expect(view.current.run.cancelling).toBe(false);
    await view.unmount();
  });

  it("后端说已经结束了：记一句话，不改这一轮的状态", async () => {
    fake.streamRun = stalled([START()]);
    fake.cancelRun = async () => ({ cancelled: false, runId: "run-1", status: "completed" });

    const view = await runOnce("任务甲", "s1");
    await act(async () => {
      view.current.run.cancel();
    });
    await flush();

    const run = view.current.transcripts.get("s1").runs[0];
    const notice = run?.events.find((event) => event.kind === "notice");
    expect(notice).toMatchObject({ text: "这一轮已经结束了，取消来晚了一步" });
    expect(run?.phase).toBe("running"); // 等后端的终态事件，不抢答
    await view.unmount();
  });

  it("取消请求本身失败：记一句 warn，这一轮还在跑", async () => {
    fake.streamRun = stalled([START()]);
    fake.cancelRun = async () => {
      throw new ApiError(500, "后端处理这一轮时出错了（500），详情在后端的日志里");
    };

    const view = await runOnce("任务甲", "s1");
    await act(async () => {
      view.current.run.cancel();
    });
    await flush();

    const run = view.current.transcripts.get("s1").runs[0];
    expect(run?.events.some((event) => event.kind === "notice" && event.level === "warn")).toBe(true);
    expect(run?.phase).toBe("running");
    expect(view.current.errors[0]).toContain("取消失败");
    await view.unmount();
  });

  it("收下取消却迟迟不来终态：宽限期一到按「已取消」记，界面不会永远转圈", async () => {
    fake.streamRun = stalled([START(), TOOL_START]);
    fake.cancelRun = async () => ({ cancelled: true, runId: "run-1", status: "running" });

    const view = await runOnce("任务甲", "s1");
    vi.useFakeTimers(); // 之后的计时器（宽限期那 10 秒）都由测试来推

    await act(async () => {
      view.current.run.cancel();
    });
    await flush();
    expect(view.current.run.busy).toBe(true);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    const run = view.current.transcripts.get("s1").runs[0];
    expect(run?.phase).toBe("cancelled");
    expect(run?.message).toContain("后端收下了取消请求");
    expect(view.current.run.busy).toBe(false);
    expect(stored("s1")?.runs[0]?.phase).toBe("cancelled");
    await view.unmount();
  });
});

describe("Token", () => {
  it("发给后端，但绝不进 localStorage；刷新后也不在", async () => {
    let seen: string | null = "还没跑";
    fake.streamRun = async (options) => {
      seen = options.token;
      for (const event of [START(), COMPLETED()]) options.onEvent(event);
    };

    const view = await runOnce("任务甲", null, TOKEN);
    expect(seen).toBe(TOKEN); // 确实用上了（在内存里传给请求）

    const dump = JSON.stringify(window.localStorage);
    expect(dump).not.toContain(TOKEN);
    expect(dump.toLowerCase()).not.toContain("token");

    // 盘上只有记录这一种键（没有 token / 设置之类的键）
    const keys: string[] = [];
    for (let index = 0; index < window.localStorage.length; index += 1) {
      keys.push(window.localStorage.key(index) ?? "");
    }
    expect(keys.length).toBeGreaterThan(0);
    expect(keys.every((key) => key.startsWith(PREFIX))).toBe(true);

    await view.unmount();
  });
});

describe("删会话", () => {
  it("会话没了，本机那份记录也一起清掉", async () => {
    fake.streamRun = scripted([START(), COMPLETED()]);
    const view = await runOnce("任务甲", "s1");
    expect(stored("s1")).not.toBeNull();

    await act(async () => {
      view.current.transcripts.remove("s1");
    });

    expect(view.current.transcripts.get("s1").runs).toEqual([]);
    expect(stored("s1")).toBeNull();
    await view.unmount();
  });

  it("清空记录只清本机那份，会话本身还在（切回去是空的）", async () => {
    fake.streamRun = scripted([START(), COMPLETED()]);
    const view = await runOnce("任务甲", "s1");

    await act(async () => {
      view.current.transcripts.clear("s1");
    });

    expect(view.current.transcripts.get("s1").runs).toEqual([]);
    expect(stored("s1")?.runs).toEqual([]);
    await view.unmount();
  });
});
