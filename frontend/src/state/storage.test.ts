// @vitest-environment jsdom
/**
 * 落盘这一层的测试。localStorage 在这里被当成「随时可能是坏的 / 随时可能写不进去」
 * 的东西来对待：JSON 手改过、版本对不上、字段类型不对、写的时候爆配额——每种
 * 都得有个说得过去的反应，**不能把页面带崩**。
 *
 * 还有一条是安全相关的：Token 不进 localStorage。这一层存进去的东西完全由
 * Transcript 的结构决定，Token 根本不在结构里——所以这里能钉的是「写进去的
 * 字节里没有那种东西」，真正的「Token 没被存」在 useRun.test.tsx 里从跑一遍
 * 流的角度再钉一次。
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import { applyFrame, beginRun, emptyTranscript, markInterrupted } from "./transcript";
import type { Transcript } from "./transcript";
import {
  MAX_STORED,
  listStoredSessions,
  loadAllTranscripts,
  readTranscript,
  removeAllTranscripts,
  removeTranscript,
  saveTranscript,
} from "./storage";

const T0 = 1_700_000_000_000;
const PREFIX = "uac.frontend.transcript.v1.";

beforeEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

/** 造一份有内容的记录：一轮跑完，带一条工具、一条模型、一条回答。 */
function sample(task = "任务甲", at = T0): Transcript {
  let t = beginRun(emptyTranscript(), task, at, "r1");
  t = applyFrame(t, "r1", { type: "start", task, sessionId: "s1", runId: "run-1" }, 1, at);
  t = applyFrame(t, "r1", { type: "tool_start", tool: "read_file", arguments: {} }, 2, at + 100);
  t = applyFrame(
    t,
    "r1",
    { type: "tool_result", tool: "read_file", success: true, durationSeconds: 0.2, error: null, locations: [] },
    3,
    at + 300,
  );
  t = applyFrame(t, "r1", { type: "llm_start", model: "m-1", attempt: 1, maxAttempts: 2 }, 4, at + 400);
  t = applyFrame(
    t,
    "r1",
    {
      type: "llm_end",
      model: "m-1",
      success: true,
      durationSeconds: 1,
      attemptCount: 1,
      totalTokens: 12,
      error: null,
    },
    5,
    at + 1_400,
  );
  return applyFrame(t, "r1", { type: "completed", answer: "算出来是 2", sessionId: "s1", runId: "run-1" }, 6, at + 1_500);
}

describe("存取一来一回", () => {
  it("存进去读回来是同一份（原样，不是近似）", () => {
    const transcript = sample();
    expect(saveTranscript("s1", transcript)).toBe("ok");

    expect(readTranscript("s1")).toEqual(transcript);
  });

  it("进页面时把所有会话都读回来", () => {
    saveTranscript("s1", sample("任务甲", T0));
    saveTranscript("s2", sample("任务乙", T0 + 10_000));

    const all = loadAllTranscripts();
    expect(Object.keys(all).sort()).toEqual(["s1", "s2"]);
    expect(all.s2?.runs[0]?.task).toBe("任务乙");
  });

  it("没存过的会话读出来是 null，不是空壳", () => {
    expect(readTranscript("没这个会话")).toBeNull();
  });

  it("别的前缀的键不碰（同一个域下还有别的东西）", () => {
    window.localStorage.setItem("别的程序的键", "别动我");
    saveTranscript("s1", sample());

    expect(listStoredSessions()).toEqual(["s1"]);
    removeAllTranscripts();
    expect(window.localStorage.getItem("别的程序的键")).toBe("别动我");
    expect(window.localStorage.getItem(`${PREFIX}s1`)).toBeNull();
  });

  it("位置、分类、条数、是不是链：读回来一个不少（14.3）", () => {
    let t = beginRun(emptyTranscript(), "任务甲", T0, "r1");
    t = applyFrame(
      t,
      "r1",
      {
        type: "tool_start",
        tool: "call_graph",
        arguments: { function: "top", target: "leaf", direction: "path" },
      },
      1,
      T0,
    );
    t = applyFrame(
      t,
      "r1",
      {
        type: "tool_result",
        tool: "call_graph",
        success: true,
        durationSeconds: 0.4,
        error: null,
        locations: [
          { path: "a.py", line: 9, endLine: 10, symbol: "top", kind: "function" },
          { path: "a.py", line: 1, endLine: 2, symbol: "leaf", kind: "function" },
        ],
      },
      2,
      T0 + 100,
    );
    // 跑完这一轮：还在 running 的话，读回来会被 markInterrupted 改成「中断」，
    // 那验的就不是「原样存读」了
    t = applyFrame(
      t,
      "r1",
      { type: "completed", answer: "看完了", sessionId: "s1", runId: "run-1" },
      3,
      T0 + 200,
    );

    const before = t.runs[0]?.events[0];
    expect(saveTranscript("s1", t)).toBe("ok");

    expect(readTranscript("s1")?.runs[0]?.events[0]).toEqual(before);
  });
});

describe("坏数据：一律当没存过，并且不许崩", () => {
  it("根本不是 JSON", () => {
    window.localStorage.setItem(`${PREFIX}s1`, "{这不是 json");
    expect(readTranscript("s1")).toBeNull();
    expect(loadAllTranscripts()).toEqual({});
  });

  it("是 JSON 但不是这一层的东西（数组 / 数字 / null）", () => {
    window.localStorage.setItem(`${PREFIX}s1`, "[1,2,3]");
    window.localStorage.setItem(`${PREFIX}s2`, "42");
    window.localStorage.setItem(`${PREFIX}s3`, "null");
    expect(loadAllTranscripts()).toEqual({});
  });

  it("版本对不上就当空：不猜别的版本的字段", () => {
    window.localStorage.setItem(`${PREFIX}s1`, JSON.stringify({ version: 99, runs: [], updatedAt: T0 }));
    expect(readTranscript("s1")).toBeNull();
  });

  it("runs 不是数组 → 整份丢掉", () => {
    window.localStorage.setItem(`${PREFIX}s1`, JSON.stringify({ version: 1, runs: "一堆", updatedAt: T0 }));
    expect(readTranscript("s1")).toBeNull();
  });

  it("坏的那一份不影响好的那一份", () => {
    saveTranscript("好会话", sample());
    window.localStorage.setItem(`${PREFIX}坏会话`, "{{{");

    const all = loadAllTranscripts();
    expect(Object.keys(all)).toEqual(["好会话"]);
  });

  it("坏的一轮丢掉，同一份里好的一轮留着", () => {
    saveTranscript("s1", sample());
    const raw = JSON.parse(window.localStorage.getItem(`${PREFIX}s1`) ?? "{}") as {
      runs: unknown[];
      updatedAt: number;
    };
    raw.runs.unshift({ 没有key也没有task: true }, { key: "r0", task: "只有身份" });
    window.localStorage.setItem(`${PREFIX}s1`, JSON.stringify({ version: 1, runs: raw.runs, updatedAt: T0 }));

    const loaded = readTranscript("s1");
    expect(loaded?.runs.map((run) => run.task)).toEqual(["只有身份", "任务甲"]);
  });

  it("认不出来的事件丢掉，那一轮照样在", () => {
    window.localStorage.setItem(
      `${PREFIX}s1`,
      JSON.stringify({
        version: 1,
        updatedAt: T0,
        runs: [
          {
            key: "r1",
            task: "任务甲",
            startedAt: T0,
            endedAt: null,
            phase: "completed",
            events: [
              { kind: "unknown-kind", id: "e1", at: T0 },
              { kind: "tool", id: "e2", tool: "grep", state: "ok", at: T0, durationSeconds: null, error: null },
              { kind: "notice", id: "e3", text: "说一句", at: T0 },
            ],
          },
        ],
      }),
    );

    const run = readTranscript("s1")?.runs[0];
    expect(run?.events.map((event) => event.kind)).toEqual(["tool", "notice"]);
  });

  it("位置存坏了：坏的那条丢掉，好的那条留着（认不出来不猜）", () => {
    window.localStorage.setItem(
      `${PREFIX}s1`,
      JSON.stringify({
        version: 1,
        updatedAt: T0,
        runs: [
          {
            key: "r1",
            task: "任务甲",
            startedAt: T0,
            phase: "completed",
            events: [
              {
                kind: "tool",
                id: "e1",
                tool: "search_code",
                state: "ok",
                at: T0,
                durationSeconds: null,
                error: null,
                locations: [
                  null, // 不是对象
                  { line: 3 }, // 没有路径
                  { path: "", line: 3 }, // 空路径
                  "a.py:1", // 根本不是一条位置
                  { path: "好.py", line: 7, endLine: null, symbol: "f", kind: "function" },
                ],
                locationCount: 5,
              },
            ],
          },
        ],
      }),
    );

    const event = readTranscript("s1")?.runs[0]?.events[0];
    expect(event?.kind === "tool" ? event.locations : []).toEqual([
      { path: "好.py", line: 7, endLine: null, symbol: "f", kind: "function" },
    ]);
    // 条数是后端给的 5，不是留下来的 1——界面要说「还有 4 处」
    expect(event).toMatchObject({ locationCount: 5 });
  });

  it("老记录（14.3 之前）没有这些字段：分类现算，位置就是空", () => {
    window.localStorage.setItem(
      `${PREFIX}s1`,
      JSON.stringify({
        version: 1,
        updatedAt: T0,
        runs: [
          {
            key: "r1",
            task: "任务甲",
            startedAt: T0,
            phase: "completed",
            events: [
              { kind: "tool", id: "e1", tool: "search_code", state: "ok", at: T0, durationSeconds: null, error: null },
            ],
          },
        ],
      }),
    );

    const event = readTranscript("s1")?.runs[0]?.events[0];
    expect(event).toMatchObject({
      family: "search", // 工具名还在，算得出来
      intent: null,
      locations: [],
      locationCount: 0,
      chain: false,
    });
  });

  it("状态是认不出来的字 → 落到 failed（宁可说失败，也不显示运行中）", () => {
    window.localStorage.setItem(
      `${PREFIX}s1`,
      JSON.stringify({
        version: 1,
        updatedAt: T0,
        runs: [{ key: "r1", task: "任务甲", startedAt: T0, phase: "什么状态", events: [] }],
      }),
    );

    expect(readTranscript("s1")?.runs[0]?.phase).toBe("failed");
  });

  it("计数缺了就按留下来的事件数一个（老记录也得有数字）", () => {
    window.localStorage.setItem(
      `${PREFIX}s1`,
      JSON.stringify({
        version: 1,
        updatedAt: T0,
        runs: [
          {
            key: "r1",
            task: "任务甲",
            startedAt: T0,
            phase: "completed",
            events: [
              { kind: "tool", id: "e1", tool: "a", state: "ok", at: T0 },
              { kind: "tool", id: "e2", tool: "b", state: "ok", at: T0 },
              { kind: "notice", id: "e3", text: "说一句", at: T0 },
            ],
          },
        ],
      }),
    );

    const run = readTranscript("s1")?.runs[0];
    expect(run?.toolCount).toBe(2);
    expect(run?.llmCount).toBe(0);
    expect(run?.eventCount).toBe(3);
    expect(run?.truncated).toBe(false);
  });
});

describe("刷新之后不能有假的「运行中」", () => {
  it("存着 running 的，读出来就是 interrupted，并且顺手写回盘上", () => {
    const running = beginRun(emptyTranscript(), "任务甲", T0, "r1");
    expect(saveTranscript("s1", running)).toBe("ok");

    const loaded = readTranscript("s1", T0 + 60_000);
    expect(loaded?.runs[0]?.phase).toBe("interrupted");
    expect(loaded?.runs[0]?.endedAt).toBe(T0 + 60_000);

    // 写回去了：盘上不再是 running（别的工具来看这份数据也不会被骗）
    const raw = window.localStorage.getItem(`${PREFIX}s1`) ?? "";
    expect(raw).not.toContain('"running"');
    expect(JSON.parse(raw)).toMatchObject({ runs: [{ phase: "interrupted" }] });
  });

  it("已经是终态的不动（连内容都不改写）", () => {
    saveTranscript("s1", sample());
    const before = window.localStorage.getItem(`${PREFIX}s1`);
    readTranscript("s1", T0 + 99_999);
    expect(window.localStorage.getItem(`${PREFIX}s1`)).toBe(before);
  });

  it("loadAllTranscripts 这一路同样会治", () => {
    saveTranscript("s1", beginRun(emptyTranscript(), "任务甲", T0, "r1"));
    expect(loadAllTranscripts(T0 + 1_000).s1?.runs[0]?.phase).toBe("interrupted");
  });

  it("已经中断过的再读一遍也还是中断（不改结束时间）", () => {
    const interrupted = markInterrupted(beginRun(emptyTranscript(), "任务甲", T0, "r1"), T0 + 5_000);
    saveTranscript("s1", interrupted);

    expect(readTranscript("s1", T0 + 60_000)?.runs[0]?.endedAt).toBe(T0 + 5_000);
  });
});

describe("删除", () => {
  it("删会话就删掉它那份，别的会话不受影响", () => {
    saveTranscript("s1", sample());
    saveTranscript("s2", sample("任务乙"));

    removeTranscript("s1");
    expect(readTranscript("s1")).toBeNull();
    expect(readTranscript("s2")).not.toBeNull();
    expect(listStoredSessions()).toEqual(["s2"]);
  });

  it("删没有的会话不报错", () => {
    expect(() => removeTranscript("没这个会话")).not.toThrow();
  });

  it("全部删干净", () => {
    saveTranscript("s1", sample());
    saveTranscript("s2", sample());
    removeAllTranscripts();
    expect(listStoredSessions()).toEqual([]);
    expect(loadAllTranscripts()).toEqual({});
  });
});

describe("写不进去的时候", () => {
  /** 让第 n 次 setItem 抛配额错；其余的照常写。 */
  function quotaOnCalls(failCalls: number[]): void {
    const real = Storage.prototype.setItem;
    let call = 0;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (
      this: Storage,
      key: string,
      value: string,
    ) {
      call += 1;
      if (failCalls.includes(call)) {
        const error = new Error("配额满了");
        error.name = "QuotaExceededError";
        throw error;
      }
      real.call(this, key, value);
    });
  }

  it("爆配额时先清别的会话里最老的，清完再写 → trimmed", () => {
    for (let index = 0; index < MAX_STORED; index += 1) {
      saveTranscript(`老会话${index}`, sample(`任务${index}`, T0 + index * 1_000));
    }
    const oldest = listStoredSessions()[0] ?? "";

    quotaOnCalls([1]); // 第一下写不进去，之后就正常
    const result = saveTranscript("新会话", sample("新任务", T0 + 999_999));

    expect(result).toBe("trimmed");
    expect(readTranscript("新会话")).not.toBeNull();
    expect(listStoredSessions()).toHaveLength(MAX_STORED); // 进一个出一个
    expect(listStoredSessions()).not.toContain(oldest); // 走的是最老的那个
  });

  it("实在写不下就砍自己最老的几轮，写完说明是 trimmed", () => {
    let long = emptyTranscript();
    for (let index = 0; index < 4; index += 1) {
      long = beginRun(long, `第 ${index} 轮`, T0 + index, `r${index}`);
    }

    quotaOnCalls([1, 2]); // 前两下写不进去，第三下成功
    const result = saveTranscript("s1", long);

    expect(result).toBe("trimmed");
    expect(readTranscript("s1")?.runs).toHaveLength(1); // 砍到只剩一轮，还没得砍就写进去了
  });

  it("只剩一轮还写不进去 → failed，让页面去说「本机存不下了」", () => {
    quotaOnCalls([1, 2, 3, 4, 5]);
    expect(saveTranscript("s1", sample())).toBe("failed");
  });

  it("不是配额的错（比如被禁用）→ 直接 failed，不做无用功", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("存储被禁用了");
    });
    expect(saveTranscript("s1", sample())).toBe("failed");
  });

  it("写失败也不影响已经存着的别的内容", () => {
    saveTranscript("s1", sample());
    quotaOnCalls([1, 2, 3, 4, 5]);
    expect(saveTranscript("s2", sample())).toBe("failed");
    expect(readTranscript("s1")).not.toBeNull();
  });
});

describe("写进去的字节里不该有的东西", () => {
  it("没有 token / 密钥字段，也没有工具参数", () => {
    let t = beginRun(emptyTranscript(), "任务甲", T0, "r1");
    t = applyFrame(
      t,
      "r1",
      { type: "tool_start", tool: "read_file", arguments: { path: "secret.cpp", content: "密钥" } },
      1,
      T0,
    );
    saveTranscript("s1", t);

    const raw = window.localStorage.getItem(`${PREFIX}s1`) ?? "";
    expect(raw).not.toContain("密钥"); // 参数里的内容
    expect(raw).not.toContain("arguments"); // 参数本身
    for (const field of ["token", "apiKey", "api_key", "authorization"]) {
      expect(raw.toLowerCase()).not.toContain(field.toLowerCase());
    }
  });

  it("存着的是派生出来的一行「在看什么」，不是参数原文（14.3）", () => {
    // 路径会存：它是指给用户看的「这次调用落在哪个文件」，不是被读文件的内容。
    // 但只此一行——参数里的其它键（可能带整份源码）一个都不存。
    let t = beginRun(emptyTranscript(), "任务甲", T0, "r1");
    t = applyFrame(
      t,
      "r1",
      {
        type: "tool_start",
        tool: "edit_file",
        arguments: { path: "src/a.cpp", old_text: "旧源码", new_text: "新源码" },
      },
      1,
      T0,
    );
    saveTranscript("s1", t);

    const raw = window.localStorage.getItem(`${PREFIX}s1`) ?? "";
    expect(raw).toContain("src/a.cpp");
    expect(raw).not.toContain("旧源码");
    expect(raw).not.toContain("新源码");
  });
});
