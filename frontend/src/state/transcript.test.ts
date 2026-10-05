/**
 * 记录模型的单测：全是纯函数，喂事件就行，不联网、不碰时钟（时间由测试传）。
 *
 * 这里钉的是几件「错了会很难查」的事：
 * - 配对（tool_start ↔ tool_result、llm_start ↔ llm_end）；
 * - 终态之后不再改这一轮（后端发完 completed 就关连接，读端再冒出什么都不算）；
 * - 同一帧不重复记（cursor）；
 * - 刷新后没有「假的运行中」（markInterrupted）；
 * - 上限到了要**丢得诚实**：轮数和过程条数有上限，但计数不能跟着骗人。
 */

import { describe, expect, it } from "vitest";

import type { RunEvent } from "../api/types";
import {
  MAX_ANSWER,
  MAX_EVENT_LOCATIONS,
  MAX_EVENTS,
  MAX_RUNS,
  addNotice,
  applyFrame,
  beginRun,
  clearTranscript,
  currentRun,
  displayState,
  emptyTranscript,
  finishRun,
  latestRun,
  markInterrupted,
  outcomeText,
  runStats,
  terminalStatus,
  toolFamily,
} from "./transcript";
import type { Transcript } from "./transcript";

const T0 = 1_700_000_000_000;

function openRun(task = "任务甲", key = "r1", now = T0): Transcript {
  return beginRun(emptyTranscript(), task, now, key);
}

/** 往一轮里喂一帧（seq 自己递增，和 useRun 里一样）。 */
function feed(t: Transcript, key: string, event: RunEvent, seq: number, now = T0): Transcript {
  return applyFrame(t, key, event, seq, now);
}

const START = (runId = "run-1"): RunEvent => ({
  type: "start",
  task: "任务甲",
  sessionId: "s1",
  runId,
});

/** 一条 tool_result；位置信息默认没有（要验它的用例自己给）。 */
const RESULT = (
  tool: string,
  overrides: Partial<Extract<RunEvent, { type: "tool_result" }>> = {},
): RunEvent => ({
  type: "tool_result",
  tool,
  success: true,
  durationSeconds: 0.25,
  error: null,
  locations: [],
  ...overrides,
});

describe("开一轮", () => {
  it("任务是任务、状态是 running，过程还空着", () => {
    // 任务原样存下（首尾空格是 useRun.start 那边先 trim 掉的，不归这一层管）
    const t = openRun("把 1+1 算出来");
    expect(t.runs).toHaveLength(1);
    expect(t.runs[0]).toMatchObject({
      key: "r1",
      runId: null,
      task: "把 1+1 算出来",
      phase: "running",
      answer: null,
      message: null,
      toolCount: 0,
      llmCount: 0,
      cursor: 0,
    });
    expect(displayState(t)).toBe("running");
    expect(currentRun(t)?.key).toBe("r1");
  });

  it("start 事件只补 run_id，不长过程记录", () => {
    const t = feed(openRun(), "r1", START("run-abc"), 1);

    expect(t.runs[0]?.runId).toBe("run-abc");
    expect(t.runs[0]?.events).toEqual([]);
    expect(t.runs[0]?.cursor).toBe(1);
  });
});

describe("工具 / 模型事件的配对", () => {
  it("tool_start 开一条，tool_result 补上结果与用时", () => {
    const started = feed(openRun(), "r1", { type: "tool_start", tool: "read_file", arguments: {} }, 1);
    const done = feed(
      started,
      "r1",
      RESULT("read_file"),
      2,
    );

    expect(done.runs[0]?.events).toHaveLength(1); // 还是一条：结果是补上去的
    expect(done.runs[0]?.events[0]).toMatchObject({
      kind: "tool",
      tool: "read_file",
      state: "ok",
      durationSeconds: 0.25,
      error: null,
    });
    expect(done.runs[0]?.toolCount).toBe(1);
  });

  it("同名工具连着跑两次，结果各归各的（配最近那条还没结束的）", () => {
    let t = openRun();
    t = feed(t, "r1", { type: "tool_start", tool: "grep", arguments: {} }, 1);
    t = feed(t, "r1", RESULT("grep", { durationSeconds: 1 }), 2);
    t = feed(t, "r1", { type: "tool_start", tool: "grep", arguments: {} }, 3);
    t = feed(t, "r1", RESULT("grep", { success: false, durationSeconds: 2, error: "没找到" }), 4);

    const events = t.runs[0]?.events ?? [];
    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({ state: "ok", durationSeconds: 1 });
    expect(events[1]).toMatchObject({ state: "fail", durationSeconds: 2, error: "没找到" });
    expect(t.runs[0]?.toolCount).toBe(2);
  });

  it("结果先到（没配对上）也单独记一条，不把结果丢了", () => {
    const t = feed(
      openRun(),
      "r1",
      RESULT("calculator", { durationSeconds: 0.1 }),
      1,
    );

    expect(t.runs[0]?.events).toHaveLength(1);
    expect(t.runs[0]?.events[0]).toMatchObject({ kind: "tool", tool: "calculator", state: "ok" });
  });

  it("llm_start / llm_end 配对，用量和 attempt 都留着", () => {
    let t = openRun();
    t = feed(t, "r1", { type: "llm_start", model: "m-1", attempt: 1, maxAttempts: 3 }, 1);
    t = feed(
      t,
      "r1",
      {
        type: "llm_end",
        model: "m-1",
        success: true,
        durationSeconds: 1.5,
        attemptCount: 1,
        totalTokens: 42,
        error: null,
      },
      2,
    );

    expect(t.runs[0]?.events).toHaveLength(1);
    expect(t.runs[0]?.events[0]).toMatchObject({
      kind: "llm",
      model: "m-1",
      attempt: 1,
      maxAttempts: 3,
      state: "ok",
      durationSeconds: 1.5,
      totalTokens: 42,
    });
    expect(t.runs[0]?.llmCount).toBe(1);
  });

  it("工具参数不写进记录（可能带着被读文件的内容）", () => {
    const t = feed(
      openRun(),
      "r1",
      { type: "tool_start", tool: "read_file", arguments: { path: "a.cpp", content: "秘密" } },
      1,
    );

    const record = t.runs[0]?.events[0];
    expect(record?.kind).toBe("tool");
    // 参数里的**内容**一个字都不留
    expect(JSON.stringify(t)).not.toContain("秘密");
    // 参数本身也不留成一个对象（没有 arguments 这个键）
    expect(JSON.stringify(t)).not.toContain("arguments");
    // 14.3 起留的是**派生**出来的一行：read_file 在看哪个文件（见 toolIntent）。
    // 它是指给用户看的位置，不是参数原文——所以 a.cpp 这儿是**故意**留着的。
    expect(record).toMatchObject({ intent: "a.cpp" });
  });

  it("参数里的内容不进 intent：只取白名单那一个键", () => {
    // edit_file 的 new_text 可能是整份源码，run_command 的 command 是整条命令行
    const t = feed(
      openRun(),
      "r1",
      {
        type: "tool_start",
        tool: "edit_file",
        arguments: { path: "a.cpp", old_text: "旧的秘密", new_text: "新的秘密" },
      },
      1,
    );

    expect(t.runs[0]?.events[0]).toMatchObject({ intent: "a.cpp" });
    expect(JSON.stringify(t)).not.toContain("旧的秘密");
    expect(JSON.stringify(t)).not.toContain("新的秘密");
  });

  it("不认识的工具不写 intent（宁可不写，也不摊开参数）", () => {
    const t = feed(
      openRun(),
      "r1",
      { type: "tool_start", tool: "run_command", arguments: { command: "cat /etc/passwd" } },
      1,
    );

    expect(t.runs[0]?.events[0]).toMatchObject({ intent: null, family: "run" });
    expect(JSON.stringify(t)).not.toContain("passwd");
  });
});

describe("终态", () => {
  it("completed：状态、回答、结束时间都落下来", () => {
    const t = feed(
      openRun(),
      "r1",
      { type: "completed", answer: "# 好了\n\n算出来是 2", sessionId: "s1", runId: "run-1" },
      1,
      T0 + 5_000,
    );

    expect(t.runs[0]).toMatchObject({ phase: "completed", answer: "# 好了\n\n算出来是 2", endedAt: T0 + 5_000 });
    expect(displayState(t)).toBe("completed");
    expect(currentRun(t)).toBeNull();
    expect(runStats(t.runs[0]!, T0 + 99_000).durationSeconds).toBe(5); // 结束后用 endedAt，不跟着 now 涨
  });

  it("cancelled / timed_out：状态照搬，说明留着", () => {
    const cancelled = feed(
      openRun(),
      "r1",
      { type: "stopped", status: "cancelled", message: "已取消", sessionId: "s1", runId: "run-1" },
      1,
    );
    const timedOut = feed(
      openRun(),
      "r1",
      { type: "stopped", status: "timed_out", message: "超时了", sessionId: "s1", runId: "run-1" },
      1,
    );

    expect(cancelled.runs[0]).toMatchObject({ phase: "cancelled", message: "已取消" });
    expect(timedOut.runs[0]).toMatchObject({ phase: "timed_out", message: "超时了" });
  });

  it("error：记成 failed，且不把 Python 异常的原文摆出来", () => {
    const t = feed(
      openRun(),
      "r1",
      {
        type: "error",
        message: "KeyError: 'OPENAI_API_KEY' (/home/me/secret.py)",
        sessionId: "s1",
        runId: "run-1",
      },
      1,
    );

    expect(t.runs[0]?.phase).toBe("failed");
    expect(t.runs[0]?.message).not.toContain("KeyError");
    expect(t.runs[0]?.message).not.toContain("/home/me");
  });

  it("后端手写的那两种说明照原样显示", () => {
    const t = feed(
      openRun(),
      "r1",
      { type: "error", message: "调用模型失败：连不上", sessionId: "s1", runId: "run-1" },
      1,
    );
    expect(t.runs[0]?.message).toBe("调用模型失败：连不上");
  });

  it("终态之后再来事件，这一轮一个字段都不改", () => {
    const done = feed(openRun(), "r1", { type: "completed", answer: "好了", sessionId: "s1", runId: "run-1" }, 1);
    const after = feed(done, "r1", { type: "tool_start", tool: "晚到的工具", arguments: {} }, 2);

    expect(after).toBe(done); // 连对象都没换（界面上不会白重渲染一次）
  });

  it("不认识的轮次 key 喂进来，原样返回", () => {
    const t = openRun();
    expect(feed(t, "没有这一轮", START(), 1)).toBe(t);
  });

  it("terminalStatus 只认三条终态", () => {
    expect(terminalStatus({ type: "completed", answer: "", sessionId: "", runId: "" })).toBe("completed");
    expect(
      terminalStatus({ type: "stopped", status: "timed_out", message: "", sessionId: "", runId: "" }),
    ).toBe("timed_out");
    expect(terminalStatus({ type: "error", message: "", sessionId: "", runId: "" })).toBe("failed");
    expect(terminalStatus(START())).toBeNull();
  });
});

describe("重复帧", () => {
  it("同一帧喂两遍，只记一条", () => {
    const once = feed(openRun(), "r1", { type: "tool_start", tool: "grep", arguments: {} }, 1);
    const twice = feed(once, "r1", { type: "tool_start", tool: "grep", arguments: {} }, 1);

    expect(twice.runs[0]?.events).toHaveLength(1);
    expect(twice.runs[0]?.toolCount).toBe(1);
    expect(twice).toBe(once);
  });

  it("整段流被应用两遍，结果和一遍一样", () => {
    const events: RunEvent[] = [
      START(),
      { type: "llm_start", model: "m", attempt: 1, maxAttempts: 1 },
      { type: "llm_end", model: "m", success: true, durationSeconds: 1, attemptCount: 1, totalTokens: 5, error: null },
      { type: "tool_start", tool: "grep", arguments: {} },
      RESULT("grep", { durationSeconds: 0.5 }),
      { type: "completed", answer: "好了", sessionId: "s1", runId: "run-1" },
    ];

    const play = (t: Transcript): Transcript =>
      events.reduce((acc, event, index) => feed(acc, "r1", event, index + 1), t);

    // 事件的 id 是自增的（同一个进程里连绵不断），所以比的时候把它去掉——
    // 要比的是「记了什么」，不是「第几个记的」
    const noIds = (t: Transcript) =>
      t.runs.map((run) => ({
        ...run,
        events: run.events.map(({ id: _id, ...rest }) => rest),
      }));

    const once = play(openRun());
    const twice = play(play(openRun()));

    expect(noIds(twice)).toEqual(noIds(once));
    expect(twice.runs[0]?.events).toHaveLength(2); // 一条模型、一条工具（start / completed 不长过程）
    expect(twice.runs[0]).toMatchObject({ toolCount: 1, llmCount: 1, phase: "completed" });
  });
});

describe("前端自己判定的结束", () => {
  it("finishRun：还在跑的工具标成失败，不留一个永远转圈的圈", () => {
    const running = feed(openRun(), "r1", { type: "tool_start", tool: "grep", arguments: {} }, 1);
    const failed = finishRun(running, "r1", "failed", "流断了", T0 + 3_000);

    expect(failed.runs[0]).toMatchObject({ phase: "failed", message: "流断了", endedAt: T0 + 3_000 });
    expect(failed.runs[0]?.events[0]).toMatchObject({ state: "fail" });
  });

  it("已经结束的一轮，finishRun 不再改（终态是定死的）", () => {
    const done = feed(openRun(), "r1", { type: "completed", answer: "好了", sessionId: "s1", runId: "run-1" }, 1);
    expect(finishRun(done, "r1", "failed", "晚了", T0 + 9)).toBe(done);
  });

  it("取消的说明挂在那一轮下面", () => {
    const t = addNotice(openRun(), "r1", "已发出取消", "info", T0 + 100);
    expect(t.runs[0]?.events[0]).toMatchObject({ kind: "notice", text: "已发出取消", level: "info" });
  });
});

describe("刷新之后不能有假的「运行中」", () => {
  it("running → interrupted，并说明后端可能还在跑", () => {
    const before = feed(openRun(), "r1", { type: "llm_start", model: "m", attempt: 1, maxAttempts: 1 }, 1);
    const after = markInterrupted(before, T0 + 60_000);

    expect(after.runs[0]?.phase).toBe("interrupted");
    expect(after.runs[0]?.endedAt).toBe(T0 + 60_000);
    expect(after.runs[0]?.message).toContain("断线不等于取消");
    expect(currentRun(after)).toBeNull();
    expect(displayState(after)).toBe("interrupted");
  });

  it("没有在跑的就不动（连对象都不换）", () => {
    const done = feed(openRun(), "r1", { type: "completed", answer: "好了", sessionId: "s1", runId: "run-1" }, 1);
    expect(markInterrupted(done, T0 + 1)).toBe(done);
  });

  it("只有那一轮受影响，别的轮次原样", () => {
    let t = openRun("任务甲", "r1");
    t = feed(t, "r1", { type: "completed", answer: "甲的答案", sessionId: "s1", runId: "run-1" }, 1);
    t = beginRun(t, "任务乙", T0 + 10, "r2");

    const after = markInterrupted(t, T0 + 20);
    expect(after.runs[0]?.phase).toBe("completed");
    expect(after.runs[1]?.phase).toBe("interrupted");
  });
});

describe("上限：丢得诚实", () => {
  it(`超过 ${MAX_RUNS} 轮丢最旧的`, () => {
    let t = emptyTranscript();
    for (let index = 0; index < MAX_RUNS + 5; index += 1) {
      t = beginRun(t, `第 ${index} 轮`, T0 + index, `r${index}`);
    }

    expect(t.runs).toHaveLength(MAX_RUNS);
    expect(t.runs[0]?.task).toBe("第 5 轮");
    expect(t.runs[MAX_RUNS - 1]?.task).toBe(`第 ${MAX_RUNS + 4} 轮`);
  });

  it(`过程超过 ${MAX_EVENTS} 条只留最近的，但计数照旧准`, () => {
    let t = openRun();
    for (let index = 0; index < MAX_EVENTS + 10; index += 1) {
      t = feed(t, "r1", { type: "tool_start", tool: `t${index}`, arguments: {} }, index + 1);
    }

    expect(t.runs[0]?.events).toHaveLength(MAX_EVENTS);
    expect(t.runs[0]?.events[0]).toMatchObject({ tool: "t10" }); // 最早那批没了
    expect(t.runs[0]?.toolCount).toBe(MAX_EVENTS + 10); // 但次数没骗人
    expect(t.runs[0]?.eventCount).toBe(MAX_EVENTS + 10);
    expect(t.runs[0]?.truncated).toBe(true);
  });

  it(`回答超过 ${MAX_ANSWER} 字会截断，并说明截了`, () => {
    const long = "字".repeat(MAX_ANSWER + 100);
    const t = feed(openRun(), "r1", { type: "completed", answer: long, sessionId: "s1", runId: "run-1" }, 1);

    expect(t.runs[0]?.answer?.length).toBeLessThan(long.length);
    expect(t.runs[0]?.answer).toContain("本地记录只留了前一段");
  });
});

describe("看记录", () => {
  it("displayState：没有记录是 idle，跑着是 running，其余看最后一轮", () => {
    expect(displayState(emptyTranscript())).toBe("idle");
    expect(displayState(openRun())).toBe("running");

    const done = feed(openRun(), "r1", { type: "completed", answer: "好了", sessionId: "s1", runId: "run-1" }, 1);
    expect(displayState(done)).toBe("completed");
  });

  it("runStats：次数、时长、当前阶段、最近一次事件", () => {
    let t = openRun();
    t = feed(t, "r1", { type: "tool_start", tool: "grep", arguments: {} }, 1, T0 + 1_000);
    t = feed(t, "r1", RESULT("grep", { durationSeconds: 0.2 }), 2, T0 + 1_200);
    t = feed(t, "r1", { type: "llm_start", model: "m", attempt: 2, maxAttempts: 3 }, 3, T0 + 2_000);

    const stats = runStats(t.runs[0]!, T0 + 3_500);
    expect(stats.tools).toBe(1);
    expect(stats.llms).toBe(1);
    expect(stats.durationSeconds).toBeCloseTo(3.5, 3);
    expect(stats.phaseText).toBe("模型 m 第 2/3 次调用中");
    expect(stats.lastEventText).toBe("模型 m");
    expect(stats.lastEventAt).toBe(T0 + 2_000);
  });

  it("还在跑但一条事件都没来：说的是「已派给后端」", () => {
    const stats = runStats(openRun().runs[0]!, T0 + 500);
    expect(stats.phaseText).toBe("已派给后端，等第一条事件");
    expect(stats.lastEventText).toBe("");
  });

  it("outcomeText：六种状态都有一句话", () => {
    expect(outcomeText({ ...openRun().runs[0]!, phase: "running" })).toBe("运行中");
    expect(outcomeText({ ...openRun().runs[0]!, phase: "completed" })).toBe("已完成");
    expect(outcomeText({ ...openRun().runs[0]!, phase: "cancelled" })).toBe("已取消");
    expect(outcomeText({ ...openRun().runs[0]!, phase: "timed_out" })).toBe("已超时");
    expect(outcomeText({ ...openRun().runs[0]!, phase: "failed" })).toBe("失败");
    expect(outcomeText({ ...openRun().runs[0]!, phase: "interrupted" })).toBe("已中断");
  });

  it("latestRun 拿最后一轮", () => {
    let t = beginRun(emptyTranscript(), "甲", T0, "r1");
    t = beginRun(t, "乙", T0 + 1, "r2");
    expect(latestRun(t)?.task).toBe("乙");
    expect(latestRun(emptyTranscript())).toBeNull();
  });

  it("clearTranscript 清干净", () => {
    expect(clearTranscript().runs).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// 工具分类、位置、意图（14.3）
// ---------------------------------------------------------------------------

describe("工具分类", () => {
  it("认识的工具各归各类", () => {
    expect(toolFamily("search_code")).toBe("search");
    expect(toolFamily("search_code_chunks")).toBe("search");
    expect(toolFamily("read_file")).toBe("read");
    expect(toolFamily("edit_file")).toBe("edit");
    expect(toolFamily("call_graph")).toBe("graph");
    expect(toolFamily("run_command")).toBe("run");
    expect(toolFamily("compile")).toBe("build");
    expect(toolFamily("build")).toBe("build");
    expect(toolFamily("calculator")).toBe("compute");
  });

  it("不认识的工具是 other——不是不能显示，只是没分类", () => {
    expect(toolFamily("未来才有的工具")).toBe("other");
    expect(toolFamily("")).toBe("other");
  });

  it("记录里带着分类（界面不用自己再判一次）", () => {
    const t = feed(openRun(), "r1", { type: "tool_start", tool: "call_graph", arguments: {} }, 1);

    expect(t.runs[0]?.events[0]).toMatchObject({ family: "graph" });
  });
});

describe("位置与「在干什么」", () => {
  const LOCATIONS = [
    { path: "a.py", line: 12, endLine: null, symbol: "f", kind: "function" },
    { path: "b.py", line: 3, endLine: null, symbol: null, kind: null },
  ];

  it("结果里的位置进记录，条数按后端给的记", () => {
    let t = feed(openRun(), "r1", { type: "tool_start", tool: "search_code", arguments: {} }, 1);
    t = feed(t, "r1", RESULT("search_code", { locations: LOCATIONS }), 2);

    expect(t.runs[0]?.events[0]).toMatchObject({ locations: LOCATIONS, locationCount: 2 });
  });

  it(`一条事件只存前 ${MAX_EVENT_LOCATIONS} 条位置，但条数不骗人`, () => {
    const many = Array.from({ length: MAX_EVENT_LOCATIONS + 30 }, (_, index) => ({
      path: `f${index}.py`,
      line: index + 1,
      endLine: null,
      symbol: null,
      kind: null,
    }));

    let t = feed(openRun(), "r1", { type: "tool_start", tool: "search_code", arguments: {} }, 1);
    t = feed(t, "r1", RESULT("search_code", { locations: many }), 2);

    expect(t.runs[0]?.events[0]).toMatchObject({ locationCount: many.length });
    const stored = t.runs[0]?.events[0];
    expect(stored?.kind === "tool" ? stored.locations : []).toHaveLength(MAX_EVENT_LOCATIONS);
  });

  it("结果里没有位置就留着原来那条（read_file 的路径不会丢）", () => {
    let t = feed(openRun(), "r1", { type: "tool_start", tool: "read_file", arguments: { path: "a.cpp" } }, 1);
    expect(t.runs[0]?.events[0]).toMatchObject({ locationCount: 1 });

    t = feed(t, "r1", RESULT("read_file"), 2);

    const record = t.runs[0]?.events[0];
    expect(record?.kind === "tool" ? record.locations.map((l) => l.path) : []).toEqual(["a.cpp"]);
  });

  it("参数被后端截断过（以 … 结尾）就不当位置——那是半个路径", () => {
    const t = feed(
      openRun(),
      "r1",
      { type: "tool_start", tool: "read_file", arguments: { path: "很长的目录/一直长到被后端截断…" } },
      1,
    );

    const record = t.runs[0]?.events[0];
    expect(record?.kind === "tool" ? record.locations : []).toEqual([]);
    expect(record).toMatchObject({ intent: null });
  });

  it("只有 call_graph 的 path 模式是「链」", () => {
    const path = feed(
      openRun(),
      "r1",
      { type: "tool_start", tool: "call_graph", arguments: { function: "a", target: "b", direction: "path" } },
      1,
    );
    const callers = feed(
      openRun(),
      "r1",
      { type: "tool_start", tool: "call_graph", arguments: { function: "a", direction: "callers" } },
      1,
    );

    expect(path.runs[0]?.events[0]).toMatchObject({ chain: true });
    expect(callers.runs[0]?.events[0]).toMatchObject({ chain: false });
  });

  it("没见到 tool_start 的孤儿结果：位置照收，不造 intent、也不当链", () => {
    const t = feed(openRun(), "r1", RESULT("call_graph", { locations: LOCATIONS }), 1);

    expect(t.runs[0]?.events[0]).toMatchObject({
      intent: null,
      chain: false,
      locations: LOCATIONS,
      locationCount: 2,
    });
  });

  it("失败的那次调用：位置是空的（后端失败时不给 locations）", () => {
    let t = feed(openRun(), "r1", { type: "tool_start", tool: "search_code", arguments: {} }, 1);
    t = feed(t, "r1", RESULT("search_code", { success: false, error: "搜索炸了" }), 2);

    expect(t.runs[0]?.events[0]).toMatchObject({ state: "fail", locations: [], locationCount: 0 });
  });
});
