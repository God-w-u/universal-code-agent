/**
 * SSE 解析的单元测试：不联网、不起服务，喂字符串就行。
 *
 * 重点在两件容易出错、又只会在线上偶发的事：
 * - **一帧被切成几块**（网络分片、代理缓冲），半个帧不能当完整的用；
 * - **事件名 / 字段对不上**时不许把页面带崩（老前端配新后端）。
 */

import { describe, expect, it } from "vitest";

import realStream from "./__fixtures__/stream-real.sse?raw";
import {
  MAX_LOCATION_KIND,
  MAX_LOCATION_PATH,
  MAX_LOCATION_SYMBOL,
  MAX_LOCATIONS,
  createFrameDecoder,
  parseFrame,
  toRunEvent,
} from "./sse";

/** 按后端的写法拼一帧（api/sse.py 的 frame()）。 */
function frame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

describe("createFrameDecoder", () => {
  it("一帧拆成三块喂进去，最后一块才吐出来", () => {
    const push = createFrameDecoder();
    const raw = frame("start", { task: "任务甲", session_id: "s1", run_id: "r1" });

    expect(push(raw.slice(0, 7))).toEqual([]);
    expect(push(raw.slice(7, 20))).toEqual([]);

    const frames = push(raw.slice(20));
    expect(frames).toHaveLength(1);
    expect(frames[0]?.event).toBe("start");
    expect(JSON.parse(frames[0]?.data ?? "")).toEqual({
      task: "任务甲",
      session_id: "s1",
      run_id: "r1",
    });
  });

  it("一块里挤着好几帧，全都吐出来", () => {
    const push = createFrameDecoder();
    const raw =
      frame("llm_start", { model: "m", attempt: 1, max_attempts: 3 }) +
      frame("llm_end", { model: "m", success: true }) +
      frame("completed", { answer: "好了" });

    expect(push(raw).map((item) => item.event)).toEqual([
      "llm_start",
      "llm_end",
      "completed",
    ]);
  });

  it("分隔符正好被切成两块也算一帧（空行还没到齐）", () => {
    const push = createFrameDecoder();

    expect(push('event: error\ndata: {"message": "断了"}\n')).toEqual([]);
    expect(push("\n").map((item) => item.event)).toEqual(["error"]);
  });

  it("CRLF 换行照样认；末尾落单的 \\r 留着等下块", () => {
    const push = createFrameDecoder();
    const raw = 'event: error\r\ndata: {"message": "断了"}\r\n\r\n';

    // 最后那个 \r 可能是被切开的 \r\n 的前半个，这一块里不该结帧
    const first = push(raw.slice(0, raw.length - 1));
    expect(first).toEqual([]);

    expect(push("\n").map((item) => item.event)).toEqual(["error"]);
  });

  it("注释行忽略；只有 event 名、没有 data 的帧不算事件", () => {
    const push = createFrameDecoder();

    expect(push(": 心跳\n\n")).toEqual([]);
    // 空心跳：有事件名但一个 data 都没有，SSE 的规矩就是不派发
    expect(push("event: completed\n\n")).toEqual([]);
    // 注释和真事件混在一帧里：注释跳过，data 照常认
    expect(
      push(': 注释\nevent: completed\ndata: {"answer":"甲"}\n\n').map((item) => item.event),
    ).toEqual(["completed"]);
  });
});

describe("parseFrame", () => {
  it("冒号后的一个空格是分隔符，不算值的一部分", () => {
    expect(parseFrame('event: completed\ndata: {"answer":"甲"}')).toEqual({
      event: "completed",
      data: '{"answer":"甲"}',
    });
  });

  it("没有 event 字段时按 SSE 默认的 message 算", () => {
    expect(parseFrame('data: {"answer":"甲"}')?.event).toBe("message");
  });

  it("data 出现多次就按行拼起来", () => {
    expect(parseFrame("data: 第一行\ndata: 第二行")?.data).toBe("第一行\n第二行");
  });

  it("认得的事件名之外的字段（id / retry）忽略，不影响解析", () => {
    expect(parseFrame("id: 7\nretry: 3000\ndata: {}")?.data).toBe("{}");
  });
});

describe("真后端发来的那段字节", () => {
  /**
   * 这份 fixture 是真服务（`python -m api` + 假模型客户端）一次请求的原始输出，
   * 一个字没改。重新抓的话：
   *
   *   curl -sN -X POST http://127.0.0.1:8000/api/run/stream \
   *     -H "Content-Type: application/json" --data-binary @body.json
   *
   * 它钉的是「后端的帧格式」这一条契约：真改了 api/sse.py 的写法，这里会红，
   * 而不是等到浏览器里默默不显示。
   */
  it("后端怎么发的，这里就怎么认（含中文和尾部空行）", () => {
    const push = createFrameDecoder();
    const events = push(realStream)
      .map((frame) => toRunEvent(frame))
      .filter((event) => event !== null);

    expect(events.map((event) => event.type)).toEqual([
      "start",
      "tool_start",
      "tool_result",
      "completed",
    ]);
    expect(events[0]).toMatchObject({ task: "算一下 1+1（中文字段也要原样回来）" });
    expect(events[1]).toMatchObject({
      tool: "calculator",
      arguments: { expression: "1+1" },
    });
    expect(events[3]).toMatchObject({ answer: "算完了：2" });
  });
});

describe("toRunEvent", () => {
  it("start / completed：把后端的蛇形字段翻成这边的驼峰", () => {
    expect(
      toRunEvent({
        event: "start",
        data: '{"task":"任务甲","session_id":"s1","run_id":"r1"}',
      }),
    ).toEqual({ type: "start", task: "任务甲", sessionId: "s1", runId: "r1" });

    expect(
      toRunEvent({
        event: "completed",
        data: '{"success":true,"answer":"好了","session_id":"s1","run_id":"r1"}',
      }),
    ).toEqual({
      type: "completed",
      answer: "好了",
      sessionId: "s1",
      runId: "r1",
    });
  });

  it("tool_start 的参数摘要是「键 → 字符串」的对象", () => {
    const event = toRunEvent({
      event: "tool_start",
      data: '{"tool":"read_file","arguments":{"path":"src/app.cpp","start_line":"1"}}',
    });

    expect(event).toEqual({
      type: "tool_start",
      tool: "read_file",
      arguments: { path: "src/app.cpp", start_line: "1" },
    });
  });

  it("tool_result：空的 error 归成 null，不显示成空字符串", () => {
    expect(
      toRunEvent({
        event: "tool_result",
        data: '{"tool":"read_file","success":true,"duration_seconds":0.25,"error":""}',
      }),
    ).toEqual({
      type: "tool_result",
      tool: "read_file",
      success: true,
      durationSeconds: 0.25,
      error: null,
      // 老后端（14.3 之前）不发 locations：没有就是空数组，不是错误
      locations: [],
    });
  });

  it("llm_end：只取总 token 数，usage 是 null 就也是 null", () => {
    expect(
      toRunEvent({
        event: "llm_end",
        data:
          '{"model":"m","success":true,"duration_seconds":1.5,"attempt_count":2,' +
          '"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},' +
          '"error":"","session_id":"s1","run_id":"r1"}',
      }),
    ).toEqual({
      type: "llm_end",
      model: "m",
      success: true,
      durationSeconds: 1.5,
      attemptCount: 2,
      totalTokens: 15,
      error: null,
    });

    // usage 是 null（接口没返回用量）时，token 数就是 null——不替它编一个 0
    expect(
      toRunEvent({
        event: "llm_end",
        data: '{"model":"m","success":true,"usage":null}',
      }),
    ).toEqual({
      type: "llm_end",
      model: "m",
      success: true,
      durationSeconds: 0,
      attemptCount: 0,
      totalTokens: null,
      error: null,
    });
  });

  it("cancelled / timed_out 合成同一种事件，靠 status 分", () => {
    const cancelled = toRunEvent({
      event: "cancelled",
      data: '{"message":"停了","session_id":"s1","run_id":"r1"}',
    });
    const timedOut = toRunEvent({
      event: "timed_out",
      data: '{"message":"超时","session_id":"s1","run_id":"r1"}',
    });

    expect(cancelled).toMatchObject({ type: "stopped", status: "cancelled" });
    expect(timedOut).toMatchObject({ type: "stopped", status: "timed_out" });
  });

  it("不认识的事件名 → null（以后后端加新事件，老前端不崩）", () => {
    expect(toRunEvent({ event: "thinking", data: '{"x":1}' })).toBeNull();
  });

  it("data 不是 JSON 对象 → null", () => {
    expect(toRunEvent({ event: "completed", data: "不是 JSON" })).toBeNull();
    expect(toRunEvent({ event: "completed", data: "[1,2,3]" })).toBeNull();
  });

  it("字段类型不对时不猜：字符串位给空串，数字位给 0", () => {
    expect(toRunEvent({ event: "tool_result", data: '{"tool":42,"success":"yes"}' })).toEqual({
      type: "tool_result",
      tool: "",
      success: false,
      durationSeconds: 0,
      error: null,
      locations: [],
    });
  });
});

// ---------------------------------------------------------------------------
// 位置信息（14.3）
// ---------------------------------------------------------------------------

/** 一条 tool_result 帧的 data（只写关心的字段）。 */
function resultData(extra: string): string {
  return `{"tool":"search_code","success":true,"duration_seconds":0.2,"error":""${extra}}`;
}

function locationsOf(data: string) {
  const event = toRunEvent({ event: "tool_result", data });
  if (event === null || event.type !== "tool_result") throw new Error("没解析成 tool_result");
  return event.locations;
}

describe("tool_result 里的位置", () => {
  it("路径、行号、起止行、符号、类型都翻过来（蛇形 → 驼峰）", () => {
    expect(
      locationsOf(
        resultData(',"locations":[{"path":"index/chunks.py","line":12,"end_line":40,"symbol":"ChunkIndex.build","kind":"method"}]'),
      ),
    ).toEqual([
      { path: "index/chunks.py", line: 12, endLine: 40, symbol: "ChunkIndex.build", kind: "method" },
    ]);
  });

  it("只有路径也行：行号 / 符号 / 类型都当没有（不编）", () => {
    expect(locationsOf(resultData(',"locations":[{"path":"a.py"}]'))).toEqual([
      { path: "a.py", line: 0, endLine: null, symbol: null, kind: null },
    ]);
  });

  it("结束行不大于起始行时不算范围（单行就是单行）", () => {
    expect(locationsOf(resultData(',"locations":[{"path":"a.py","line":9,"end_line":9}]'))).toEqual([
      { path: "a.py", line: 9, endLine: null, symbol: null, kind: null },
    ]);
  });

  it("没有路径的那一条整条丢掉（立不住的位置不当位置）", () => {
    expect(
      locationsOf(resultData(',"locations":[{"line":3},{"path":""},{"path":"ok.py","line":1}]')),
    ).toEqual([{ path: "ok.py", line: 1, endLine: null, symbol: null, kind: null }]);
  });

  it("明显不是位置的东西一律不认", () => {
    // 字符串 / 数字 / null / 数组混进来：全部丢掉，不猜
    expect(locationsOf(resultData(',"locations":["a.py:1",7,null,[]]'))).toEqual([]);
    expect(locationsOf(resultData(',"locations":"a.py:1"'))).toEqual([]); // 根本不是数组
  });

  it("行号不是正经数字就当没有行号（0，不是 NaN）", () => {
    expect(locationsOf(resultData(',"locations":[{"path":"a.py","line":"12"}]'))).toEqual([
      { path: "a.py", line: 0, endLine: null, symbol: null, kind: null },
    ]);
    expect(locationsOf(resultData(',"locations":[{"path":"a.py","line":-3}]'))).toEqual([
      { path: "a.py", line: 0, endLine: null, symbol: null, kind: null },
    ]);
  });

  it("条数封顶：后端一次给再多也只认 50 条", () => {
    const many = Array.from({ length: 80 }, (_, index) => `{"path":"f${index}.py","line":1}`);
    expect(locationsOf(resultData(`,"locations":[${many.join(",")}]`))).toHaveLength(MAX_LOCATIONS);
  });

  it("路径 / 符号 / 类型太长就截断（不让一帧撑破版面）", () => {
    const [location] = locationsOf(
      resultData(
        `,"locations":[{"path":"${"a".repeat(300)}","line":1,"symbol":"${"b".repeat(200)}","kind":"${"c".repeat(80)}"}]`,
      ),
    );

    expect(location?.path.length).toBeLessThanOrEqual(MAX_LOCATION_PATH);
    expect(location?.path.endsWith("…")).toBe(true);
    expect(location?.symbol?.length).toBeLessThanOrEqual(MAX_LOCATION_SYMBOL);
    expect(location?.kind?.length).toBeLessThanOrEqual(MAX_LOCATION_KIND);
  });

  it("老后端不发 locations：空数组，不是错误", () => {
    expect(locationsOf('{"tool":"echo","success":true,"duration_seconds":0.1,"error":""}')).toEqual([]);
  });
});
