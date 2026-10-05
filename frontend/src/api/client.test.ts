/**
 * api/client.ts 的联调测试：**真的发 HTTP**，只是后端换成这个文件里现造的
 * 小服务器（node:http）。测的是「前端读后端的字节」这一段——fetch 发出去的
 * 请求长什么样、SSE 的字节被切成两块也读得出来、错误码变成哪句话。
 *
 * 后端本身不在这儿测（那是 Python 那边的 tests/test_api_*.py 的事），这里只
 * 钉住**契约**：请求体字段、Content-Type、Authorization 头、事件顺序。
 * 全在本机回环上，不联网、不要 API Key。
 */

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

import type { RunEvent } from "./types";

interface Recorded {
  method: string;
  url: string;
  headers: IncomingMessage["headers"];
  body: Record<string, unknown>;
}

let server: ReturnType<typeof createServer>;
let client: typeof import("./client");
const recorded: Recorded[] = [];
/** 让假后端把探活答成 403（真后端在 Origin 不在允许列表时就是这么答的）。 */
let healthForbidden = false;

/** 把请求体读完再交给处理函数（不读干净，客户端会拿到 ECONNRESET）。 */
async function readBody(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(chunk as Buffer);
  const text = Buffer.concat(chunks).toString("utf-8");
  try {
    const parsed: unknown = JSON.parse(text);
    return typeof parsed === "object" && parsed !== null
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
}

function json(response: ServerResponse, status: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  response.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  response.end(body);
}

const START = { task: "任务甲", session_id: "s1", run_id: "r1" };

function sseText(): string {
  return (
    `event: start\ndata: ${JSON.stringify(START)}\n\n` +
    `event: tool_start\ndata: ${JSON.stringify({
      tool: "read_file",
      arguments: { path: "a.cpp" },
    })}\n\n` +
    `event: tool_result\ndata: ${JSON.stringify({
      tool: "read_file",
      success: true,
      duration_seconds: 0.25,
      error: "",
    })}\n\n` +
    `event: llm_start\ndata: ${JSON.stringify({
      model: "m",
      attempt: 1,
      max_attempts: 3,
      session_id: "s1",
      run_id: "r1",
    })}\n\n` +
    `event: llm_end\ndata: ${JSON.stringify({
      model: "m",
      success: true,
      duration_seconds: 1.5,
      attempt_count: 1,
      usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      error: "",
      session_id: "s1",
      run_id: "r1",
    })}\n\n` +
    `event: completed\ndata: ${JSON.stringify({
      success: true,
      answer: "好了",
      session_id: "s1",
      run_id: "r1",
    })}\n\n`
  );
}

beforeAll(async () => {
  server = createServer((request, response) => {
    void (async () => {
      const body = await readBody(request);
      recorded.push({
        method: request.method ?? "",
        url: request.url ?? "",
        headers: request.headers,
        body,
      });
      const url = request.url ?? "";

      if (url === "/api/health") {
        if (healthForbidden) {
          // 真后端的行为：来源检查在路由之前跑，探活也会被挡（api/server.py）
          json(response, 403, {
            success: false,
            answer: "Origin 'http://localhost:5173' 不在允许列表里：……",
          });
          return;
        }
        json(response, 200, { status: "ok" });
        return;
      }

      if (url === "/api/run/stream") {
        // 两种「还没开头就拒绝」的情况，和 /api/run 一样回普通 JSON
        if (body.task === "任务乙") {
          json(response, 400, { success: false, answer: "task 不能是空的" });
          return;
        }
        if (body.task === "任务丁") {
          json(response, 403, {
            success: false,
            answer: "Origin 'http://localhost:5173' 不在允许列表里：……",
          });
          return;
        }
        if (body.session_id === "没了") {
          json(response, 404, {
            success: false,
            answer: "session 不存在或已过期：没了（session_id 由服务端在响应里发）",
          });
          return;
        }

        if (body.task === "任务丙") {
          // 发完终态就把连接掐掉（不发 FIN，读端会看到「流断了」）：
          // 终态已经到手，这次断线不该把这一轮改判成失败
          response.writeHead(200, { "Content-Type": "text/event-stream; charset=utf-8" });
          response.write(
            `event: completed\ndata: ${JSON.stringify({
              success: true,
              answer: "好了",
              session_id: "s1",
              run_id: "r1",
            })}\n\n`,
          );
          setTimeout(() => response.destroy(), 5);
          return;
        }

        const raw = sseText();
        response.writeHead(200, { "Content-Type": "text/event-stream; charset=utf-8" });
        // 故意从一帧的中间切开写：前端那边的解码器得把两块拼起来
        response.write(raw.slice(0, 40));
        setTimeout(() => {
          response.write(raw.slice(40));
          response.end();
        }, 5);
        return;
      }

      if (url === "/api/run") {
        json(response, 200, {
          success: true,
          answer: "好了",
          session_id: "s1",
          run_id: "r1",
        });
        return;
      }

      if (url === "/api/runs/cancel") {
        json(response, 200, {
          success: true,
          session_id: body.session_id,
          run_id: body.run_id ?? "r1",
          status: "running",
          cancelled: true,
        });
        return;
      }

      if (url === "/api/sessions") {
        if (body.session_id === "没了") {
          json(response, 404, {
            success: false,
            answer: "session 不存在或已过期：没了（session_id 由服务端在响应里发）",
          });
          return;
        }
        json(response, 200, { success: true, session_id: body.session_id, deleted: true });
        return;
      }

      json(response, 404, { success: false, answer: "没有这个接口" });
    })();
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("测试服务器没拿到端口");
  }

  // client.ts 在导入时就把这个地址读走了，所以要先起服务器、再改环境变量、
  // 最后才 import——顺序错了地址就是空的
  vi.stubEnv("VITE_API_BASE_URL", `http://127.0.0.1:${address.port}`);
  client = await import("./client");
});

afterAll(async () => {
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
});

describe("streamRun", () => {
  it("把 9 种事件里的 6 种按顺序读出来（一帧被切成两块也照样读）", async () => {
    const events: RunEvent[] = [];
    const controller = new AbortController();

    await client.streamRun({
      task: "任务甲",
      sessionId: null,
      token: null,
      signal: controller.signal,
      onEvent: (event) => events.push(event),
    });

    expect(events.map((event) => event.type)).toEqual([
      "start",
      "tool_start",
      "tool_result",
      "llm_start",
      "llm_end",
      "completed",
    ]);
    expect(events[0]).toEqual({
      type: "start",
      task: "任务甲",
      sessionId: "s1",
      runId: "r1",
    });
    expect(events[events.length - 1]).toEqual({
      type: "completed",
      answer: "好了",
      sessionId: "s1",
      runId: "r1",
    });
  });

  it("请求长成后端要的样子：POST + JSON + 不带 session_id 就不发这个字段", async () => {
    recorded.length = 0;
    const controller = new AbortController();

    await client.streamRun({
      task: "任务甲",
      sessionId: null,
      token: null,
      signal: controller.signal,
      onEvent: () => undefined,
    });

    const request = recorded[0];
    expect(request?.method).toBe("POST");
    expect(request?.url).toBe("/api/run/stream");
    expect(request?.headers["content-type"]).toBe("application/json");
    expect(request?.headers.authorization).toBeUndefined(); // 没填 Token 就不带
    expect(request?.body).toEqual({ task: "任务甲" });
  });

  it("填了 Token 就带 Authorization: Bearer（两头的空格不算数）", async () => {
    recorded.length = 0;
    const controller = new AbortController();

    await client.streamRun({
      task: "任务甲",
      sessionId: "s1",
      token: " s3cret-token ",
      signal: controller.signal,
      onEvent: () => undefined,
    });

    expect(recorded[0]?.headers.authorization).toBe("Bearer s3cret-token");
    expect(recorded[0]?.body).toEqual({ task: "任务甲", session_id: "s1" });
  });

  it("Token 里有中文：说清楚是它的问题，不是「连不上后端」", async () => {
    const controller = new AbortController();

    await expect(
      client.streamRun({
        task: "任务甲",
        sessionId: null,
        token: "秘密",
        signal: controller.signal,
        onEvent: () => undefined,
      }),
    ).rejects.toMatchObject({ status: null, message: expect.stringContaining("非 ASCII") });
  });

  it("终态之后连接被掐：终态照样送到，不算这一轮失败", async () => {
    const events: RunEvent[] = [];
    const controller = new AbortController();

    try {
      await client.streamRun({
        task: "任务丙",
        sessionId: null,
        token: null,
        signal: controller.signal,
        onEvent: (event) => events.push(event),
      });
    } catch {
      // 读端把这次断线报成异常是有可能的（连接没好好关）。真正要钉的是：
      // completed 已经交到调用方手里了，useRun 那边见到终态就不再改判
    }

    expect(events.map((event) => event.type)).toEqual(["completed"]);
  });

  it("400 直接抛 ApiError，message 是后端那句话", async () => {
    const controller = new AbortController();

    await expect(
      client.streamRun({
        task: "任务乙",
        sessionId: null,
        token: null,
        signal: controller.signal,
        onEvent: () => undefined,
      }),
    ).rejects.toMatchObject({ status: 400, message: "task 不能是空的" });
  });

  it("404（会话没了）也是 ApiError，状态码带着，调用方靠它认人", async () => {
    const controller = new AbortController();

    await expect(
      client.streamRun({
        task: "任务甲",
        sessionId: "没了",
        token: null,
        signal: controller.signal,
        onEvent: () => undefined,
      }),
    ).rejects.toMatchObject({ status: 404 });
  });
});

describe("其它四个接口", () => {
  it("checkHealth：后端说 ok 就是 ok", async () => {
    await expect(client.checkHealth()).resolves.toBe("ok");
  });

  it("checkHealth：403 报 blocked（后端活着，只是没把本页面加进允许列表）", async () => {
    healthForbidden = true;
    try {
      await expect(client.checkHealth()).resolves.toBe("blocked");
    } finally {
      healthForbidden = false;
    }
  });

  it("403 的话里带上要配的那一行；node 下没有 window 也不许炸", async () => {
    const controller = new AbortController();

    const error = await client
      .streamRun({
        task: "任务丁",
        sessionId: null,
        token: null,
        signal: controller.signal,
        onEvent: () => undefined,
      })
      .then(
        () => null,
        (cause: unknown) => cause,
      );

    expect(error).toMatchObject({ status: 403 });
    // 具体是哪个地址由浏览器给（测试环境没有 window，只说「本页面的地址」），
    // 但「改哪儿」必须说清楚
    expect((error as Error).message).toContain("API_CORS_ORIGINS");
  });

  it("runTask（普通入口）：响应翻成驼峰，status 缺省是 null", async () => {
    const controller = new AbortController();

    await expect(
      client.runTask("任务甲", "s1", null, controller.signal),
    ).resolves.toEqual({
      success: true,
      answer: "好了",
      sessionId: "s1",
      runId: "r1",
      status: null,
    });
  });

  it("cancelRun：run_id 原样发出去（前端不自己编）", async () => {
    recorded.length = 0;

    await expect(client.cancelRun("s1", "r1", null)).resolves.toEqual({
      cancelled: true,
      runId: "r1",
      status: "running",
    });
    expect(recorded[0]?.body).toEqual({ session_id: "s1", run_id: "r1" });
  });

  it("cancelRun：没有 run_id 时就只报会话（让后端找当前那一轮）", async () => {
    recorded.length = 0;

    await client.cancelRun("s1", null, null);

    expect(recorded[0]?.body).toEqual({ session_id: "s1" });
  });

  it("deleteSession 成功；后端说没有这条会话时抛 404", async () => {
    recorded.length = 0;

    await expect(client.deleteSession("s1", null)).resolves.toBeUndefined();
    expect(recorded[0]?.method).toBe("DELETE");

    await expect(client.deleteSession("没了", null)).rejects.toMatchObject({ status: 404 });
  });
});
