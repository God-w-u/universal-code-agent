// @vitest-environment jsdom
/**
 * 工作台的「代码分析」那几块：工具行、位置列表、调用链、时间线。
 *
 * 钉住的是规格里那几条**不能破的分寸**：
 * - 认识了工具就分门别类，**不认识的照样显示**（不认识 ≠ 不能显示）；
 * - 位置只画后端给的那几条，**没有就是没有**——不按别的信息倒推一个出来；
 * - 路径长了截断显示，但完整值在 title 里、复制出来的也是完整的；
 * - 调用链有环不会画不完，太长会截断并说还有几跳；
 * - 「在编辑器打开」做不到就是做不到：摆一个禁用按钮，不假装能点；
 * - 时间线一行一件事，顺序就是收到的顺序。
 */

import { describe, expect, it, vi } from "vitest";

import type { RunEvent, ToolLocation } from "../api/types";
import { applyFrame, beginRun, emptyTranscript } from "../state/transcript";
import type { RunRecord, Transcript } from "../state/transcript";
import { flush, mountDom } from "../test/harness";
import { ChainView } from "./ChainView";
import { LocationList } from "./LocationList";
import { RunCard } from "./RunCard";

const T0 = 1_700_000_000_000;

function location(overrides: Partial<ToolLocation> = {}): ToolLocation {
  return { path: "src/a.cpp", line: 12, endLine: null, symbol: null, kind: null, ...overrides };
}

/** 造一轮记录：喂几帧给真模型（beginRun + applyFrame）。 */
function run(events: RunEvent[], times: number[] = []): RunRecord {
  const start = beginRun(emptyTranscript(), "任务甲", T0, "r1");
  const played = events.reduce<Transcript>(
    (acc, event, index) => applyFrame(acc, "r1", event, index + 1, times[index] ?? T0 + index * 1_000),
    start,
  );
  const record = played.runs[0];
  if (record === undefined) throw new Error("没造出记录");
  return record;
}

const TOOL = (tool: string, overrides: Record<string, unknown> = {}): RunEvent => ({
  type: "tool_start",
  tool,
  arguments: {},
  ...overrides,
});

const RESULT = (tool: string, overrides: Partial<Extract<RunEvent, { type: "tool_result" }>> = {}): RunEvent => ({
  type: "tool_result",
  tool,
  success: true,
  durationSeconds: 0.3,
  error: null,
  locations: [],
  ...overrides,
});

// ---------------------------------------------------------------------------
// 调用链
// ---------------------------------------------------------------------------

describe("ChainView：调用链", () => {
  const CHAIN: ToolLocation[] = [
    location({ path: "a.py", line: 9, endLine: 10, symbol: "top", kind: "function" }),
    location({ path: "a.py", line: 5, endLine: 6, symbol: "middle", kind: "function" }),
    location({ path: "a.py", line: 1, endLine: 2, symbol: "leaf", kind: "function" }),
  ];

  it("按链的顺序画节点，跳与跳之间有箭头", async () => {
    const view = mountDom(<ChainView locations={CHAIN} total={CHAIN.length} />);

    const names = [...view.container.querySelectorAll(".chain-name")].map((n) => n.textContent);
    expect(names).toEqual(["top", "middle", "leaf"]);
    // 箭头只在跳之间：n 个节点 n-1 个箭头（第一个节点没有上行箭头）
    expect(view.container.querySelectorAll(".chain-arrow")).toHaveLength(2);
    expect(view.container.querySelector(".chain-node .chain-where")?.textContent).toBe("a.py:9-10");
    await view.unmount();
  });

  it("环：重复的节点不再画，并说一声", async () => {
    const cyclic = [CHAIN[0]!, CHAIN[1]!, CHAIN[0]!];

    const view = mountDom(<ChainView locations={cyclic} total={cyclic.length} />);

    expect(view.container.querySelectorAll(".chain-node")).toHaveLength(2);
    expect(view.container.textContent).toContain("链里有环");
    await view.unmount();
  });

  it("自环（自己指向自己）也拦得住", async () => {
    const loop = [location({ symbol: "f", line: 1 }), location({ symbol: "f", line: 1 })];

    const view = mountDom(<ChainView locations={loop} total={loop.length} />);

    expect(view.container.querySelectorAll(".chain-node")).toHaveLength(1);
    expect(view.container.textContent).toContain("链里有环");
    await view.unmount();
  });

  it("太长：只画前 8 个，剩下的说还有几跳", async () => {
    const long = Array.from({ length: 12 }, (_, index) =>
      location({ symbol: `f${index}`, line: index + 1 }),
    );

    const view = mountDom(<ChainView locations={long} total={long.length} />);

    expect(view.container.querySelectorAll(".chain-node")).toHaveLength(8);
    expect(view.container.textContent).toContain("还有 4 跳");
    await view.unmount();
  });

  it("节点没有行号就只显示路径（不编一个行号）", async () => {
    const view = mountDom(
      <ChainView locations={[location({ path: "a.py", line: 0, symbol: "f" })]} total={1} />,
    );

    expect(view.container.querySelector(".chain-where")?.textContent).toBe("a.py");
    expect(view.container.textContent).not.toContain(":0");
    await view.unmount();
  });
});

// ---------------------------------------------------------------------------
// 位置列表
// ---------------------------------------------------------------------------

/** jsdom 没有剪贴板：装一个假的，把写进去的东西记下来。 */
function fakeClipboard(): { written: string[]; restore: () => void } {
  const written: string[] = [];
  const original = Object.getOwnPropertyDescriptor(navigator, "clipboard");
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn((text: string) => (written.push(text), Promise.resolve())) },
  });
  return {
    written,
    restore: () => {
      if (original === undefined) delete (navigator as { clipboard?: unknown }).clipboard;
      else Object.defineProperty(navigator, "clipboard", original);
    },
  };
}

describe("LocationList：位置列表", () => {
  it("一行一条：path:line + 符号 + 类型", async () => {
    const view = mountDom(
      <LocationList
        locations={[location({ path: "index/chunks.py", line: 12, endLine: 40, symbol: "ChunkIndex.build", kind: "method" })]}
        total={1}
      />,
    );

    expect(view.container.querySelector(".location-where")?.textContent).toBe("index/chunks.py:12-40");
    expect(view.container.querySelector(".location-symbol")?.textContent).toBe("ChunkIndex.build");
    expect(view.container.querySelector(".location-kind")?.textContent).toBe("method");
    await view.unmount();
  });

  it("没有行号（line = 0）就只显示路径", async () => {
    const view = mountDom(<LocationList locations={[location({ path: "a.py", line: 0 })]} total={1} />);

    expect(view.container.querySelector(".location-where")?.textContent).toBe("a.py");
    expect(view.container.textContent).not.toContain(":0");
    await view.unmount();
  });

  it("路径长了截断显示，但 title 里是完整路径", async () => {
    const long = `${"deep/".repeat(20)}main.py`;
    const view = mountDom(<LocationList locations={[location({ path: long, line: 7 })]} total={1} />);

    const where = view.container.querySelector(".location-where");
    expect(where?.textContent?.length).toBeLessThan(long.length);
    expect(where?.textContent).toContain("…");
    expect(where?.getAttribute("title")).toBe(`${long}:7`);
    await view.unmount();
  });

  it("复制路径：写进剪贴板的是完整路径，按钮回一句「已复制」", async () => {
    const clipboard = fakeClipboard();
    const view = mountDom(
      <LocationList locations={[location({ path: "src/很长的目录/a.cpp", line: 12 })]} total={1} />,
    );

    const copyPath = [...view.container.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("复制路径"),
    );
    copyPath?.click();
    await flush();

    expect(clipboard.written).toEqual(["src/很长的目录/a.cpp"]);
    expect(copyPath?.textContent).toBe("已复制");
    clipboard.restore();
    await view.unmount();
  });

  it("复制位置：`path:line` 一起复制", async () => {
    const clipboard = fakeClipboard();
    const view = mountDom(<LocationList locations={[location({ path: "a.py", line: 12 })]} total={1} />);

    const copyWhere = [...view.container.querySelectorAll("button")].find(
      (b) => b.textContent === "a.py:12",
    );
    copyWhere?.click();
    await flush();

    expect(clipboard.written).toEqual(["a.py:12"]);
    clipboard.restore();
    await view.unmount();
  });

  it("剪贴板写不进去时不撒谎：按钮不显示「已复制」", async () => {
    const clipboard = fakeClipboard();
    vi.mocked(navigator.clipboard.writeText).mockRejectedValue(new Error("没权限"));
    const view = mountDom(<LocationList locations={[location()]} total={1} />);

    view.container.querySelector("button")?.click();
    await flush();

    expect(view.container.textContent).not.toContain("已复制");
    clipboard.restore();
    await view.unmount();
  });

  it("「在编辑器打开」是明确禁用的：不假装浏览器能打开本地文件", async () => {
    const view = mountDom(<LocationList locations={[location()]} total={1} />);

    const open = [...view.container.querySelectorAll("button")].find((b) =>
      b.textContent?.includes("在编辑器打开"),
    );
    expect(open?.hasAttribute("disabled")).toBe(true);
    expect(open?.getAttribute("title")).toContain("浏览器打不开本地文件");
    await view.unmount();
  });

  it("后端给的比留着的多时，说清楚还有几处", async () => {
    const view = mountDom(<LocationList locations={[location(), location({ line: 20 })]} total={7} />);

    expect(view.container.textContent).toContain("还有 5 处");
    await view.unmount();
  });
});

// ---------------------------------------------------------------------------
// 工具行 + 时间线
// ---------------------------------------------------------------------------

describe("RunCard：工具行与时间线", () => {
  it("每条工具带类别、状态和用时", async () => {
    const view = mountDom(
      <RunCard
        run={run([TOOL("search_code_chunks", { arguments: { query: "widget" } }), RESULT("search_code_chunks")])}
        now={T0 + 3_000}
      />,
    );

    expect(view.container.querySelector(".event-name")?.textContent).toBe("search_code_chunks");
    expect(view.container.querySelector(".event-family")?.textContent).toBe("检索");
    expect(view.container.querySelector(".event-state")?.textContent).toBe("成功");
    await view.unmount();
  });

  it("认识的白名单工具显示「在干什么」，但只显示那一行", async () => {
    const view = mountDom(
      <RunCard
        run={run([
          TOOL("search_code", { arguments: { keyword: "def run", case_sensitive: "true" } }),
          RESULT("search_code"),
        ])}
        now={T0 + 3_000}
      />,
    );

    expect(view.container.querySelector(".event-intent")?.textContent).toBe("关键词 def run");
    // 别的参数（case_sensitive）不进界面
    expect(view.container.textContent).not.toContain("case_sensitive");
    await view.unmount();
  });

  it("不认识的工具照样显示：名字、成败都在，标记成「其它」", async () => {
    const view = mountDom(
      <RunCard run={run([TOOL("未来才有的工具"), RESULT("未来才有的工具", { success: false, error: "没这个工具" })])} now={T0 + 3_000} />,
    );

    expect(view.container.querySelector(".event-name")?.textContent).toBe("未来才有的工具");
    expect(view.container.querySelector(".event-family")?.textContent).toBe("其它");
    expect(view.container.querySelector(".event-state")?.textContent).toBe("失败");
    expect(view.container.textContent).toContain("没这个工具");
    await view.unmount();
  });

  it("没有位置就不长位置那一块（不造一个假的出来）", async () => {
    const view = mountDom(<RunCard run={run([TOOL("echo"), RESULT("echo")])} now={T0 + 3_000} />);

    expect(view.container.querySelector(".locations")).toBeNull();
    expect(view.container.querySelector(".chain")).toBeNull();
    expect(view.container.querySelector(".event-detail")).toBeNull();
    await view.unmount();
  });

  it("检索结果：命中条数写在行上，点开是位置列表", async () => {
    const locations = [
      location({ path: "a.py", line: 3 }),
      location({ path: "b.py", line: 8 }),
    ];
    const view = mountDom(
      <RunCard
        run={run([TOOL("search_code", { arguments: { keyword: "x" } }), RESULT("search_code", { locations })])}
        now={T0 + 3_000}
      />,
    );

    expect(view.container.querySelector(".event-hits")?.textContent).toBe("命中 2 处");
    expect(view.container.querySelectorAll(".location")).toHaveLength(2);
    await view.unmount();
  });

  it("调用图 path 模式：画成链，不画成清单", async () => {
    const locations = [
      location({ path: "a.py", line: 9, symbol: "top" }),
      location({ path: "a.py", line: 1, symbol: "leaf" }),
    ];
    const view = mountDom(
      <RunCard
        run={run([
          TOOL("call_graph", { arguments: { function: "top", target: "leaf", direction: "path" } }),
          RESULT("call_graph", { locations }),
        ])}
        now={T0 + 3_000}
      />,
    );

    expect(view.container.querySelector(".chain")).not.toBeNull();
    expect(view.container.querySelector(".locations")).toBeNull();
    expect(view.container.querySelector(".event-intent")?.textContent).toBe("top → leaf 的调用链");
    expect(view.container.querySelector(".event-hits")?.textContent).toBe("2 个节点");
    await view.unmount();
  });

  it("调用图的 callers 是清单，不画箭头", async () => {
    const view = mountDom(
      <RunCard
        run={run([
          TOOL("call_graph", { arguments: { function: "leaf", direction: "callers" } }),
          RESULT("call_graph", { locations: [location({ symbol: "middle", line: 5 })] }),
        ])}
        now={T0 + 3_000}
      />,
    );

    expect(view.container.querySelector(".chain")).toBeNull();
    expect(view.container.querySelectorAll(".location")).toHaveLength(1);
    expect(view.container.querySelector(".event-intent")?.textContent).toBe("谁调用了 leaf");
    await view.unmount();
  });

  it("read_file 只有参数里那个路径：只给路径，不给行号", async () => {
    const view = mountDom(
      <RunCard
        run={run([TOOL("read_file", { arguments: { path: "src/a.cpp" } }), RESULT("read_file")])}
        now={T0 + 3_000}
      />,
    );

    expect(view.container.querySelector(".location-where")?.textContent).toBe("src/a.cpp");
    expect(view.container.textContent).not.toContain(":0");
    await view.unmount();
  });

  it("结果里带了位置就换成结果里的（比参数推的准）", async () => {
    const view = mountDom(
      <RunCard
        run={run([
          TOOL("read_file", { arguments: { path: "src/a.cpp" } }),
          RESULT("read_file", { locations: [location({ path: "src/b.cpp", line: 4 })] }),
        ])}
        now={T0 + 3_000}
      />,
    );

    const wheres = [...view.container.querySelectorAll(".location-where")].map((n) => n.textContent);
    expect(wheres).toEqual(["src/b.cpp:4"]);
    await view.unmount();
  });

  it("时间线：按收到的顺序排，每条都有时间", async () => {
    // 三次调用：tool_result 是**补到**它那条 tool_start 上的（不是新的一行），
    // 所以六帧是三条行——时间取的是这一行开头那次（工具开始跑的时候）
    const view = mountDom(
      <RunCard
        run={run(
          [
            TOOL("echo"),
            RESULT("echo"),
            TOOL("run_command"),
            RESULT("run_command"),
            TOOL("search_code"),
            RESULT("search_code"),
          ],
          [T0, T0 + 1_000, T0 + 2_000, T0 + 3_000, T0 + 4_000, T0 + 5_000],
        )}
        now={T0 + 6_000}
      />,
    );

    const times = [...view.container.querySelectorAll(".event-time")].map((n) => n.textContent ?? "");
    expect(times).toHaveLength(3);
    for (const time of times) expect(time).toMatch(/^\d{2}:\d{2}:\d{2}$/);
    expect(times).toEqual([...times].sort()); // 时间只会往前走
    await view.unmount();
  });
});
