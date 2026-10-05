// @vitest-environment jsdom
/**
 * 回答的 Markdown 渲染：**能读，但注入不进来**。
 *
 * 这里全是「往里面灌 XSS payload，然后翻 DOM」的测试。它们钉住的是那条设计：
 * 走 React 元素、不碰 innerHTML、不装 rehype-raw。哪天有人为了「支持 HTML」
 * 加上 rehype-raw 或改成 dangerouslySetInnerHTML，这些用例会红——那正是它们
 * 存在的意义。
 *
 * 另外还钉了两条「安全但不影响阅读」的行为：
 * - `<`、`>` 这种比较符在普通文字里要原样显示（Agent 讲代码时常见）；
 * - 回答里的图片**不自动加载**（不让一条回答拿你的 IP 去外面报信），只显示成链接。
 */

import { describe, expect, it } from "vitest";

import { mountDom } from "../test/harness";
import { Markdown } from "./Markdown";

/** 渲染一段 markdown，把 container 交回来（调用方自己卸载）。 */
function render(text: string): { container: HTMLElement; unmount: () => Promise<void> } {
  return mountDom(<Markdown text={text} />);
}

function xssValue(): unknown {
  return (window as unknown as Record<string, unknown>).__xss;
}

/** HTML 里绝不该出现的东西（不能执行代码，也不能自己去外面取东西）。 */
const FORBIDDEN_TAGS = ["script", "iframe", "object", "embed", "svg", "img", "form", "style", "base", "meta", "link"];

function expectNoDangerousDom(container: HTMLElement): void {
  for (const tag of FORBIDDEN_TAGS) {
    expect(container.querySelectorAll(tag), `不该出现 <${tag}>`).toHaveLength(0);
  }
  for (const element of container.querySelectorAll("*")) {
    for (const attribute of element.attributes) {
      // on* 是一类，srcdoc 是 iframe 的另一种写法；都不该有
      expect(attribute.name.startsWith("on"), `${element.tagName} 上有 ${attribute.name}`).toBe(false);
      expect(attribute.name).not.toBe("srcdoc");
    }
  }
  for (const anchor of container.querySelectorAll("a")) {
    const href = anchor.getAttribute("href") ?? "";
    expect(href.toLowerCase().startsWith("javascript:")).toBe(false);
    expect(href.toLowerCase().startsWith("data:")).toBe(false);
  }
}

const PAYLOADS: [string, string][] = [
  ["script 标签", "<script>window.__xss = 1</script>"],
  ["img + onerror", '<img src="x" onerror="window.__xss = 2">'],
  ["raw 的 a + javascript:", '<a href="javascript:window.__xss=3">点我</a>'],
  ["iframe", '<iframe src="javascript:window.__xss=4"></iframe>'],
  ["svg onload", '<svg onload="window.__xss=5"></svg>'],
  ["div onclick", '<div onclick="window.__xss=6">点我</div>'],
  ["markdown 链接里的 javascript:", "[点我](javascript:window.__xss=7)"],
  ["markdown 链接里的 data:", "[点我](data:text/html;base64,PHNjcmlwdD4=)"],
  ["markdown 图片指向外面", "![图](https://example.com/x.png)"],
  ["裸的尖括号", "如果 a < b 而且 b > c，就是对的"],
];

describe("XSS payload 灌进去", () => {
  it.each(PAYLOADS)("%s：不执行，也不产生危险的 DOM", async (name, payload) => {
    const view = render(payload);

    expect(xssValue(), `${name} 执行了`).toBeUndefined();
    expectNoDangerousDom(view.container);

    await view.unmount();
  });

  it("几种 payload 一起来也一样", async () => {
    const view = render(PAYLOADS.map(([, payload]) => payload).join("\n\n"));

    expect(xssValue()).toBeUndefined();
    expectNoDangerousDom(view.container);

    await view.unmount();
  });

  it("HTML 是当文字显示的（不吞掉，也不当标签）", async () => {
    const view = render('<script>window.__xss = 1</script>');

    // 原样摆着给人看，而不是被渲染成一个真的 script
    expect(view.container.textContent).toContain("<script>");
    await view.unmount();
  });
});

describe("正常回答要能读", () => {
  it("标题、粗体、列表、行内代码", async () => {
    const view = render(
      ["# 结论", "", "**要点**：", "", "- 第一点", "- 第二点", "", "用 `RunEvent` 表示一帧。", ""].join("\n"),
    );

    expect(view.container.querySelector("h1")?.textContent).toBe("结论");
    expect(view.container.querySelector("strong")?.textContent).toBe("要点");
    expect(view.container.querySelectorAll("li")).toHaveLength(2);
    expect(view.container.querySelector("code")?.textContent).toBe("RunEvent");
    await view.unmount();
  });

  it("代码块能读：有 pre > code，原文一行不少", async () => {
    const view = render(["```python", "def f(x):", "    return x + 1", "```", ""].join("\n"));

    const code = view.container.querySelector("pre code");
    expect(code?.textContent).toContain("def f(x):");
    expect(code?.textContent).toContain("return x + 1");
    await view.unmount();
  });

  it("表格（GitHub 风格）画得出来", async () => {
    const view = render(["| 轮次 | 状态 |", "| --- | --- |", "| 1 | 完成 |", ""].join("\n"));

    expect(view.container.querySelector("table")).not.toBeNull();
    expect(view.container.querySelectorAll("td")).toHaveLength(2);
    await view.unmount();
  });

  it("普通文字里的比较符原样显示（讲代码时常见）", async () => {
    const view = render("如果 a < b 而且 b > c，就是对的");

    expect(view.container.textContent).toContain("a < b");
    expect(view.container.textContent).toContain("b > c");
    await view.unmount();
  });

  it("普通链接：新窗口打开，并断掉 opener", async () => {
    const view = render("见 [文档](https://example.com/docs)。");

    const anchor = view.container.querySelector("a");
    expect(anchor?.getAttribute("href")).toBe("https://example.com/docs");
    expect(anchor?.getAttribute("target")).toBe("_blank");
    expect(anchor?.getAttribute("rel")).toContain("noreferrer");
    await view.unmount();
  });

  it("图片不自动加载：只留一个链接，不产生 img 请求", async () => {
    const view = render("![流程图](https://example.com/track.png)");

    expect(view.container.querySelector("img")).toBeNull();
    const anchor = view.container.querySelector("a");
    expect(anchor?.getAttribute("href")).toBe("https://example.com/track.png");
    expect(view.container.textContent).toContain("[图片]");
    expect(view.container.textContent).toContain("流程图");
    await view.unmount();
  });

  it("纯文本回答（没有 markdown）照样显示", async () => {
    const view = render("读完了，一共 3 个函数。");

    expect(view.container.textContent).toContain("读完了，一共 3 个函数。");
    await view.unmount();
  });

  it("空回答不炸", async () => {
    const view = render("");
    expect(view.container.textContent).toBe("");
    await view.unmount();
  });
});
