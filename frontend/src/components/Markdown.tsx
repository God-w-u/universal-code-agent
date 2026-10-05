/**
 * Agent 最终回答的 Markdown 渲染。
 *
 * 「安全」不是靠过滤字符串，而是靠**根本不走 HTML**，一共三条：
 *
 * 1. react-markdown 把 markdown 转成 **React 元素**，全程不碰 innerHTML——
 *    这个文件里没有 dangerouslySetInnerHTML，也没有别的地方有；
 * 2. **不装 rehype-raw**（那是唯一能把 markdown 里的 HTML 当 HTML 渲染的插件）。
 *    所以 `<img src=x onerror=alert(1)>` 只会**原样显示成一串字**，不会变成
 *    一个真的 img；`< b` 这种比较符也照样是文本；
 * 3. 链接地址由 react-markdown 默认的 urlTransform 过滤，`javascript:` 这种
 *    协议会被换成空串。
 *
 * 这三条每一條都有测试钉着：往 XSS payload 里灌一顿，断言渲染出来的 DOM 里
 * 没有 script / img / on* 属性（见 Markdown.test.tsx）。以后有人手滑加上
 * rehype-raw 或者换成 dangerouslySetInnerHTML，那些测试会红。
 *
 * 代码块要能读：不引第三方高亮（那是一个更大的依赖面和 XSS 面），只给
 * <pre><code> 一套等宽字体 + 横向滚动的样式（见 styles.css 的 .markdown）。
 */

import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// 表格、删除线、任务列表这些 GitHub 风格的东西：Agent 的回答里很常见
const plugins = [remarkGfm];

const components: Components = {
  // 外链一律新窗口打开，并断掉 opener（window.opener 是钓鱼的老路）
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  ),
  // 图片不自动加载：回答里一句 ![](http://某个地址/x.png) 就能在你毫不知情时
  // 向外发一个请求（带上你的 IP 和「你看过这条回答」）。所以只显示成链接，
  // 真要看得自己点一下。
  img: ({ src, alt }) => (
    <a
      href={typeof src === "string" ? src : undefined}
      target="_blank"
      rel="noreferrer noopener"
    >
      [图片] {alt ?? ""}
    </a>
  ),
};

export interface MarkdownProps {
  /** markdown 原文（Agent 的回答）。 */
  text: string;
}

export function Markdown({ text }: MarkdownProps) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={plugins} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
