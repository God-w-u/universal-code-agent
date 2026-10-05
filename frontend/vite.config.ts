import react from "@vitejs/plugin-react";
// defineConfig 从 vitest/config 拿：它多认识一个 test 字段（Vite 自己的那个
// 不认识，会报「未知配置项」）。这个文件同时给 vite 和 vitest 用
import { defineConfig } from "vitest/config";

/**
 * 这里**故意不配 proxy**：开发服务器转发 /api 并不能绕开后端的来源检查，
 * 只会让人以为「不用配 CORS」。浏览器发 POST 时一定会带 Origin 头，转发
 * 是原样把请求（连 Origin）交给后端的——实测经代理同样是 403。
 *
 * 所以前端一律直连后端，地址由 VITE_API_BASE_URL 给（见 .env.example）；
 * 后端的 API_CORS_ORIGINS 里配上页面地址，是**唯一**要做的配置。
 */
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  preview: { port: 4173 },
  // 默认在 node 里跑（SSE 解析、记录生成这些纯函数不需要 DOM）；
  // 需要 DOM 的那几个文件在头上写 `// @vitest-environment jsdom`（见
  // Markdown.test.tsx / useRun.test.tsx）。
  test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
});
