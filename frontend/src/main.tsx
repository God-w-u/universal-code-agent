/** 入口：把 App 挂到 index.html 的 #root 上。 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

const container = document.getElementById("root");
if (container === null) {
  // index.html 是跟这个文件一起的，缺了只可能是被改坏了：早点炸比白屏好查
  throw new Error("index.html 里找不到 #root");
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
