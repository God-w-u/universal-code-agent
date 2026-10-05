/// <reference types="vite/client" />

/**
 * 这个前端认的环境变量（构建时被替换成字面量，见 .env.example）。
 * 声明一下是为了让 `import.meta.env.VITE_API_BASE_URL` 有类型、拼错名字能报错。
 */
interface ImportMetaEnv {
  /** 后端地址。不写 = 用默认的 http://127.0.0.1:8000；写成空串 = 同源。 */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
