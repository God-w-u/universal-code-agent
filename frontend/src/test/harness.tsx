/**
 * 测试用的小渲染器。**故意不引 @testing-library/react**：这里只需要「挂一个
 * 组件、拿到它每次渲染后的结果、卸载」，用 React 自带的 act + react-dom 的
 * createRoot 十几行就够了，不必为此多一个依赖树。
 *
 * 只给测试用（文件名不在 *.test.* 里，所以会进 tsconfig.json 那个浏览器程序
 * 一起被类型检查——它本身也确实只用浏览器 API，不碰 node）。
 */

import { act } from "react";
import type { ReactElement } from "react";
import { createRoot } from "react-dom/client";

// React 要求：在测试环境里调 act 得先把这面旗立起来（否则每次都打一条警告）
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** 把 setState / 副作用里的微任务跑干净。全程在 act 里，React 才不会抱怨。 */
export async function flush(times = 5): Promise<void> {
  for (let index = 0; index < times; index += 1) {
    await act(async () => {
      await Promise.resolve();
    });
  }
}

export interface HookView<T> {
  /** 最近一次渲染时 hook 的返回值。 */
  readonly current: T;
  readonly container: HTMLElement;
  unmount: () => Promise<void>;
}

/** 挂一个只用 hook 的探针组件（相当于 testing-library 的 renderHook）。 */
export function mountHook<T>(use: () => T): HookView<T> {
  const box: { value: T | undefined } = { value: undefined };

  function Probe(): null {
    box.value = use();
    return null;
  }

  const view = mountDom(<Probe />);

  return {
    get current(): T {
      if (box.value === undefined) throw new Error("探针还没渲染出结果");
      return box.value;
    },
    container: view.container,
    unmount: view.unmount,
  };
}

export interface DomView {
  readonly container: HTMLElement;
  unmount: () => Promise<void>;
}

/** 挂一段 JSX，之后直接查 container 里的 DOM。 */
export function mountDom(element: ReactElement): DomView {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);

  act(() => {
    root.render(element);
  });

  return {
    container,
    unmount: async () => {
      await act(async () => {
        root.unmount();
      });
      container.remove();
    },
  };
}
