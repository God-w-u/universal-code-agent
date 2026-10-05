"""SSE（Server-Sent Events）这一小块：帧怎么拼、Trace 怎么变成事件。

两件事，各管各的：

- frame()：一帧 SSE 的字节——`event: 名字` + `data: 一行 JSON` + 空行；
- TraceStreamer：实现 agent/trace.py 要的那四个观察者回调，把工具调用
  翻译成 tool_start / tool_result 帧、把模型调用（13.5）翻译成
  llm_start / llm_end 帧，往 HTTP 响应里写。

**推给前端的东西要克制**（13.2 规格第 5 条）：工具的返回正文（ToolResult
的 content）压根不进 Trace，所以也进不了这里的任何一帧；参数值、错误文本
这些可能很长的，一律截断到 SUMMARY_LIMIT。start / completed / error 三个
事件由 api/server.py 发，它们的内容（任务原文、最终回答）本来就是这一问
一答本身，和 POST /api/run 返回的是同一份东西。

**写失败不算事**：客户端断了就闭嘴（closed 一置，后面的帧全跳过）。这一轮
任务会继续跑完——没有取消机制，也不该在 HTTP 层现造一套；只是没人接了。
"""

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 只为类型检查：运行时不 import，省得 sse 和 agent / llm 之间多运行期依赖
    from agent.trace import TraceEvent
    from llm.client import LLMCallStats

# 摘要上限：参数值、错误文本最多推这么多字符。定这么小是因为它只是给前端
# 显示「在干什么」，不是给机器读的
SUMMARY_LIMIT = 120


def frame(event: str, data: dict[str, Any]) -> bytes:
    """一帧 SSE：event 行 + data 行 + 空行。

    json.dumps 出来的字符串里不会有裸换行（字符串自带的换行会被转义成 \\n），
    所以一帧就是规规矩矩的两行，读的人不用管 SSE 的续行规则。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def shorten(value: str) -> str:
    """太长就截断：错误文本里可能有整段编译输出、长路径。"""
    return value if len(value) <= SUMMARY_LIMIT else value[:SUMMARY_LIMIT] + "…"


def summarize(arguments: dict[str, Any]) -> dict[str, str]:
    """参数摘要：值一律转成字符串再截断。

    edit_file 的 content 可能是整份源码，原样推给前端既没用又违背规格
    （「不直接把完整源码推给前端」）。转成字符串只是为了显示，前端别指望
    能拿它原样调工具。
    """
    summary = {}
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        summary[str(key)] = shorten(text)
    return summary


class TraceStreamer:
    """把 Trace 的通知写成 SSE 帧。

    用法：把四个回调（on_tool_start / on_tool_end / on_llm_start /
    on_llm_end）交给 Agent.run()，自己用 send() 发 start / completed /
    error。write 是「写一帧」的出口（服务里是 wfile.write），测试里可以是
    一张列表——这样这一层不用认识 HTTP。

    session_id / run_id（13.5）：**只加在 llm_* 两个事件上**（规格要求新
    事件带这两个字段）。start / completed / error / cancelled / timed_out
    由 api/server.py 发，本来就带着；tool_start / tool_result 保持一字不差
    ——它们从 13.2 起就是这个形状，前端和几十处断言按它写的。
    """

    def __init__(
        self,
        write: Callable[[bytes], None],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._write = write
        self.closed = False
        # 这一轮的身份：llm_* 帧带上它们，前端好对号入座
        self.session_id = session_id
        self.run_id = run_id

    def send(self, event: str, data: dict[str, Any]) -> None:
        """发一帧；已经关了（对面断了）就什么也不做。"""
        if self.closed:
            return
        try:
            self._write(frame(event, data))
        except Exception:
            # 写不出去多半是客户端断开（OSError）；别的意外也一样，这条流
            # 没救了。关掉自己，别让异常顺着 Agent → ToolExecutor 的栈
            # 把这一轮任务变成失败——它还在替别人干活呢
            self.closed = True

    def on_tool_start(self, tool: str, arguments: dict[str, Any]) -> None:
        """工具要开始跑了（Agent 的观察者回调）。"""
        self.send("tool_start", {"tool": tool, "arguments": summarize(arguments)})

    def on_tool_end(self, event: "TraceEvent") -> None:
        """工具跑完了（Agent 的观察者回调）。

        **不带正文**：只给成没成、多久、错在哪（截断），要看内容让前端
        自己拿位置去 read_file——和工具层的分寸一致。
        """
        self.send(
            "tool_result",
            {
                "tool": event.tool,
                "success": event.success,
                "duration_seconds": round(event.duration_seconds, 3),
                "error": shorten(event.error) if event.error else "",
            },
        )

    def on_llm_start(self, model: str, attempt: int, max_attempts: int) -> None:
        """模型要开始答了（13.5；重试的话每次重试都来一下）。

        只给「哪个模型、第几次试、最多试几次」——prompt 一个字都不推。
        """
        self.send(
            "llm_start",
            {
                "model": model,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "session_id": self.session_id,
                "run_id": self.run_id,
            },
        )

    def on_llm_end(self, stats: "LLMCallStats") -> None:
        """模型这一跳答完了 / 彻底失败了（13.5）。

        给的还是数字：成没成、多久、试了几次、token 用量、错误摘要（截断）。
        没有 prompt、没有回复正文、更没有 key——和统计对象（llm/client.py
        的 LLMCallStats）能提供的完全是同一份东西。
        """
        self.send(
            "llm_end",
            {
                "model": stats.model,
                "success": stats.success,
                "duration_seconds": round(stats.duration_seconds, 3),
                "attempt_count": stats.attempt_count,
                "usage": stats.usage.to_dict() if stats.usage is not None else None,
                "error": shorten(stats.error) if stats.error else "",
                "session_id": self.session_id,
                "run_id": self.run_id,
            },
        )
