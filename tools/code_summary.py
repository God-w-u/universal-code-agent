"""工具：提取源码结构摘要（文件里有哪些类 / 函数 / 方法）。

Python 文件用 tools/pyast.py 的 ast；C/C++ 文件用 tools/cpp.py 的逐行扫描
（没有 Clang）。只做结构提取、不读全文，用来先看清项目骨架，省掉大量 read_file。
安全边界复用 tools/paths.py，遍历规则复用 tools/tree.py，
读取与解码复用 tools/textfile.py。
"""

from pathlib import Path

from tools import cpp, pyast
from tools.base import Tool, ToolError
from tools.paths import check_root, resolve_in_root
from tools.textfile import decode_text, read_bytes, split_lines
from tools.tree import iter_files

# 能分析的文件类型：.py 走 ast，其余走逐行扫描
SUFFIXES = (".py", ".c", ".h", ".cpp", ".hpp")


class CodeSummaryTool(Tool):
    """列出目录下 Python / C++ 文件的类与函数结构。"""

    name = "code_summary"
    description = (
        "列出目录下源码文件的结构，适合先看清项目骨架再决定读哪个文件，"
        "比逐个 read_file 省上下文。"
        "Python 文件（.py）给出每个文件有哪些类、函数、方法（含 async）。"
        "C/C++ 文件（.c/.h/.cpp/.hpp）给出 #include、namespace、"
        "class/struct/union/enum、函数声明与定义，每条都带行号"
        "（和 read_file 是同一套行号，可以直接拿来引用位置）。"
        "只返回定义名和签名，不返回函数体。"
        "C/C++ 那部分是逐行扫的近似结果：宏展开、宏生成的名字、复杂模板可能漏掉或认错。"
        "会自动跳过 .git、.venv、__pycache__ 等噪音目录和其它类型的文件。"
    )
    parameters = {
        "path": {
            "type": "string",
            "description": "要分析的目录，相对项目根目录；省略则分析整个项目",
            "required": False,
        },
    }

    # 最多分析多少个文件
    max_files = 200
    # 单个文件最多读多少字节，超过就跳过（这种体积的源码本来也不该整体分析）
    max_file_bytes = 256 * 1024
    # 最多输出多少字符，避免撑爆 LLM 上下文
    max_output_chars = 20_000

    def __init__(self, root: Path | None = None) -> None:
        self.root = check_root(root)

    def execute(self, path: str = ".") -> str:
        directory = resolve_in_root(path, self.root)

        if not directory.is_dir():
            raise ToolError(f"不是目录：{path!r}")

        try:
            files = sorted(iter_files(directory))
        except OSError as exc:
            raise ToolError(f"读取目录失败：{exc}") from exc

        sources = [item for item in files if item.suffix in SUFFIXES]
        if not sources:
            return f"（{path!r} 下没有可分析的源文件）"

        blocks = [self._describe(item) for item in sources[: self.max_files]]

        # 按整个文件为单位拼，避免把一个文件的类/函数列表切成两半
        kept: list[str] = []
        used = 0
        for block in blocks:
            if kept and used + len(block) + 2 > self.max_output_chars:
                break
            kept.append(block)
            used += len(block) + 2

        text = "\n\n".join(kept)

        notes = []
        if len(sources) > self.max_files:
            notes.append(
                f"共 {len(sources)} 个源文件，只分析了前 {self.max_files} 个"
            )
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars]
            notes.append(f"结构摘要超过 {self.max_output_chars} 个字符，只返回了开头")
        elif len(kept) < len(blocks):
            notes.append(f"共 {len(blocks)} 个文件，只显示了前 {len(kept)} 个")

        if notes:
            # 截断了就必须说清楚，否则模型会以为这就是全部
            text += "\n\n…（" + "；".join(notes) + "）"
        return text

    def _describe(self, item: Path) -> str:
        """一个文件一段：第一行是路径，后面是缩进表示层级的结构行。"""
        relative = item.relative_to(self.root).as_posix()

        try:
            raw, too_big = read_bytes(item, self.max_file_bytes)
        except OSError:
            return f"{relative}:\n  （读取失败，已跳过）"
        if too_big:
            return f"{relative}:\n  （文件过大，已跳过）"

        try:
            text = decode_text(raw, relative)
        except ToolError:
            return f"{relative}:\n  （不是文本文件，已跳过）"

        if item.suffix == ".py":
            return self._describe_python(relative, text)
        return self._describe_c(relative, text)

    def _describe_python(self, relative: str, text: str) -> str:
        try:
            definitions = pyast.extract(text)
        except (SyntaxError, ValueError, RecursionError):
            # 语法错误、ast 本身拒绝的输入、嵌套过深：都只影响这一个文件
            return f"{relative}:\n  （语法错误，已跳过）"

        if not definitions:
            return f"{relative}:\n  （没有类或函数）"
        return f"{relative}:\n" + "\n".join(
            "  " * (item.depth + 1) + _label(item) for item in definitions
        )

    def _describe_c(self, relative: str, text: str) -> str:
        """C/C++ 那一份。每条带行号，数法和宽度都跟 read_file 一样。"""
        entries = cpp.extract(text)
        if not entries:
            return f"{relative}:\n  （没有可提取的结构）"

        # 宽度按整个文件的行数算，和 read_file 一个规矩：
        # 同一个文件里同一行，两个工具报出来的编号得长得一样
        width = len(str(max(len(split_lines(text)), 1)))
        body = "\n".join(
            f"{'  ' * (entry.depth + 1)}{entry.line:>{width}}|{entry.text}"
            for entry in entries
        )
        return f"{relative}:\n{body}"


def _label(item: pyast.Definition) -> str:
    """一条定义在摘要里的写法：`class Alpha` / `async def go` / `def run`。"""
    if item.kind == "class":
        return f"class {item.name}"
    return f"{'async def' if item.is_async else 'def'} {item.name}"
