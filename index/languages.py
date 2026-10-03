"""认语言：文件后缀 → 语言 id。

Project Index 给每个文件记一个 `language` 就靠这张表。它只回答「这是什么
语言」，不回答「怎么分析它」——结构提取还是 tools/code_summary.py 那套
（Python 走 ast，C/C++ 走 tools/cpp.py 的逐行扫描），本阶段两者还没接上。

加一门语言，第一步就是往 LANGUAGE_BY_SUFFIX 里加后缀；等哪天要给它抽结构，
在 code_summary 那边加提取器就行，索引这边不用动。

id 一律小写短名（"python" / "cpp" / …），和 agent/verify.py 里的
SOURCE / HEADER / PYTHON 一样：这是代码之间用的标识，不是给模型看的文本。

两处刻意的地方：

- **C 和 C++ 合成一门（"cpp"）**：结构提取本来就不分（同一个 cpp.extract
  吃 .c / .h / .cpp / .hpp），没有 Clang 也真分辨不出来。要分开的话，
  改这张表是第一步，第二步才是提取器。
- **Java / Go / Rust / JS / TS 现在只是"认得出名字"**：索引里照样记着
  （后面按语言筛文件、切块都靠它），只是还没有结构提取器。它们写在这里，
  就是给后面几个阶段留的入口。
"""

from pathlib import Path

# 后缀认不出来时用的 id
UNKNOWN_LANGUAGE = "unknown"

# 后缀（小写，含点）→ 语言 id
LANGUAGE_BY_SUFFIX = {
    # 已经有结构提取器的两种
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".inl": "cpp",
    # 认得出、还没法抽结构的后续语言
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


def language_of(path: Path) -> str:
    """按后缀认语言；认不出来就是 UNKNOWN_LANGUAGE。

    大小写不敏感：Windows 上 `MAIN.PY` 和 `main.py` 是同一类文件，
    不该一个认得、一个不认得。没有后缀（Makefile）和点开头的文件
    （.gitignore）都自然落到"认不出来"，不用特判。
    """
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), UNKNOWN_LANGUAGE)
