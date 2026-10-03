"""给 embedding 的紧凑文本（9.3）。

9.2 的探针量出来两件事：整块代码直接喂给模型，全库 45.7% 的块超过 128
token 的窗口被截断；而且路径 / 符号名 / 整段代码混在一起，语义信号被
稀释（同一条查询：整块 0.2273，只留中文 docstring 0.3265）。切块本身不能
动（chunk 的 `code` 是给 read_file / 工具看的原文），所以为 embedding
单独拼一段**紧凑文本**：

    index/symbols.py:SymbolIndex.read_verified [method]
    按索引里的路径把文件读回来，顺便核对它还是不是当初那份。
    def read_verified(self, entry: IndexedFile) -> tuple[str | None, str | None]:

        path = self.project.root / entry.path

（描述段和代码里的那一段是同一段文字，只是挪了个位置——代码开头里原来
放 docstring 的地方被掏空了。）

三段，能省的省：

- **位置行**：`path:symbol [kind]`——文件名、符号名、类型都在一行里，
  字面上的线索一个不少。
- **描述段**：Python 的 docstring（第一个三引号块），空白压平、截到
  `MAX_DOC_CHARS`。它常是整块里最像「人话」的部分。
- **代码开头**：截到 `MAX_CODE_CHARS` 字符。没有 docstring 的块（C/C++
  等）就靠它——注释风格先不猜，提取留给以后。**描述已经从代码里挖走
  的那一段要先掏空**：docstring 本来就长在代码开头里，再抄一遍等于同一
  段文字喂两遍（实测多语言模型上超窗比例反而从 46.2% 涨到 51.2%），掏
  掉之后两边各出现一次。

三条边界：

- **不改 chunk 的 code**：这里只是另拼一段字符串，`CodeChunk` 一个字段
  都没动。
- **同输入同输出**：纯字符串拼装，不取时间、不取 id、不取随机数——缓存
  key 依赖它稳定，测试钉着这一条。
- **上限都在常量里**：觉得不合适就调 `MAX_DOC_CHARS` / `MAX_CODE_CHARS`，
  换策略不用碰 scorer / Retriever；**拼法改了就把 `TEXT_STRATEGY` 加
  一版**——评估入口把它拼进缓存的命名空间，旧向量自动作废、不会拿着
  「旧拼法算出来的向量」冒充新结果。
"""

from index.chunks import CodeChunk

# 拼法版本：改了 compact_text 的规则就 +1。评估入口拿它当缓存命名空间
# 的一部分（见 index/__main__._hybrid），这样旧向量自动失效。
TEXT_STRATEGY = "compact-v1"

# 描述段的字符上限。中文在模型分词器里大约 1 字 1~2 个 token，200 字符
# 够装下一段正经的说明，又不至于把窗口全吃掉。
MAX_DOC_CHARS = 200

# 代码开头的字符上限。代码是「描述缺席时」的信号，给到够看清签名和头几
# 行就行——整段喂进去只会把语义信号稀释掉。
MAX_CODE_CHARS = 400

# Python 的三引号：先找双引号，没闭合再找单引号。
_DELIMS = ('"""', "'''")


def compact_text(chunk: CodeChunk) -> str:
    """chunk → 喂给 embedding 的紧凑文本（确定性：同输入同输出）。"""
    parts = [f"{chunk.path}:{chunk.symbol} [{chunk.kind}]"]

    description, span = _docstring(chunk)
    if description:
        parts.append(_clip(" ".join(description.split()), MAX_DOC_CHARS))

    code = _clip(_body(chunk.code, span), MAX_CODE_CHARS)
    if code:
        parts.append(code)

    return "\n".join(parts)


def _docstring(chunk: CodeChunk) -> tuple[str, tuple[int, int] | None]:
    """Python chunk 的第一个三引号块：返回（内容, 在 code 里的起止）。

    没有就返回空串 + None。刻意不写完整的解析器：找到第一个三引号、再找
    它后面的下一个同样的三引号，中间就是。docstring 里再套同样的引号这种
    极端情况认不出来——best effort，认不出顶多少一段描述。
    """
    if chunk.language != "python":
        return "", None

    for delim in _DELIMS:
        start = chunk.code.find(delim)
        if start == -1:
            continue
        end = chunk.code.find(delim, start + len(delim))
        if end != -1:
            return chunk.code[start + len(delim) : end], (start, end + len(delim))
    return "", None


def _body(code: str, span: tuple[int, int] | None) -> str:
    """代码去描述版：把 span 那一段（连同两头的三引号）掏掉，顺手去掉每行
    的行尾空白——被掏空的那些行不该再带着一截缩进。"""
    if span is not None:
        start, end = span
        code = code[:start] + code[end:]
    return "\n".join(line.rstrip() for line in code.split("\n"))


def _clip(text: str, limit: int) -> str:
    """去首尾空白；超长就截断（截完再清一次尾部的空白/换行）。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()
