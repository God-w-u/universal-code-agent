"""读文本文件的共用逻辑：有上限的读取 + 编码回退。

read_file 和 search_code 都要这套东西，所以抽出来共用。
"""

import codecs
from pathlib import Path

from tools.base import ToolError

# 按顺序尝试的编码。utf-8-sig 同时覆盖带 BOM 和不带 BOM 的 UTF-8；
# gbk 是中文 Windows 上常见的源码编码。
ENCODINGS = ("utf-8-sig", "gbk")


def read_bytes(path: Path, max_bytes: int) -> tuple[bytes, bool]:
    """读文件开头至多 max_bytes 字节，返回 (内容, 后面是否还有更多)。

    只读开头，是为了避免把超大文件整个读进内存。
    """
    with path.open("rb") as handle:
        # 多读 1 字节，用来判断文件到底读完了没有
        raw = handle.read(max_bytes + 1)
    return raw[:max_bytes], len(raw) > max_bytes


def decode_with_encoding(raw: bytes, name: str) -> tuple[str, str]:
    """把字节按常见编码解成文本，并告诉调用方用的是哪一种。

    需要知道编码的是**写回文件**的工具（edit_file）：读的时候是挨个试出来的，
    写回去的时候不能靠猜——GBK 的文件按 UTF-8 写回去，整个文件的中文就全毁了。
    只读的工具不用管这件事，用下面的 decode_text 就行。

    调不开的（二进制、两种编码都不是）抛 ToolError。
    """
    if b"\x00" in raw:
        raise ToolError(
            f"{name!r} 含 NUL 字节，不是 UTF-8/GBK 文本"
            f"（可能是二进制文件，或 UTF-16 等其它编码）"
        )

    for encoding in ENCODINGS:
        try:
            # final=False：字节正好被截断在字符中间时，既不报错也不吐出半个字符
            decoder = codecs.getincrementaldecoder(encoding)()
            text = decoder.decode(raw, final=False)
        except UnicodeDecodeError:
            continue

        if encoding == "utf-8-sig" and not raw.startswith(codecs.BOM_UTF8):
            # 没 BOM 的 UTF-8 要报 utf-8，不能报 utf-8-sig：两者解码一样，
            # 但 utf-8-sig 的**编码器**会自作主张在开头加一个 BOM。
            # 报错了这一个名字，编辑工具就会给每个改过的文件都塞三个字节。
            return text, "utf-8"
        return text, encoding

    raise ToolError(f"无法用 {' 或 '.join(ENCODINGS)} 解码 {name!r}，可能是二进制文件")


def decode_text(raw: bytes, name: str) -> str:
    """把字节按常见编码解成文本；不是可读文本就抛 ToolError。

    调用方只想跳过解不开的文件时（比如搜索），捕获 ToolError 即可。
    """
    return decode_with_encoding(raw, name)[0]


def split_lines(text: str) -> list[str]:
    """把文本按行切开，用来数行号。

    read_file 和 search_code 都要数行号，两边**必须数出一样的结果**——
    一边说某处在第 42 行、另一边说第 40 行，模型就会被带到沟里。
    有测试钉着这条：tests/test_search_code.py 的 test_line_numbers_agree_with_read_file。

    所以这里的规矩是：
    - 用 split("\\n") 而不是 splitlines()：后者把 \\x0c 这类字符也当换行，
      而换页符在老的 C 代码里是当分页用的，断在那儿行号就和编辑器对不上；
    - 先把 CRLF 归一成 LF，行尾就不会挂着一个看不见的 \\r；
    - 末尾那个空串不算一行（文件以换行结尾时 split 会多给它一个）。

    单独一个 \\r（很老的 Mac 换行）不切开——几乎见不到了，而且 search_code
    也不切它，跟着它走两边才一致。
    """
    lines = text.replace("\r\n", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines
