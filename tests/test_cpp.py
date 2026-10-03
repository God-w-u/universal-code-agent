"""tools/cpp.py 测试：C/C++ 结构提取。

这是逐行扫出来的近似结果，「认不出来」是允许的——所以这里钉的是另一面：
认出来的必须对（行号、层级、签名的样子），畸形输入只能让结果少一点，
不能崩、不能卡住、不能把不相干的语句报成结构。

全部在字符串上跑，不联网。
"""

import unittest

from tools.cpp import MAX_SIGNATURE, extract, spans


def rows(source: str) -> list[tuple[int, int, str]]:
    """提取结果转成 (行号, 层级, 文本)。"""
    return [(entry.line, entry.depth, entry.text) for entry in extract(source)]


def texts(source: str) -> list[str]:
    return [entry.text for entry in extract(source)]


class IncludeTest(unittest.TestCase):
    def test_angle_brackets(self):
        self.assertEqual(rows("#include <vector>\n"), [(1, 0, "#include <vector>")])

    def test_quotes(self):
        self.assertEqual(rows('#include "a/b.h"\n'), [(1, 0, '#include "a/b.h"')])

    def test_spacing_is_tolerated(self):
        self.assertEqual(rows("#  include   <map>\n"), [(1, 0, "#include <map>")])

    def test_a_trailing_comment_is_dropped(self):
        self.assertEqual(rows("#include <set>  // 集合\n"), [(1, 0, "#include <set>")])

    def test_the_line_number_is_the_directive_line(self):
        self.assertEqual(rows("\n\n#include <cstdio>\n"), [(3, 0, "#include <cstdio>")])

    def test_other_directives_are_not_reported(self):
        self.assertEqual(rows("#pragma once\n#define MAX 10\n#ifdef __cplusplus\n"), [])

    def test_a_brace_in_a_macro_does_not_open_a_block(self):
        # #define 里的花括号不是代码里的花括号。数进去的话，后面每一条的层级
        # 都会错，而且这段宏结束时还会多弹一次栈
        source = "#define BEGIN {\nvoid after(void);\n"

        self.assertEqual(rows(source), [(2, 0, "after(void)")])

    def test_a_continuation_line_belongs_to_the_macro(self):
        # 续行的第二行不以 # 开头，但同样是宏的一部分，一样不能参与扫描
        source = (
            "#define LOOP(x) \\\n"
            "    for (int i = 0; i < (x); ++i) { \\\n"
            "    }\n"
            "int after(void);\n"
        )

        self.assertEqual(rows(source), [(4, 0, "after(void)")])


class NamespaceTest(unittest.TestCase):
    def test_named(self):
        self.assertEqual(rows("namespace app {\n}\n"), [(1, 0, "namespace app")])

    def test_anonymous(self):
        self.assertEqual(rows("namespace {\n}\n"), [(1, 0, "namespace (匿名)")])

    def test_shorthand_nesting(self):
        self.assertEqual(rows("namespace a::b::c {\n}\n"), [(1, 0, "namespace a::b::c")])

    def test_inline(self):
        self.assertEqual(rows("inline namespace v1 {\n}\n"), [(1, 0, "namespace v1")])

    def test_an_alias_is_not_a_definition(self):
        # `namespace fs = std::filesystem;` 后面跟的是 =，不是 { 或 ;
        self.assertEqual(rows("namespace fs = std::filesystem;\n"), [])

    def test_things_inside_are_one_level_deeper(self):
        self.assertEqual(
            rows("namespace app {\nclass A {\n};\n}\n"),
            [(1, 0, "namespace app"), (2, 1, "class A")],
        )


class AggregateTest(unittest.TestCase):
    def test_class(self):
        self.assertEqual(rows("class A {\n};\n"), [(1, 0, "class A")])

    def test_struct(self):
        self.assertEqual(rows("struct S {\n};\n"), [(1, 0, "struct S")])

    def test_union(self):
        self.assertEqual(rows("union U {\n};\n"), [(1, 0, "union U")])

    def test_enum(self):
        self.assertEqual(rows("enum Color { Red };\n"), [(1, 0, "enum Color")])

    def test_enum_class(self):
        # enum 得排在 class 前面认：`enum class Color` 里也有一个 class
        self.assertEqual(rows("enum class Color { Red };\n"), [(1, 0, "enum class Color")])

    def test_forward_declaration(self):
        self.assertEqual(rows("class Foo;\n"), [(1, 0, "class Foo")])

    def test_base_class_and_final_are_not_part_of_the_name(self):
        self.assertEqual(rows("class A final : public B {\n};\n"), [(1, 0, "class A")])

    def test_a_template_parameter_is_not_a_class(self):
        # template <class T> 里的 T 后面跟的是 >，不是 { : 或者 final
        self.assertEqual(rows("template <class T>\nclass Vec {\n};\n"), [(2, 0, "class Vec")])

    def test_an_elaborated_type_is_not_a_definition(self):
        # `struct Point p;` 里的 Point 后面跟的是变量名，不是定义
        self.assertEqual(rows("struct Point p;\n"), [])

    def test_a_friend_declaration_is_not_reported(self):
        # friend 声明的不是自己的成员，报出来会让人以为类里有这么个方法
        self.assertEqual(rows("class A {\nfriend class B;\n};\n"), [(1, 0, "class A")])

    def test_a_nested_class_is_one_level_deeper(self):
        self.assertEqual(
            rows("class Outer {\nclass Inner {\n};\n};\n"),
            [(1, 0, "class Outer"), (2, 1, "class Inner")],
        )


class FunctionTest(unittest.TestCase):
    def test_a_free_function(self):
        self.assertEqual(rows("int main(void) {\n}\n"), [(1, 0, "main(void)")])

    def test_a_declaration(self):
        self.assertEqual(rows("void init(void);\n"), [(1, 0, "init(void)")])

    def test_a_method(self):
        self.assertEqual(
            rows("class A {\nvoid run();\n};\n"),
            [(1, 0, "class A"), (2, 1, "run()")],
        )

    def test_an_out_of_line_definition(self):
        self.assertEqual(rows("void A::run() {\n}\n"), [(1, 0, "A::run()")])

    def test_a_constructor_and_a_destructor(self):
        self.assertEqual(
            rows("class A {\nA();\n~A();\n};\n"),
            [(1, 0, "class A"), (2, 1, "A()"), (3, 1, "~A()")],
        )

    def test_an_operator_overload(self):
        self.assertEqual(
            texts("class A {\nbool operator==(const A& o) const;\n};\n")[1],
            "operator==(const A& o) const",
        )

    def test_qualifiers_are_kept(self):
        self.assertEqual(
            rows("class A {\nvirtual void f() const noexcept override = 0;\n};\n")[1],
            (2, 1, "f() const noexcept override = 0"),
        )

    def test_the_return_type_is_left_out(self):
        # 返回类型常常是一串 std::，占地方；要看它 read_file 就行
        self.assertEqual(
            rows("static inline std::vector<int> build(int n) {\n}\n"),
            [(1, 0, "build(int n)")],
        )

    def test_parameters_spanning_lines_are_joined(self):
        self.assertEqual(
            rows("void draw(int x,\n          int y) const {\n}\n"),
            [(1, 0, "draw(int x, int y) const")],
        )

    def test_the_line_is_where_the_name_is_not_where_the_statement_starts(self):
        # template、返回类型、访问修饰符都可能另起一行，
        # 报起始行的话模型去那一行核对时什么都对不上
        self.assertEqual(
            rows("template <typename T>\nstatic inline\nvoid build(T t) {\n}\n"),
            [(3, 0, "build(T t)")],
        )

    def test_overloads_are_told_apart_by_their_parameters(self):
        # 这也是参数要留在签名里的原因：不留的话两个重载长得一模一样
        self.assertEqual(
            texts("void f(int a);\nvoid f(double a);\n"),
            ["f(int a)", "f(double a)"],
        )

    def test_a_member_initialised_with_parentheses_is_not_a_function(self):
        # `int port_(8080);` 是初始化，不是声明一个叫 port_ 的函数。
        # C++ 里没有哪种类型是以数字或者引号开头的。
        self.assertEqual(rows("class A {\nint port_(8080);\n};\n"), [(1, 0, "class A")])


class ScopeTest(unittest.TestCase):
    """函数体里的东西不算结构——不然函数体里每个函数调用都是一条「函数」。"""

    def test_a_call_inside_a_body_is_not_reported(self):
        self.assertEqual(rows("void f(void) {\n    g(1);\n}\n"), [(1, 0, "f(void)")])

    def test_a_local_type_is_not_reported(self):
        self.assertEqual(
            rows("void f(void) {\n    struct Local { int x; };\n}\n"),
            [(1, 0, "f(void)")],
        )

    def test_a_lambda_inside_a_body_is_not_reported(self):
        self.assertEqual(
            rows("void f(void) {\n    auto cb = [](int v) { return g(v); };\n}\n"),
            [(1, 0, "f(void)")],
        )

    def test_a_lambda_at_namespace_scope_is_not_reported(self):
        # 命名空间下的 λ 没有外层函数挡着，靠 `=` 认出它是代码不是声明
        self.assertEqual(rows("auto handler = [](int v) { return g(v); };\n"), [])

    def test_a_brace_initialised_variable_is_not_reported(self):
        self.assertEqual(rows("std::vector<int> v = {1, 2, 3};\n"), [])

    def test_declarations_inside_extern_c_are_still_found(self):
        # extern "C" 是个认不出的块，但里面照样是声明，不能跟着一起屏蔽
        self.assertEqual(
            rows('extern "C" {\nvoid init(void);\n}\n'), [(2, 0, "init(void)")]
        )

    def test_a_class_body_is_declaration_scope(self):
        self.assertEqual(
            rows("class A {\nvoid f();\nint port_;\n};\n"),
            [(1, 0, "class A"), (2, 1, "f()")],
        )


class CommentTest(unittest.TestCase):
    def test_commented_out_code_is_not_reported(self):
        # 注释掉的代码到处都是。当成真代码的话，报出来的全是假结构
        source = "// void old_fn(void) {\n// }\nint real(void);\n"

        self.assertEqual(rows(source), [(3, 0, "real(void)")])

    def test_a_block_comment_does_not_shift_line_numbers(self):
        self.assertEqual(
            rows("/* 一大段\n   说明\n   文字 */\nint real(void);\n"),
            [(4, 0, "real(void)")],
        )

    def test_an_unterminated_block_comment_swallows_the_rest(self):
        self.assertEqual(rows("/* 没关\nint real(void);\n"), [])

    def test_a_line_comment_after_code(self):
        self.assertEqual(rows("int f(void); // 注释里的 { 不算数\n"), [(1, 0, "f(void)")])

    def test_an_apostrophe_in_a_comment_does_not_eat_the_file(self):
        # 撇号要是被当成字符串的起始引号，后面整份文件都会被吃掉
        self.assertEqual(rows("// don't do this\nvoid f(void);\n"), [(2, 0, "f(void)")])


class StringTest(unittest.TestCase):
    def test_braces_inside_a_string_do_not_count(self):
        self.assertEqual(
            rows('const char* s = "}{";\nvoid f(void);\n'), [(2, 0, "f(void)")]
        )

    def test_an_escaped_quote_does_not_end_the_string(self):
        self.assertEqual(
            rows('const char* s = "a\\"}{";\nvoid f(void);\n'), [(2, 0, "f(void)")]
        )

    def test_a_char_literal_brace_does_not_count(self):
        self.assertEqual(rows("char c = '{';\nvoid f(void);\n"), [(2, 0, "f(void)")])

    def test_a_raw_string_is_skipped_whole(self):
        # 原始字符串里可以有花括号和引号，得整体跳过
        self.assertEqual(
            rows('const char* json = R"({"a": 1})";\nvoid f(void);\n'),
            [(2, 0, "f(void)")],
        )

    def test_strings_are_shown_as_written(self):
        # 显示用的是干净版：参数里的字符串要照原样给出来，不能变成一串空格
        self.assertEqual(
            rows('void log(const char* msg = "hi");\n'),
            [(1, 0, 'log(const char* msg = "hi")')],
        )


class LineNumberTest(unittest.TestCase):
    def test_crlf_does_not_add_a_line(self):
        self.assertEqual(rows("class A {\r\n};\r\n"), [(1, 0, "class A")])

    def test_a_form_feed_does_not_shift_line_numbers(self):
        # 换页符在老的 C 代码里是当分页用的。splitlines() 会在那儿断行，
        # 行号就和编辑器对不上了
        self.assertEqual(
            rows("void a(void);\x0c\nvoid b(void);\n"),
            [(1, 0, "a(void)"), (2, 0, "b(void)")],
        )

    def test_numbers_are_one_based(self):
        self.assertEqual(extract("void f(void);\n")[0].line, 1)


class RobustnessTest(unittest.TestCase):
    """畸形输入只能让结果少一点。崩了或者卡住了，整个工具就不能用了。"""

    def test_empty_source(self):
        self.assertEqual(rows(""), [])

    def test_only_comments(self):
        self.assertEqual(rows("// 什么都没有\n/* 也没有 */\n"), [])

    def test_a_surplus_closing_brace_does_not_unbalance_the_stack(self):
        # 栈被弹穿的话，后面每一条的层级都会错
        self.assertEqual(rows("}\nclass A {\n};\n"), [(2, 0, "class A")])

    def test_an_unclosed_block(self):
        self.assertEqual(
            rows("class A {\nvoid f();\n"), [(1, 0, "class A"), (2, 1, "f()")]
        )

    def test_an_unterminated_statement_at_the_end(self):
        self.assertEqual(rows("void f(void)"), [(1, 0, "f(void)")])

    def test_garbage_does_not_raise(self):
        for source in (
            "@@@\n###\n",
            "int ((((((;\n",
            "class\n",
            "struct {\n",
            "template <\nclass A\n",
            "void f(\n",
            ")))]]}}\n",
            "\x00\x01\x02\n",
        ):
            with self.subTest(source=source):
                extract(source)

    def test_a_very_long_statement_is_skipped_not_hung(self):
        # 正则碰上一段又长又配不上括号的文本会退化成平方级。
        # 这条钉住「太长就跳过」，否则工具会卡在这儿
        self.assertEqual(rows("int " + "a" * 20000 + "(\n"), [])

    def test_a_wall_of_parentheses_does_not_hang(self):
        extract("void f(" * 500)

    def test_a_long_parameter_list_is_truncated_in_the_display(self):
        source = "void f(" + ", ".join(f"int p{index}" for index in range(200)) + ");\n"

        entry = extract(source)[0]

        self.assertLessEqual(len(entry.text), MAX_SIGNATURE + 1)
        self.assertTrue(entry.text.endswith("…"))


class SpanTest(unittest.TestCase):
    """spans() 是 extract() 的加料版：同一条扫描，多带种类、名字和行范围。

    行范围是给 Code Chunk 切代码用的，所以这里钉的是「范围必须能圈住这个
    结构」——起始行是签名第一行（模板、返回类型都要算进去），结束行是配对
    的 `}`（没配上的算到文件末尾）。
    """

    def one(self, source: str, index: int = 0):
        return spans(source)[index]

    def test_extract_and_spans_report_the_same_scan(self):
        source = (
            "#include <vector>\n"
            "namespace app {\n"
            "class Box {\n"
            "public:\n"
            "    void put(int value);\n"
            "};\n"
            "void free(int x) { }\n"
            "}\n"
        )

        entries = [(item.line, item.depth, item.text) for item in extract(source)]
        spanned = [(item.line, item.depth, item.text) for item in spans(source)]

        self.assertEqual(entries, spanned)

    def test_kinds_and_names(self):
        source = (
            "#include <vector>\n"
            "namespace app {\n"
            "class Box { };\n"
            "struct Point { int x; };\n"
            "union U { int a; };\n"
            "enum class Color { Red };\n"
            "void run(int x) { }\n"
            "}\n"
        )

        seen = [(item.kind, item.name) for item in spans(source)]

        self.assertEqual(
            seen,
            [
                ("include", ""),
                ("namespace", "app"),
                ("class", "Box"),
                ("struct", "Point"),
                ("union", "U"),
                ("enum", "Color"),
                ("function", "run"),
            ],
        )

    def test_a_braced_class_runs_from_its_first_line_to_the_closing_brace(self):
        source = "struct Point {\n    int x;\n    int y;\n};\n"

        span = self.one(source)

        self.assertEqual((span.start, span.line, span.end), (1, 1, 4))

    def test_a_declaration_ending_in_a_semicolon_stops_there(self):
        source = "class Box {\n};\nvoid run();\n"

        self.assertEqual((self.one(source, 1).start, self.one(source, 1).end), (3, 3))

    def test_a_multiline_signature_starts_at_the_return_type(self):
        # 名字在第 2 行，但声明从第 1 行就开始了：切块要从那儿切
        source = "static inline int\nrun(\n    int x);\n"

        span = self.one(source)

        self.assertEqual((span.start, span.line, span.end), (1, 2, 3))

    def test_a_template_starts_at_the_template_line(self):
        source = "template <typename T>\nclass Box {\n};\n"

        span = self.one(source)

        self.assertEqual((span.start, span.line, span.end), (1, 2, 3))

    def test_an_access_specifier_is_not_part_of_the_declaration(self):
        # `public:` 没有分号，会和后面的声明粘成一段——它不是声明的一部分
        source = "class Box {\npublic:\n    void put(int v);\n};\n"

        span = self.one(source, 1)

        self.assertEqual((span.kind, span.name), ("function", "put"))
        self.assertEqual((span.start, span.line, span.end), (3, 3, 3))

    def test_an_unclosed_brace_runs_to_the_end_of_the_file(self):
        # 文件被截断 / #ifdef 掐掉一半：留着半空的范围，切块的人不知道切到哪儿
        source = "struct Broken {\n    int x;\n"

        self.assertEqual(self.one(source).end, 2)

    def test_an_include_is_its_own_one_line_span(self):
        span = self.one('#include "a/b.h"\n')

        self.assertEqual((span.kind, span.start, span.line, span.end), ("include", 1, 1, 1))

    def test_a_file_with_nothing_interesting_has_no_spans(self):
        self.assertEqual(spans("int x = 1;\nreturn x;\n"), [])


if __name__ == "__main__":
    unittest.main()
