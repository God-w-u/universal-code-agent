"""tools/pyast.py 测试：Python 结构提取。

ast 给的行号是准的，所以这里的重点是「怎么走这棵树」：类和函数的层级、
类体里的函数算方法、装饰器算进定义范围、if / try / match 这些包在中间的
语句不算一层。全部在字符串上跑，不联网。
"""

import unittest

from tools.pyast import Definition, extract


def brief(source: str) -> list[tuple[str, str, int, int, int, str | None, bool]]:
    """提取结果压成好比较的元组。"""
    return [
        (item.kind, item.name, item.start, item.end, item.depth, item.parent, item.is_async)
        for item in extract(source)
    ]


class ModuleLevelTest(unittest.TestCase):
    def test_a_class(self):
        self.assertEqual(
            brief("class Alpha:\n    pass\n"),
            [("class", "Alpha", 1, 2, 0, None, False)],
        )

    def test_a_function(self):
        self.assertEqual(
            brief("def run():\n    return 1\n"),
            [("function", "run", 1, 2, 0, None, False)],
        )

    def test_an_async_function(self):
        self.assertEqual(
            brief("async def go():\n    pass\n"),
            [("function", "go", 1, 2, 0, None, True)],
        )

    def test_an_end_line_covers_the_whole_body(self):
        source = "def run():\n    a = 1\n    return a\n\n\nx = 2\n"

        self.assertEqual(extract(source)[0].end, 3)

    def test_an_empty_file_has_nothing(self):
        self.assertEqual(extract(""), [])

    def test_a_file_without_definitions_has_nothing(self):
        self.assertEqual(extract("x = 1\nprint(x)\n"), [])

    def test_definitions_come_in_source_order(self):
        source = "def b():\n    pass\n\n\nclass A:\n    pass\n\n\ndef a():\n    pass\n"

        self.assertEqual(
            [item.name for item in extract(source)], ["b", "A", "a"]
        )


class ClassBodyTest(unittest.TestCase):
    def test_a_function_directly_in_a_class_is_a_method(self):
        source = "class Alpha:\n    def run(self):\n        pass\n"

        self.assertEqual(
            brief(source),
            [
                ("class", "Alpha", 1, 3, 0, None, False),
                ("method", "run", 2, 3, 1, "Alpha", False),
            ],
        )

    def test_a_function_inside_a_method_is_a_function(self):
        # 方法里套的函数不是方法：它不在类体里
        source = "class Alpha:\n    def run(self):\n        def helper():\n            pass\n"

        self.assertEqual(brief(source)[-1][:2], ("function", "helper"))

    def test_a_class_inside_a_class_keeps_the_chain(self):
        source = "class Outer:\n    class Inner:\n        pass\n"

        self.assertEqual(
            brief(source),
            [
                ("class", "Outer", 1, 3, 0, None, False),
                ("class", "Inner", 2, 3, 1, "Outer", False),
            ],
        )

    def test_a_class_inside_a_method_brings_back_methods(self):
        # 函数里定义的类，类体里的函数照样是方法
        source = (
            "def outer():\n"
            "    class Inner:\n"
            "        def run(self):\n"
            "            pass\n"
        )

        self.assertEqual(
            brief(source),
            [
                ("function", "outer", 1, 4, 0, None, False),
                ("class", "Inner", 2, 4, 1, "outer", False),
                ("method", "run", 3, 4, 2, "outer.Inner", False),
            ],
        )

    def test_an_async_method_is_still_a_method(self):
        source = "class Alpha:\n    async def go(self):\n        pass\n"

        self.assertEqual(brief(source)[1][:2], ("method", "go"))
        self.assertTrue(brief(source)[1][-1])


class WrappingStatementTest(unittest.TestCase):
    """if / try / with / match 本身不是定义，但里面可能包着定义。"""

    def test_a_definition_inside_if_is_found(self):
        source = "if True:\n    def run():\n        pass\n"

        self.assertEqual(brief(source), [("function", "run", 2, 3, 0, None, False)])

    def test_a_definition_inside_if_is_not_nested_deeper(self):
        source = "class Alpha:\n    if True:\n        def run(self):\n            pass\n"

        # 层级还是 1（在类里），不是 2：if 不占一层
        self.assertEqual(brief(source)[1][:2], ("method", "run"))
        self.assertEqual(brief(source)[1][4], 1)

    def test_a_definition_inside_try_and_except_is_found(self):
        source = (
            "try:\n"
            "    pass\n"
            "except ValueError:\n"
            "    def handler():\n"
            "        pass\n"
            "finally:\n"
            "    def cleanup():\n"
            "        pass\n"
        )

        self.assertEqual(
            [item.name for item in extract(source)], ["handler", "cleanup"]
        )

    def test_a_definition_inside_match_case_is_found(self):
        # ast 把 case 子句建模成独立节点，不单独走进去就会漏
        source = (
            "match command:\n"
            "    case 'go':\n"
            "        def go():\n"
            "            pass\n"
            "    case _:\n"
            "        def other():\n"
            "            pass\n"
        )

        self.assertEqual([item.name for item in extract(source)], ["go", "other"])

    def test_a_definition_inside_with_is_found(self):
        source = "with open('a') as handle:\n    def run():\n        pass\n"

        self.assertEqual([item.name for item in extract(source)], ["run"])

    def test_else_and_for_bodies_are_walked_too(self):
        source = (
            "for item in items:\n"
            "    def per_item():\n"
            "        pass\n"
            "else:\n"
            "    def after():\n"
            "        pass\n"
        )

        self.assertEqual([item.name for item in extract(source)], ["per_item", "after"])


class DecoratorTest(unittest.TestCase):
    def test_a_decorated_definition_starts_at_the_decorator(self):
        # 装饰器属于这个定义：切块从装饰器切起才是完整的一份
        source = "@app.route('/')\ndef index():\n    pass\n"

        self.assertEqual(brief(source), [("function", "index", 1, 3, 0, None, False)])

    def test_the_topmost_decorator_wins(self):
        source = "@first\n@second\ndef run():\n    pass\n"

        self.assertEqual(extract(source)[0].start, 1)

    def test_a_decorated_class_and_method(self):
        source = (
            "@register\n"
            "class Alpha:\n"
            "    @property\n"
            "    def value(self):\n"
            "        pass\n"
        )

        self.assertEqual(
            [(item.name, item.start, item.end) for item in extract(source)],
            [("Alpha", 1, 5), ("value", 3, 5)],
        )

    def test_a_definition_without_decorators_starts_at_its_own_line(self):
        source = "def run():\n    pass\n"

        self.assertEqual(extract(source)[0].start, extract(source)[0].end - 1)


class BadInputTest(unittest.TestCase):
    def test_a_syntax_error_is_raised_not_swallowed(self):
        # 跳过还是整件事失败由调用方决定，pyast 只如实上报
        with self.assertRaises(SyntaxError):
            extract("def broken(:\n")

    def test_a_bracket_left_open_is_a_syntax_error(self):
        with self.assertRaises(SyntaxError):
            extract("class Alpha:\n    def run(self):\n        pass\n(\n")

    def test_a_definition_is_a_plain_tuple_shaped_record(self):
        item = extract("def run():\n    pass\n")[0]

        self.assertIsInstance(item, Definition)
        self.assertEqual(item, ("function", "run", 1, 2, 0, None, False))


if __name__ == "__main__":
    unittest.main()
