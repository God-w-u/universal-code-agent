"""Tool / ToolRegistry 测试：注册、查询、列表、执行、异常。"""

import unittest

from tools import CalculatorTool, Tool, ToolError, ToolRegistry


class EchoTool(Tool):
    """测试用的第二个工具，带一个可选参数。"""

    name = "echo"
    description = "原样返回输入文本，可重复多次。"
    parameters = {
        "text": {"type": "string", "description": "要回显的文本", "required": True},
        "times": {"type": "integer", "description": "重复次数", "required": False},
    }

    def execute(self, text, times=1):
        return text * times


class NamelessTool(Tool):
    name = ""
    description = "没有名字的工具"
    parameters = {}

    def execute(self, **kwargs):
        return ""


class ToolBaseTest(unittest.TestCase):
    def test_tool_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            Tool()

    def test_subclass_declares_interface(self):
        tool = EchoTool()
        self.assertEqual(tool.name, "echo")
        self.assertTrue(tool.description)
        self.assertIn("text", tool.parameters)


class ToolRegistryRegisterTest(unittest.TestCase):
    def test_register_and_count(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())

        self.assertEqual(len(registry), 1)
        self.assertIn("calculator", registry)

    def test_register_via_constructor(self):
        registry = ToolRegistry([CalculatorTool(), EchoTool()])

        self.assertEqual(len(registry), 2)
        self.assertIn("echo", registry)

    def test_duplicate_name_rejected(self):
        registry = ToolRegistry([CalculatorTool()])

        with self.assertRaises(ToolError):
            registry.register(CalculatorTool())

        # 注册失败不应影响已有工具
        self.assertEqual(len(registry), 1)

    def test_nameless_tool_rejected(self):
        registry = ToolRegistry()

        with self.assertRaises(ToolError):
            registry.register(NamelessTool())

        self.assertEqual(len(registry), 0)


class ToolRegistryLookupTest(unittest.TestCase):
    def setUp(self):
        self.calculator = CalculatorTool()
        self.registry = ToolRegistry([self.calculator, EchoTool()])

    def test_get_returns_registered_tool(self):
        self.assertIs(self.registry.get("calculator"), self.calculator)

    def test_get_unknown_tool_raises(self):
        with self.assertRaises(ToolError):
            self.registry.get("nope")

    def test_list_returns_all_tools(self):
        tools = self.registry.list()

        self.assertEqual(len(tools), 2)
        self.assertEqual({t.name for t in tools}, {"calculator", "echo"})

    def test_contains(self):
        self.assertIn("calculator", self.registry)
        self.assertNotIn("nope", self.registry)


class ToolRegistryRunTest(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([CalculatorTool(), EchoTool()])

    def test_run_by_name(self):
        self.assertEqual(self.registry.run("calculator", expression="1+2*3"), "7")

    def test_run_unknown_tool_raises(self):
        with self.assertRaises(ToolError):
            self.registry.run("nope")

    def test_run_missing_required_arg_raises(self):
        with self.assertRaises(ToolError):
            self.registry.run("calculator")

    def test_run_wrong_arg_type_raises(self):
        with self.assertRaises(ToolError):
            self.registry.run("calculator", expression=123)

    def test_run_unknown_arg_raises(self):
        with self.assertRaises(ToolError):
            self.registry.run("calculator", expression="1+1", foo="bar")

    def test_run_optional_arg_omitted_and_given(self):
        self.assertEqual(self.registry.run("echo", text="ab"), "ab")
        self.assertEqual(self.registry.run("echo", text="ab", times=3), "ababab")

    def test_bool_is_not_accepted_as_integer(self):
        with self.assertRaises(ToolError):
            self.registry.run("echo", text="ab", times=True)


class CalculatorToolTest(unittest.TestCase):
    def setUp(self):
        self.tool = CalculatorTool()

    def test_arithmetic(self):
        cases = {
            "1+1": "2",
            "2*3+4": "10",
            "(1+2)*3": "9",
            "10/4": "2.5",
            "10/5": "2",  # 整数结果的浮点数按整数显示
            "7//2": "3",
            "7%3": "1",
            "2**10": "1024",
            "-5+3": "-2",
            "-(2+3)": "-5",
            "  3 + 4  ": "7",  # 前后空白应被容忍
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(self.tool.run(expression=expression), expected)

    def test_it_is_a_tool(self):
        self.assertIsInstance(self.tool, Tool)
        self.assertEqual(self.tool.name, "calculator")

    def test_division_by_zero_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(expression="1/0")

    def test_syntax_error_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(expression="1++")

    def test_non_numeric_input_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(expression="abc")

    def test_code_execution_is_refused(self):
        # 绝不能因为求值而执行任意代码
        dangerous = [
            "__import__('os').system('echo hi')",
            "open('x')",
            "(1).__class__",
            "lambda: 1",
            "[1,2,3]",
            "'a'*3",
            "True",
        ]
        for expression in dangerous:
            with self.subTest(expression=expression):
                with self.assertRaises(ToolError):
                    self.tool.run(expression=expression)

    def test_huge_exponent_rejected(self):
        with self.assertRaises(ToolError):
            self.tool.run(expression="9**9999999")


class ToolSchemaTest(unittest.TestCase):
    """to_schema() 生成的必须是 LLM 能直接用的 tool calling 描述。"""

    def test_calculator_schema(self):
        schema = CalculatorTool().to_schema()

        self.assertEqual(schema["type"], "function")
        function = schema["function"]
        self.assertEqual(function["name"], "calculator")
        self.assertTrue(function["description"])

        parameters = function["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(list(parameters["properties"]), ["expression"])
        self.assertEqual(parameters["required"], ["expression"])

    def test_required_flag_does_not_leak_into_properties(self):
        # required 是本项目 parameters 里的约定，不能出现在属性描述里
        prop = CalculatorTool().to_schema()["function"]["parameters"]["properties"][
            "expression"
        ]

        self.assertNotIn("required", prop)
        self.assertEqual(prop["type"], "string")
        self.assertIn("description", prop)

    def test_optional_param_is_not_required(self):
        parameters = EchoTool().to_schema()["function"]["parameters"]

        self.assertEqual(parameters["required"], ["text"])
        self.assertNotIn("required", parameters["properties"]["times"])

    def test_no_required_key_when_all_params_optional(self):
        class AllOptionalTool(Tool):
            name = "all_optional"
            description = "参数全是可选的"
            parameters = {"x": {"type": "string", "required": False}}

            def execute(self, **kwargs):
                return ""

        parameters = AllOptionalTool().to_schema()["function"]["parameters"]

        self.assertNotIn("required", parameters)

    def test_schema_follows_name_and_description(self):
        class GreetTool(Tool):
            name = "greet"
            description = "打招呼"
            parameters = {}

            def execute(self, **kwargs):
                return ""

        schema = GreetTool().to_schema()

        self.assertEqual(schema["function"]["name"], "greet")
        self.assertEqual(schema["function"]["description"], "打招呼")
        self.assertEqual(schema["function"]["parameters"]["properties"], {})

    def test_nameless_tool_cannot_build_schema(self):
        with self.assertRaises(ToolError):
            NamelessTool().to_schema()

    def test_every_tool_in_registry_has_schema(self):
        registry = ToolRegistry([CalculatorTool(), EchoTool()])

        schemas = [tool.to_schema() for tool in registry.list()]

        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            ["calculator", "echo"],
        )


if __name__ == "__main__":
    unittest.main()
