import itertools
import contextlib
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.metrics import (
    _argument_counts,
    _value_key,
    compile_tools,
    parse_calls,
    score,
    summarize,
)


def call(name="add", **arguments):
    return {"name": name, "arguments": arguments}


TOOLS = [{
    "name": "add",
    "parameters": {
        "a": {"type": "int", "required": True},
        "b": {"type": "int", "required": True},
        "label": {"type": "str", "required": False},
    },
}]
EXPECTED = [call(a=2, b=3)]


class ParsingTests(unittest.TestCase):
    def test_strict_json_and_whitespace(self):
        self.assertEqual(parse_calls(" \n" + json.dumps(EXPECTED) + "\n"), EXPECTED)

    def test_no_markdown_or_explanations(self):
        for text in (
            "```json\n" + json.dumps(EXPECTED) + "\n```",
            "Here are the calls: " + json.dumps(EXPECTED),
            json.dumps(EXPECTED) + " Done.",
        ):
            self.assertIsNone(parse_calls(text))

    def test_envelope_and_types(self):
        for value in ({"name": "add", "arguments": {}}, [{"name": "add"}],
                      [{"name": 3, "arguments": {}}],
                      [{"name": "add", "arguments": {}, "extra": 1}]):
            self.assertIsNone(parse_calls(json.dumps(value)))

    def test_duplicate_keys_and_nonfinite_values(self):
        for text in (
            '[{"name":"add","arguments":{"a":1,"a":2}}]',
            '[{"name":"add","arguments":{"a":NaN}}]',
            '[{"name":"add","arguments":{"a":Infinity}}]',
            '[{"name":"add","arguments":{"a":1e400}}]',
        ):
            self.assertIsNone(parse_calls(text))


class MetricTests(unittest.TestCase):
    def test_perfect(self):
        result = score(EXPECTED, EXPECTED, TOOLS)
        self.assertTrue(all(result[k] for k in (
            "valid", "name_match", "schema_compliant", "exact_match")))
        self.assertEqual((result["argument_correct"], result["argument_total"]), (2, 2))

    def test_partial_values_independent_of_schema(self):
        result = score([call(a=2, b=9)], EXPECTED, TOOLS)
        self.assertTrue(result["name_match"])
        self.assertTrue(result["schema_compliant"])
        self.assertFalse(result["exact_match"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (1, 2))

    def test_missing_required_argument(self):
        result = score([call(a=2)], EXPECTED, TOOLS)
        self.assertFalse(result["schema_compliant"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (1, 2))

    def test_extra_arguments_are_penalized(self):
        result = score([call(a=2, b=3, extra=5)], EXPECTED, TOOLS)
        self.assertFalse(result["schema_compliant"])
        self.assertFalse(result["exact_match"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (2, 3))

    def test_wrong_function_gets_no_argument_credit(self):
        result = score([call("other", a=2, b=3)], EXPECTED, TOOLS)
        self.assertFalse(result["name_match"])
        self.assertFalse(result["schema_compliant"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (0, 4))

    def test_optional_parameter(self):
        result = score([call(a=2, b=3, label="ok")], EXPECTED, TOOLS)
        self.assertTrue(result["schema_compliant"])
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["argument_total"], 3)

    def test_booleans_are_not_numbers(self):
        result = score([call(a=True, b=3)], EXPECTED, TOOLS)
        self.assertFalse(result["schema_compliant"])
        self.assertEqual(result["argument_correct"], 1)

    def test_equivalent_numeric_representations(self):
        result = score([call(a=2.0, b=3.0)], EXPECTED, TOOLS)
        self.assertTrue(result["schema_compliant"])
        self.assertTrue(result["exact_match"])

    def test_repeated_calls_match_one_to_one_and_ignore_order(self):
        expected = [call(a=1, b=2), call(a=3, b=4)]
        result = score(list(reversed(expected)), expected, TOOLS)
        self.assertTrue(result["exact_match"])
        result = score([expected[0], expected[0]], expected, TOOLS)
        self.assertTrue(result["name_match"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (2, 4))
        result = score(expected + [expected[0]], expected, TOOLS)
        self.assertFalse(result["name_match"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (4, 6))

    def test_nested_object_order_and_array_order(self):
        tools = [{"name": "f", "parameters": {"x": {"type": "dict", "required": True}}}]
        ref = [call("f", x={"a": [1, 2], "b": True})]
        self.assertTrue(score([call("f", x={"b": True, "a": [1, 2]})], ref, tools)["exact_match"])
        self.assertFalse(score([call("f", x={"a": [2, 1], "b": True})], ref, tools)["exact_match"])

    def test_invalid_output_counts_as_zero(self):
        result = score(None, EXPECTED, TOOLS)
        self.assertFalse(result["valid"])
        self.assertFalse(result["schema_compliant"])
        self.assertEqual((result["argument_correct"], result["argument_total"]), (0, 2))

    def test_zero_argument_and_empty_call_policies(self):
        tools = [{"name": "ping", "parameters": {}}]
        result = score([call("ping")], [call("ping")], tools)
        self.assertTrue(result["exact_match"])
        self.assertIsNone(summarize([result])["argument_accuracy"])
        result = score([], EXPECTED, TOOLS)
        self.assertTrue(result["valid"])
        self.assertTrue(result["schema_compliant"])
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["argument_total"], 2)

    def test_micro_aggregation_and_combined_counts(self):
        scores = [score(EXPECTED, EXPECTED, TOOLS), score(None, [call(a=2)], TOOLS)]
        metrics = summarize(scores)
        self.assertEqual(metrics["valid_call_rate"], 0.5)
        self.assertAlmostEqual(metrics["argument_accuracy"], 2 / 3)
        self.assertIsNone(summarize([])["valid_call_rate"])

    def test_hand_calculated_five_metric_table(self):
        predictions = [EXPECTED, [call(a=2, b=9)], [call(a=2)],
                       [call(a=2, b=3, extra=5)], [call("other", a=2, b=3)], None]
        metrics = summarize([score(p, EXPECTED, TOOLS) for p in predictions])
        self.assertAlmostEqual(metrics["valid_call_rate"], 5 / 6)
        self.assertAlmostEqual(metrics["function_accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["argument_accuracy"], 6 / 15)
        self.assertAlmostEqual(metrics["schema_compliance"], 2 / 6)
        self.assertAlmostEqual(metrics["complete_call_accuracy"], 1 / 6)

    def test_matching_against_brute_force(self):
        rng = random.Random(42)
        for _ in range(50):
            refs = [call(**{k: rng.randrange(3) for k in ("a", "b") if rng.randrange(2)}) for _ in range(3)]
            preds = [call(**{k: rng.randrange(3) for k in ("a", "c") if rng.randrange(2)}) for _ in range(3)]
            best = (-1, -1)
            for perm in itertools.permutations(preds):
                good = shared = 0
                for r, p in zip(refs, perm):
                    r, p = r["arguments"], p["arguments"]
                    keys = r.keys() & p.keys()
                    good += sum(_value_key(r[k]) == _value_key(p[k]) for k in keys)
                    shared += len(keys)
                best = max(best, (good, shared))
            good, denominator = _argument_counts(preds, refs)
            total = sum(len(c["arguments"]) for c in refs + preds)
            self.assertEqual((good, total - denominator), best)


class SchemaTests(unittest.TestCase):
    def check_type(self, hint, value, expected):
        tools = [{"name": "f", "parameters": {"x": {"type": hint, "required": True}}}]
        result = score([call("f", x=value)], [call("f", x=value)], tools)
        self.assertEqual(result["schema_compliant"], expected)

    def test_aliases_and_generics(self):
        for hint, value, expected in (
            ("float", 3, True), ("float", "3", False),
            ("List[int]", [1, 2], True), ("List[int]", [1, True], False),
            ("Optional[str]", None, True), ("str | None", 3, False),
            ("typing.List[float]", [1, 2.5], True),
            ("dict[str, int]", {"x": 1}, True),
            ("dict[str, int]", {"x": "1"}, False),
            ("Tuple[int, str]", [1, "x"], True),
            ("Tuple[int, str]", [1, "x", 2], False),
            ("Tuple[int, ...]", [1, 2, 3], True),
            ("Literal['US', 'FR']", "US", True),
            ("Literal['US', 'FR']", "Japan", False),
            ("Optional[Any]", {"x": 1}, True),
        ):
            with self.subTest(hint=hint, value=value):
                self.check_type(hint, value, expected)

    def test_json_schema_required_items_and_constraints(self):
        tools = [{"name": "f", "parameters": {
            "type": "object", "properties": {
                "xs": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                "country": {"type": "string", "enum": ["US", "FR"]},
            }, "required": ["xs"],
        }}]
        self.assertTrue(score([call("f", xs=[1])], [call("f", xs=[1])], tools)["schema_compliant"])
        for arguments in ({}, {"xs": []}, {"xs": ["1"]}, {"xs": [1], "country": "JP"}, {"xs": [1], "extra": 2}):
            self.assertFalse(score([call("f", **arguments)], [], tools)["schema_compliant"])

    def test_unknown_type_and_invalid_schemas_fail_early(self):
        for parameters in (
            {"x": {"type": "ImaginaryType"}},
            {"x": {"type": "__import__('os').system('echo BAD')"}},
            {"x": {"type": "int", "minimum": "invalid"}},
            {"x": {"type": "int", "required": "false"}},
            {"type": "object", "properties": {}, "required": True},
            {"x": {"type": "custom.List[int]"}},
            {"x": {"$ref": "https://example.invalid/schema"}},
            {"type": "object", "properties": {"x": {"type": "str"}},
             "if": {"$dynamicRef": "https://example.invalid/schema"}},
        ):
            with self.assertRaises(ValueError):
                compile_tools([{"name": "f", "parameters": parameters}])

    def test_same_name_overloads(self):
        tools = [
            {"name": "f", "parameters": {"x": {"type": "int", "required": True}}},
            {"name": "f", "parameters": {"y": {"type": "str", "required": True}}},
        ]
        self.assertTrue(score([call("f", y="ok")], [], tools)["schema_compliant"])


class BaselineIntegrationTests(unittest.TestCase):
    """Exercise CLI, serialization and aggregation with model/data test doubles."""

    def setUp(self):
        self.model_loads = 0
        owner = self
        rows = [{"id": i, "query": f"q{i}", "tools": TOOLS, "answers": EXPECTED}
                for i in range(3)]

        class FakeDataset:
            def __init__(self, examples):
                self.examples = examples

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, index):
                return self.examples[index]

            @classmethod
            def load_splits(cls):
                return {"test_seen": cls(rows[:2]), "test_unseen": cls(rows[2:]),
                        "validation": cls(rows)}

            def make_loader(self, batch_size, shuffle, num_workers):
                assert shuffle is False and num_workers == 0
                return (self.examples[i:i + batch_size]
                        for i in range(0, len(self), batch_size))

        class FakeModel:
            def __init__(self, **kwargs):
                owner.model_loads += 1
                self.model = types.SimpleNamespace(dtype="torch.bfloat16", device="cpu")

            def generate_batch(self, queries, tools, max_new_tokens):
                outputs = {
                    "q0": json.dumps(EXPECTED),
                    "q1": "```json\n" + json.dumps(EXPECTED) + "\n```",
                    "q2": json.dumps([call(a="2", b=3)]),
                }
                return [outputs[q] for q in queries]

        self.dataset_class = FakeDataset
        modules = {name: types.ModuleType(name) for name in (
            "data", "data.data", "model", "model.model")}
        modules["data"].__path__ = []
        modules["model"].__path__ = []
        modules["data.data"].FunctionCallingDataset = FakeDataset
        modules["model.model"].FunctionCallingModel = FakeModel
        modules["model.model"].MODEL_NAME = "test-double"
        self.modules = modules

    def run_cli(self, *arguments):
        from eval.baseline import main
        output = io.StringIO()
        with patch.dict(sys.modules, self.modules), patch.object(
            sys, "argv", ["baseline", *arguments]
        ), contextlib.redirect_stdout(output):
            main()
        return output.getvalue()

    def test_both_splits_saved_and_combined_correctly(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "predictions.jsonl")
            text = self.run_cli("--split", "both", "--batch-size", "8", "--output", str(output))
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            report = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(records), 3)
            self.assertEqual([r["id"] for r in records], [0, 1, 2])
            self.assertTrue(all("tools" in r and "raw_output" in r for r in records))
            self.assertEqual(set(report["splits"]), {"test_seen", "test_unseen"})
            for key in ("valid_call_rate", "function_accuracy"):
                self.assertAlmostEqual(report["combined"][key], 2 / 3)
            for key in ("schema_compliance", "complete_call_accuracy"):
                self.assertAlmostEqual(report["combined"][key], 1 / 3)
            self.assertEqual(report["combined"]["argument_accuracy"], 0.5)
            self.assertIn("Argument accuracy", text)
            self.assertIn("Schema compliance", text)

    def test_limit_and_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            # No-output and output are mutually exclusive; the default must not be used.
            with patch("eval.baseline.open", side_effect=AssertionError("unexpected file write"), create=True):
                text = self.run_cli("--split", "validation", "--limit", "1", "--no-output")
            self.assertIn("Examples: 1", text)

    def test_preflight_does_not_load_model(self):
        self.run_cli("--split", "both", "--check-data-only")
        self.assertEqual(self.model_loads, 0)

    def test_invalid_schema_stops_before_model_loading(self):
        bad = self.dataset_class([{"id": 9, "query": "q", "answers": EXPECTED,
                                  "tools": [{"name": "f", "parameters": {"x": {"type": "Unknown"}}}]}])
        with patch.object(self.dataset_class, "load_splits", return_value={"test_seen": bad}):
            with self.assertRaisesRegex(ValueError, "Example 9"):
                self.run_cli("--check-data-only")
        self.assertEqual(self.model_loads, 0)


if __name__ == "__main__":
    unittest.main()
