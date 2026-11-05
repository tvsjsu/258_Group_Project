import ast
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache

from jsonschema import Draft202012Validator
from scipy.optimize import linear_sum_assignment


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_value(v) for k, v in value.items())
    return False


def _valid_calls(calls):
    return isinstance(calls, list) and all(
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call["name"], str)
        and isinstance(call["arguments"], dict)
        and _json_value(call["arguments"])
        for call in calls
    )


def parse_calls(text):
    """Require the entire output to be JSON; allow surrounding whitespace only."""
    try:
        calls = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        return calls if _valid_calls(calls) else None
    except (ValueError, TypeError, RecursionError):
        return None


def _value_key(value):
    """JSON equality: unordered object keys, ordered arrays, distinct booleans."""
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) in (int, float):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_value_key(v) for v in value))
    return ("object", tuple(sorted((k, _value_key(v)) for k, v in value.items())))


_TYPES = {
    "str": "string", "string": "string", "int": "integer",
    "integer": "integer", "float": "number", "number": "number",
    "double": "number", "bool": "boolean", "boolean": "boolean",
    "list": "array", "array": "array", "tuple": "array",
    "dict": "object", "dictionary": "object", "object": "object",
    "none": "null", "nonetype": "null", "null": "null",
}


def _type_schema(hint):
    """Translate common xLAM/Python type labels without executing type strings."""
    if isinstance(hint, list):
        return {"anyOf": [_type_schema(t) for t in hint]}
    if not isinstance(hint, str):
        raise ValueError(f"Unsupported parameter type: {hint!r}")

    def convert(node):
        if isinstance(node, ast.Name):
            name = node.id.lower()
            if name == "any":
                return {}
            if name in _TYPES:
                return {"type": _TYPES[name]}
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("typing", "builtins"):
                return convert(ast.Name(id=node.attr))
        elif isinstance(node, ast.Constant):
            if node.value is None:
                return {"type": "null"}
            if isinstance(node.value, str):
                return _type_schema(node.value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return {"anyOf": [convert(node.left), convert(node.right)]}
        elif isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Attribute) and not (
                isinstance(base.value, ast.Name) and base.value.id in ("typing", "builtins")
            ):
                raise ValueError(f"Unsupported parameter type: {hint!r}")
            name = (base.id if isinstance(base, ast.Name) else base.attr).lower()
            args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if name in ("list", "array") and len(args) == 1:
                return {"type": "array", "items": convert(args[0])}
            if name in ("dict", "dictionary") and len(args) == 2:
                key_schema = convert(args[0])
                if key_schema not in ({}, {"type": "string"}):
                    raise ValueError("JSON dictionary keys must be strings")
                return {"type": "object", "additionalProperties": convert(args[1])}
            if name == "optional" and len(args) == 1:
                return {"anyOf": [convert(args[0]), {"type": "null"}]}
            if name == "union":
                return {"anyOf": [convert(a) for a in args]}
            if name == "literal":
                values = [ast.literal_eval(a) for a in args]
                if all(_json_value(v) for v in values):
                    return {"enum": values}
            if name == "tuple":
                if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis:
                    return {"type": "array", "items": convert(args[0])}
                return {
                    "type": "array", "prefixItems": [convert(a) for a in args],
                    "items": False, "minItems": len(args),
                }
        raise ValueError(f"Unsupported parameter type: {hint!r}")

    try:
        return convert(ast.parse(hint.strip(), mode="eval").body)
    except (SyntaxError, AttributeError) as exc:
        raise ValueError(f"Unsupported parameter type: {hint!r}") from exc


def _normalize_spec(spec):
    if isinstance(spec, bool):
        return spec
    if not isinstance(spec, dict):
        raise ValueError("Parameter definitions must be objects")
    if not spec:
        return {}
    result = dict(spec)
    if "type" in result:
        converted = _type_schema(result.pop("type"))
        if converted.keys() & result.keys():
            result = {"allOf": [converted, result]}
        else:
            result = {**converted, **result}
    if "$ref" in result:
        raise ValueError("Referenced schemas need explicit support; refusing to guess")
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if key in result:
            result[key] = [_normalize_spec(s) for s in result[key]]
    for key in ("items", "additionalProperties"):
        if isinstance(result.get(key), dict):
            result[key] = _normalize_spec(result[key])
    if "properties" in result:
        result["properties"] = {k: _normalize_spec(v) for k, v in result["properties"].items()}
    return result


def _reject_references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("$ref", "$dynamicRef", "$recursiveRef") and isinstance(item, str):
                raise ValueError("Referenced schemas are unsupported; no references will be fetched")
            _reject_references(item)
    elif isinstance(value, list):
        for item in value:
            _reject_references(item)


@lru_cache(maxsize=8192)
def _validator(parameters_json):
    parameters = json.loads(parameters_json)
    _reject_references(parameters)
    if not isinstance(parameters, dict):
        raise ValueError("Tool parameters must be an object")
    if parameters.get("type") == "object":
        schema = _normalize_spec(parameters)
        schema.setdefault("properties", {})
        schema["additionalProperties"] = False
    else:
        properties, required = {}, []
        for name, descriptor in parameters.items():
            if not isinstance(descriptor, dict):
                raise ValueError(f"Invalid parameter definition: {name!r}")
            spec = dict(descriptor)
            is_required = spec.pop("required", False)
            if not isinstance(is_required, bool):
                raise ValueError(f"Required flag must be boolean: {name!r}")
            properties[name] = _normalize_spec(spec)
            if is_required:
                required.append(name)
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    Draft202012Validator.check_schema(schema)
    if any(name not in schema["properties"] for name in schema.get("required", [])):
        raise ValueError("Required argument is missing from the parameter definitions")
    return Draft202012Validator(schema)


def compile_tools(tools):
    """Fail on unsupported tool schemas instead of miscounting model errors."""
    validators = defaultdict(list)
    for tool in tools:
        name = tool["name"]
        try:
            key = json.dumps(tool["parameters"], sort_keys=True, allow_nan=False)
            validators[name].append(_validator(key))
        except Exception as exc:
            raise ValueError(f"Cannot validate schema for tool {name!r}: {exc}") from exc
    return validators


def validate_examples(examples):
    """Check references and all supplied schemas before loading model weights."""
    for example in examples:
        try:
            if not _valid_calls(example["answers"]):
                raise ValueError("Invalid reference call structure")
            compile_tools(example["tools"])
        except Exception as exc:
            raise ValueError(f"Example {example.get('id')!r}: {exc}") from exc


def _argument_counts(predicted, expected):
    """Pair calls by name; maximize correct values, then shared argument names."""
    correct = shared = 0
    ref_total = sum(len(c["arguments"]) for c in expected)
    pred_total = sum(len(c["arguments"]) for c in predicted)
    scale = ref_total + pred_total + 1
    for name in {c["name"] for c in expected}:
        refs = [c["arguments"] for c in expected if c["name"] == name]
        preds = [c["arguments"] for c in predicted if c["name"] == name]
        if not preds:
            continue
        pairs = [[
            (sum(_value_key(r[k]) == _value_key(p[k]) for k in r.keys() & p.keys()),
             len(r.keys() & p.keys()))
            for p in preds
        ] for r in refs]
        weights = [[good * scale + overlap for good, overlap in row] for row in pairs]
        rows, cols = linear_sum_assignment(weights, maximize=True)
        for i, j in zip(rows, cols):
            good, overlap = pairs[i][j]
            correct += good
            shared += overlap
    # Reference slots plus predicted slots without a matched reference name.
    return correct, ref_total + pred_total - shared


def score(predicted, expected, tools):
    if not _valid_calls(expected):
        raise ValueError("Invalid reference call structure")
    validators = compile_tools(tools)
    valid = predicted is not None and _valid_calls(predicted)
    if not valid:
        return {
            "valid": False, "name_match": False, "exact_match": False,
            "schema_compliant": False, "argument_correct": 0,
            "argument_total": sum(len(c["arguments"]) for c in expected),
        }
    correct, total = _argument_counts(predicted, expected)
    return {
        "valid": True,
        "name_match": Counter(c["name"] for c in predicted) == Counter(c["name"] for c in expected),
        "exact_match": Counter(_value_key(c) for c in predicted) == Counter(_value_key(c) for c in expected),
        "schema_compliant": all(
            any(v.is_valid(c["arguments"]) for v in validators.get(c["name"], []))
            for c in predicted
        ),
        "argument_correct": correct,
        "argument_total": total,
    }


def summarize(scores):
    count = len(scores)
    correct = sum(s["argument_correct"] for s in scores)
    total = sum(s["argument_total"] for s in scores)
    def rate(key):
        return sum(s[key] for s in scores) / count if count else None
    return {
        "examples": count,
        "valid_call_rate": rate("valid"),
        "function_accuracy": rate("name_match"),
        "argument_accuracy": correct / total if total else None,
        "schema_compliance": rate("schema_compliant"),
        "complete_call_accuracy": rate("exact_match"),
        "argument_correct": correct,
        "argument_total": total,
    }


def print_summary(label, scores):
    metrics = summarize(scores)
    print(f"\nSplit: {label}  Examples: {metrics['examples']}")
    for title, key in (
        ("Valid-call rate", "valid_call_rate"),
        ("Function accuracy", "function_accuracy"),
        ("Argument accuracy", "argument_accuracy"),
        ("Schema compliance", "schema_compliance"),
        ("Complete-call accuracy", "complete_call_accuracy"),
    ):
        value = metrics[key]
        display = "N/A" if value is None else f"{100 * value:.2f}%"
        print(f"{title:<24} {display}")
