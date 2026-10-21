from eval.baseline import parse_calls, score, summarize


EXPECTED = [
    {"name": "add", "arguments": {"a": 2, "b": 3}},
    {"name": "mul", "arguments": {"a": 4, "b": 5}},
]


def test_parse_calls_valid():
    text = '[{"name": "add", "arguments": {"a": 2, "b": 3}}]'
    assert parse_calls(text) == [{"name": "add", "arguments": {"a": 2, "b": 3}}]


def test_parse_calls_code_fence():
    text = '```json\n[{"name": "add", "arguments": {}}]\n```'
    assert parse_calls(text) == [{"name": "add", "arguments": {}}]
    text = 'Here are the calls:\n\n```json\n[{"name": "add", "arguments": {}}]\n```\nDone.'
    assert parse_calls(text) == [{"name": "add", "arguments": {}}]


def test_parse_calls_invalid():
    assert parse_calls("I would call add(2, 3).") is None
    assert parse_calls('{"name": "add", "arguments": {}}') is None
    assert parse_calls('[{"name": "add"}]') is None
    assert parse_calls('[{"name": "add", "arguments": "a=2"}]') is None


def test_score_exact_match_ignores_order():
    assert score(list(reversed(EXPECTED)), EXPECTED) == {
        "valid": True, "name_match": True, "exact_match": True,
    }


def test_score_wrong_arguments():
    predicted = [
        {"name": "add", "arguments": {"a": 2, "b": 4}},
        {"name": "mul", "arguments": {"a": 4, "b": 5}},
    ]
    assert score(predicted, EXPECTED) == {"valid": True, "name_match": True, "exact_match": False}


def test_score_missing_call_and_invalid():
    assert score(EXPECTED[:1], EXPECTED)["name_match"] is False
    assert score(None, EXPECTED) == {"valid": False, "name_match": False, "exact_match": False}


def test_summarize():
    scores = [
        {"valid": True, "name_match": True, "exact_match": True},
        {"valid": True, "name_match": True, "exact_match": False},
        {"valid": False, "name_match": False, "exact_match": False},
        {"valid": True, "name_match": False, "exact_match": False},
    ]
    assert summarize(scores) == {
        "examples": 4, "valid_rate": 0.75, "name_accuracy": 0.5, "exact_match": 0.25,
    }
