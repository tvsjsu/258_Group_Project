"""Check pretrained model loading and single-query generation."""

import pytest

from model.model import MODEL_NAME, FunctionCallingModel


@pytest.fixture(scope="module")
def function_model():
    """Load the default model once for generation integration checks."""
    return FunctionCallingModel()


def test_load_model(function_model):
    """Verify that the expected model checkpoint is loaded."""
    assert function_model.model.config._name_or_path == MODEL_NAME
    assert function_model.tokenizer.name_or_path == MODEL_NAME
    assert not function_model.model.training


def test_generate(function_model):
    """Verify that the model generates a nonempty text response."""
    tools = [{"name": "add", "description": "Add two numbers.", "parameters": {
        "a": {"type": "int", "required": True},
        "b": {"type": "int", "required": True},
    }}]
    response = function_model.generate("Add 2 and 3.", tools, max_new_tokens=32)

    assert isinstance(response, str)
    assert response.strip()
