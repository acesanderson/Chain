from Chain import Model
from Chain.model.clients.client import Client
import pytest


@pytest.fixture
def default_model():
    return Model("haiku")


def test_model_init(default_model):
    assert default_model.model == "claude-3-5-haiku-20241022"
    assert isinstance(default_model._client, Client)
    assert default_model._client_type == ("anthropic", "AnthropicClient")


def test_model_init_fail():
    with pytest.raises(ValueError) as e:
        _ = Model("AGI")
    assert str(e.value) == "Model AGI not found in models"


def load_model_list():
    model_list = Model.models
    assert "gpt-4o" in model_list["openai"]
    assert "claude-3-5-sonnet-20241022" in model_list["anthropic"]
    assert "llama3.1:latest" in model_list["ollama"]
    assert "gemini-1.5-pro-latest" in model_list["google"]


def test_model_get_client():
    model = Model("gpt-4o")
    client = model._get_client(("openai", "OpenAIClient"))
    assert isinstance(client, Client)
