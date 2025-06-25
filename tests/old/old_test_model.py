from Chain import Model
from Chain.model.clients.client import Client
import pytest


@pytest.fixture
def default_model():
    return Model("haiku")


def test_model_init(default_model):
    assert default_model.model == "claude-3-5-haiku-20241022"
    assert isinstance(default_model._client, Client)
    assert default_model._client_type == ("anthropic", "AnthropicClientSync")


def test_model_init_fail():
    with pytest.raises(ValueError) as e:
        _ = Model("AGI")
    assert str(e.value) == "Model AGI not found in models"


def load_model_list():
    model_list = Model.models
    assert "gpt-4o" in model_list["openai"]
    assert "claude-3-5-sonnet-20241022" in model_list["anthropic"]
    assert "llama3.1:latest" in model_list["ollama"]
    assert "gemini-1.5-flash" in model_list["google"]


@pytest.mark.parametrize(
    "model_name, provider, client",
    [
        ("gpt-4o", "openai", "OpenAIClient"),
        ("claude-3-5-sonnet-20241022", "anthropic", "AnthropicClient"),
        ("gemini-1.5-flash", "google", "GoogleClient"),
    ],
)
def test_model_get_client(model_name, provider, client):
    """
    Testing all our models and clients.
    NOTE: ollama testing is not yet figured out for our containerized environment; need port forwarding to work.
    """
    model = Model(model_name)
    client = model._get_client((provider, client))
    assert isinstance(client, Client)
