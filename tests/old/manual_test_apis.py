"""
This tests our API connections.
Currently that is just openai, google, and anthropic.
This filename is prepended with "manual" to exclude it from the automated tests.
Best practice would be to route this with conftest.py at some point, for now, to invoke:
pytest manual_test_apis.py
"""

from Chain import Model
import pytest


@pytest.mark.parametrize(
    "model",
    [
        Model("haiku"),
        Model("gpt-mini"),
        Model("gemini-1.5-flash-latest"),
    ],
)
def test_api_connections(model):
    response = model.query("is this thing on?")
    assert len(response) > 0
