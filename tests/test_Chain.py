"""
Decorators:
@pytest.mark.run_every_commit
@pytest.mark.run_occasionally
"""
import pytest
from Chain import Chain, Model, Parser, Prompt

@pytest.fixture
def setup():
    # Set up any necessary preconditions or test data
    # This fixture will be executed before each test function
    # Example: Initialize the ChromaDB client and collection
    pass

@pytest.mark.run_occasionally
def test_default_Chain(setup):
    c=Chain(model=Model('gpt-3.5-turbo-0125'))
    r=c.run()
    assert isinstance(r, str)
    assert len(r) > 0
