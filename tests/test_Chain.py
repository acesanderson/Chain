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
def test_default_chain(setup):
    c=Chain()
    r=c.run()
    assert isinstance(r.content, str)
    assert len(r) > 0

@pytest.mark.run_occasionally
def test_Model(setup):
    m=Model()
    r=m.query("Name ten mammals.")
    assert isinstance(r, str)
    assert len(r) > 0

@pytest.mark.run_occasionally
def test_json_parser(setup):
    p=Parser('json')
    r=p.parse("{\"content\":\"Hello, world!\", \"status\":\"success\"}")
    assert isinstance(r, dict)
    assert len(r.keys()) == 2

