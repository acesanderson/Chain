from Chain.model.clients.openai_client import OpenAIClientAsync
from Chain.chain.asyncchain import AsyncChain
from Chain.model.model import Model
import asyncio
import pytest


@pytest.fixture
def async_openai_client():
    return OpenAIClientAsync()


@pytest.fixture
def async_chain():
    return AsyncChain(model=Model("o1-mini"))


@pytest.fixture
def prompt_strings():
    list_of_things = [
        "frogs",
        "buildings",
        "ukrainian poets",
        "careers in dentistry",
        "poetic themes",
        "Apple products",
        "dead people",
        "shocking films",
        "non-American cars",
    ]
    return ["Name five " + thing + "." for thing in list_of_things]


@pytest.fixture
def input_variables():
    list_of_things = [
        "frogs",
        "buildings",
        "ukrainian poets",
        "careers in dentistry",
        "poetic themes",
        "Apple products",
        "dead people",
        "shocking films",
        "non-American cars",
    ]
    return [{"things": thing} for thing in list_of_things]


# First, let's test synchronous usage of the async client.
def test_openai_query_async_for_sync_context(async_openai_client):
    single_prompt = "Name five frogs."
    client = async_openai_client
    response = asyncio.run(client.query_async(model="o1-mini", input=single_prompt))
    print(response)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


# Second, let's test asynchronous usage of the async client, with an external implementation.


def test_open_query_async_prompt_strings_external(async_openai_client, prompt_strings):
    async def openai_query_async_prompt_strings_external():
        client = async_openai_client
        coroutines = [
            client.query_async(model="o1-mini", input=prompt)
            for prompt in prompt_strings
        ]
        results = await asyncio.gather(*coroutines)
        print(results)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == len(prompt_strings)
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > 0 for result in results)

    asyncio.run(openai_query_async_prompt_strings_external())


# Third, now that this works, we need to add a bulk function on top. This is handled by a new AsyncChain class.
# Test init.


def test_init_asyncchain(async_chain):
    chain = async_chain
    assert chain is not None
    assert isinstance(chain, AsyncChain)
    assert isinstance(chain.model, Model)
    assert chain.model.model == "o1-mini"


# Working our way up from the bottom, let's test the _run_prompt_strings function.
def test_run_prompt_strings(async_chain, prompt_strings):
    async def run_prompt_strings():
        chain = async_chain
        results = await chain._run_prompt_strings(prompt_strings=prompt_strings)
        return results

    results = asyncio.run(run_prompt_strings())
    assert results is not None
    assert isinstance(results, list)
    assert len(results) == len(prompt_strings)
    assert all(isinstance(result, str) for result in results)
    assert all(len(result) > 0 for result in results)


# Our other function: _run_input_variables.

# def test_run_input_variables(async_chain):
