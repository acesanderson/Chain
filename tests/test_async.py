from Chain.model.clients.openai_client import OpenAIClientAsync
from Chain.model.clients.anthropic_client import AnthropicClientAsync
from Chain.chain.asyncchain import AsyncChain
from Chain.model.model import ModelAsync
from Chain.response.response import Response
from Chain.prompt.prompt import Prompt
import asyncio
import pytest


# Our fixtures
# ======================================================================================
@pytest.fixture
def prompt_object():
    prompt_string = "Name five {{things}}."
    assert "{{" in prompt_string and "}}" in prompt_string
    return Prompt(prompt_string)


@pytest.fixture
def model_string():
    # return "gpt-3.5-turbo-0125"
    return "claude-3-haiku-20240307"


@pytest.fixture
def async_client():
    # return OpenAIClientAsync()
    return AnthropicClientAsync()


@pytest.fixture
def async_chain(prompt_object, model_string):
    return AsyncChain(prompt=prompt_object, model=ModelAsync(model_string))


@pytest.fixture
def list_of_things():
    return [
        "frogs",
        "buildings",
        # "ukrainian poets",
        # "careers in dentistry",
        # "poetic themes",
        # "Apple products",
        # "dead people",
        # "shocking films",
        # "non-American cars",
    ]


@pytest.fixture
def prompt_strings(list_of_things):
    return ["Name five " + thing + "." for thing in list_of_things]


@pytest.fixture
def input_variables_list(list_of_things):
    list_of_things = [
        "frogs",
        "buildings",
        # "ukrainian poets",
        # "careers in dentistry",
        # "poetic themes",
        # "Apple products",
        # "dead people",
        # "shocking films",
        # "non-American cars",
    ]
    return [{"things": thing} for thing in list_of_things]


# Our tests
# ======================================================================================


# First, let's test synchronous usage of the async client.
def test_query_async_for_sync_context(async_client, model_string):
    single_prompt = "Name five frogs."
    client = async_client
    response = asyncio.run(client.query(model=model_string, input=single_prompt))
    print(response)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


# Second, let's test asynchronous usage of the async client, with an external implementation.


def test_query_async_prompt_strings_external(
    async_client, prompt_strings, model_string
):
    async def query_async_prompt_strings_external():
        client = async_client
        coroutines = [
            client.query(model=model_string, input=prompt) for prompt in prompt_strings
        ]
        results = await asyncio.gather(*coroutines)
        print(results)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == len(prompt_strings)
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > 0 for result in results)

    asyncio.run(query_async_prompt_strings_external())


# Now that this works, we need to add a bulk function on top. This is handled by a new AsyncChain class.
# Test init.


def test_init_asyncchain(async_chain, model_string):
    chain = async_chain
    assert chain is not None
    assert isinstance(chain, AsyncChain)
    assert isinstance(chain.model, ModelAsync)
    assert chain.model.model == model_string


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


# Now, let's see if asynchain.run works when passed a list of prompt strings.
def test_asyncchain_run_prompt_strings(async_chain, prompt_strings):
    chain = async_chain
    results = chain.run(prompt_strings=prompt_strings)
    assert results is not None
    assert isinstance(results, list)
    assert len(results) == len(prompt_strings)
    assert all(isinstance(result, Response) for result in results)
    assert all(len(result.content) > 0 for result in results)


# Our other function: _run_input_variables. First, let's start with the client-level query function
def test_query_async_input_variables(
    async_client, input_variables_list, prompt_object, model_string
):
    async def query_async_input_variables():
        client = async_client
        coroutines = [
            client.query(
                model=model_string,
                input=prompt_object.render(input_variables={"things": input_variable}),
            )
            for input_variable in input_variables_list
        ]
        results = await asyncio.gather(*coroutines)
        print(results)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == len(input_variables_list)
        assert all(isinstance(result, str) for result in results)
        assert all(len(result) > 0 for result in results)

    asyncio.run(query_async_input_variables())


# Now our run sub-function in AsyncChain
def test_run_input_variables(async_chain, input_variables_list):
    async def run_input_variables():
        chain = async_chain
        results = await chain._run_input_variables(
            input_variables_list=input_variables_list
        )
        return results

    results = asyncio.run(run_input_variables())
    assert results is not None
    assert isinstance(results, list)
    assert len(results) == len(input_variables_list)
    assert all(isinstance(result, str) for result in results)
    assert all(len(result) > 0 for result in results)


# Finally, let's test the main run function with input variables.
def test_asyncchain_run_input_variables(async_chain, input_variables_list):
    chain = async_chain
    results = chain.run(input_variables_list=input_variables_list)
    print(results)
    assert results is not None
    assert isinstance(results, list)
    assert len(results) == len(input_variables_list)
    assert all(isinstance(result, Response) for result in results)
    assert all(len(result.content) > 0 for result in results)
