from Chain import Chain, Model, Prompt, Message
import pytest

# Our fixtures
# ======================================================================================


@pytest.fixture
def default_model():
    model = "gpt3"
    # model = "haiku"
    # model = "gemini"
    return Model(model)


@pytest.fixture
def sample_prompt():
    return Prompt("What is the capital of France?")


# Our tests
# ======================================================================================


def test_chain_init(default_model, sample_prompt):
    chain = Chain(prompt=sample_prompt, model=default_model)
    assert chain.prompt == sample_prompt
    assert chain.model == default_model
    assert chain.parser == None
    assert chain.input_schema == sample_prompt.input_schema()


def test_chain_run_completion(default_model, sample_prompt):
    chain = Chain(prompt=sample_prompt, model=default_model)
    response = chain.run()
    assert response.status == "success"
    assert response.prompt == sample_prompt.prompt_string
    assert response.model == default_model.model
    assert response.duration > 0
    assert response.messages[0].role == "user"
    assert response.messages[1].role == "assistant"


def test_chain_run_messages(default_model):
    system_message = Message(role="system", content="You talk like a pirate.")
    user_message = Message(role="user", content="Name ten things on the high seas.")
    messages = [system_message, user_message]
    chain = Chain(model=default_model)
    response = chain.run_messages(messages=messages)
    assert response.status == "success"
    assert response.prompt == None


def test_chain_run_prompt_and_messages(default_model, sample_prompt):
    system_message = Message(role="system", content="You talk like a pirate.")
    chain = Chain(prompt=sample_prompt, model=default_model)
    response = chain.run(messages=[system_message])
    assert response.status == "success"
    assert response.prompt == sample_prompt.prompt_string
    assert response.messages[0].role == "system"


def test_chain_single_variable(default_model):
    prompt = Prompt("What is the capital of {{country}}?")
    chain = Chain(prompt=prompt, model=default_model)
    response = chain.run(input_variables={"country": "France"})
    assert response.status == "success"
    assert response.prompt == prompt.render({"country": "France"})
    assert response.messages[0].content == "What is the capital of France?"
