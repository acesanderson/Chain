from Chain import Chain, Model, Prompt, Message


def test_chain_init():
    model = Model()
    prompt = Prompt("What is the capital of France?")
    chain = Chain(prompt=prompt, model=model)
    assert chain.prompt == prompt
    assert chain.model == model
    assert chain.parser == None
    assert chain.input_schema == prompt.input_schema()


def test_chain_run_completion():
    model = Model("haiku")
    prompt = Prompt("What is the capital of France?")
    chain = Chain(prompt=prompt, model=model)
    response = chain.run()
    assert response.status == "success"
    assert response.prompt == prompt.prompt_string
    assert response.model == model.model
    assert response.duration > 0
    assert response.messages[0].role == "user"
    assert response.messages[1].role == "assistant"


def test_chain_run_messages():
    system_message = Message(role="system", content="You talk like a pirate.")
    user_message = Message(role="user", content="Name ten things on the high seas.")
    messages = [system_message, user_message]
    model = Model("haiku")
    chain = Chain(model=model)
    response = chain.run_messages(messages=messages)
    assert response.status == "success"
    assert response.prompt == None


def test_chain_run_prompt_and_messages():
    system_message = Message(role="system", content="You talk like a pirate.")
    prompt = Prompt("What is the capital of France?")
    model = Model("haiku")
    chain = Chain(prompt=prompt, model=model)
    response = chain.run(messages=[system_message])
    assert response.status == "success"
    assert response.prompt == prompt.prompt_string
    assert response.messages[0].role == "system"


def test_chain_single_variable():
    model = Model("haiku")
    prompt = Prompt("What is the capital of {{country}}?")
    chain = Chain(prompt=prompt, model=model)
    response = chain.run(input_variables={"country": "France"})
    assert response.status == "success"
    assert response.prompt == prompt.render({"country": "France"})
    assert response.messages[0].content == "What is the capital of France?"
