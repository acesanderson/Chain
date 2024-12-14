def test_Chain():
    chain = Chain(model=model, parser=parser)
    prompt = "What is the capital of France?"
    response = chain.run(prompt)
    assert response.status == "success"
    assert response.prompt == prompt
    assert response.model == model.model
    assert response.duration > 0
    assert response.messages[0].role == "user"
    assert response.messages[1].role == "assistant"
    assert response.variables == input
