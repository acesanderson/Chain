from Chain.api.server.ChainRequest import ChainRequest, process_ChainRequest
from Chain import Message


example_request1 = ChainRequest(
    prompt="name ten mammals", model="gpt", input_variables=None, messages=None
)

messages2 = [
    Message(role="user", content="name ten mammals"),
]
example_request2 = ChainRequest(
    prompt=None, messages=messages2, model="gemini", input_variables=None
)

messages3 = [
    Message(
        role="system",
        content="You are a helpful assistant. Your bias is to identify very rare or obscure things.",
    ),
]
example_request3 = ChainRequest(
    prompt="name ten {{things}}",
    messages=messages3,
    model="gpt",
    input_variables={"things": "birds"},
)

example_requests = [
    example_request1,
    example_request2,
    example_request3,
]

for example_request in example_requests:
    print(f"example_request: {example_request}")
    response = process_ChainRequest(example_request)
    print(f"response: {response}")
    print()
