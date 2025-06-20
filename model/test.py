from Chain import AsyncChain
from Chain.model.model_async import ModelAsync

model = ModelAsync("claude")
prompt_strings = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Spain?",
    "What is the capital of Italy?",
    "What is the capital of Portugal?",
    "What is the capital of Greece?",
]
chain = AsyncChain(model=model)
response = chain.run(prompt_strings=prompt_strings)
print(response)
