from Chain import Model, Prompt, Chain
from Chain.cache.cache import ChainCache

m = Model("gpt")
c = ChainCache("example.db")

# cache = ChainCache(db_name=".example.db")
# examples = [
#     Response(
#         prompt="name five mammals",
#         content="(1) lizard (2) dog (3) bird (4) dinosaur (5) human",
#         model="gpt",
#     ),
#     Response(prompt="what is the capital of france?", content="Paris", model="gpt"),
#     Response(
#         prompt="is this thing on?",
#         content="yes, do you have a question?",
#         model="gpt",
#     ),
#     Response(prompt="what OS am I using?", content="macOS", model="gpt"),
# ]
# examples = [cache.create_cached_request(r) for r in examples]
# for e in examples:
#     cache.insert_cached_request(e)
lookup_examples = [
    {"user_input": "name five mammals", "model": "gpt"},
    {"user_input": "what is the capital of germany?", "model": "gpt"},
    {"user_input": "what is the capital of france?", "model": "gpt"},
    {"user_input": "what is your context cutoff?", "model": "gpt"},
    {"user_input": "is this thing on?", "model": "gpt"},
    {"user_input": "what OS am I using?", "model": "gpt"},
    {"user_input": "what OS am I using?", "model": "claude"},
    {"user_input": "what OS am I using?", "model": "gemini"},
]

m._chain_cache = c

for lookup_example in lookup_examples:
    prompt = Prompt(lookup_example["user_input"])
    model = Model(lookup_example["model"])
    chain = Chain(prompt=prompt, model=model)
    response = chain.run()
    print(response)
