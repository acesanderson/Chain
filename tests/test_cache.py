from Chain import Model, Prompt, Chain
from Chain.cache.cache import ChainCache
from time import time

m = Model("gpt")
c = ChainCache("example.db")


def clear_cache():
    c.clear_cache()


def peek_at_cache():
    cached_requests = c.retrieve_cached_requests()
    for row in c.retrieve_cached_requests():
        print(row)
    return cached_requests


def run_lookups():
    lookup_examples = [
        {"user_input": "name five mammals", "model": "haiku"},
        {"user_input": "what is the capital of germany?", "model": "haiku"},
        {"user_input": "what is the capital of france?", "model": "haiku"},
        {"user_input": "what is your context cutoff?", "model": "haiku"},
        {"user_input": "is this thing on?", "model": "haiku"},
        {"user_input": "what OS am I using?", "model": "gpt-3.5-turbo-0125"},
        {"user_input": "what OS am I using?", "model": "gemini-1.5-flash-8b"},
        {"user_input": "what OS am I using?", "model": "haiku"},
    ]

    Model._chain_cache = c

    for lookup_example in lookup_examples:
        prompt = Prompt(lookup_example["user_input"])
        model = Model(lookup_example["model"])
        chain = Chain(prompt=prompt, model=model)
        response = chain.run()
        print(response)


# def test_cache():
#     start_uncached = time()
#     run_lookups()
#     end_uncached = time()
#     cached_request = peek_at_cache()
#     assert len(cached_request) == 7
#     start_cached = time()
#     run_lookups()
#     end_cached = time()
#     assert end_cached - start_cached < end_uncached - start_uncached + 1
#


def test_cache():
    # Clear our db
    c.clear_cache()
    # Measure uncached time
    uncached_start = time()
    run_lookups()
    uncached_end = time()
    # Peek at cache
    cached_request = peek_at_cache()
    assert len(cached_request) == 8
    # Measure cached time
    cached_start = time()
    run_lookups()
    cached_end = time()
    # Compare averages
    uncached_time = uncached_end - uncached_start
    cached_time = cached_end - cached_start
    # Assert cached is significantly faster (e.g. .01 seconds versus 3 seconds)
    assert cached_time < uncached_time - 2
