from Chain import Model, Chain
from Chain.cache.cache import ChainCache
from openai.types.chat.chat_completion import ChatCompletion # Import the specific type
from pydantic import BaseModel # Assuming you might use this for structured outputs
from typing import Any
import time, pytest, asyncio



# --- Fixtures (as in your original code) ---
@pytest.fixture
def default_model():
    # Using a specific model for consistency in tests
    return Model("gpt-4o-mini")

@pytest.fixture
def sample_prompt_str():
    return "What is the capital of France?"

@pytest.fixture
def test_queries_data():
    return [
        ("What is the capital of France?", "o4-mini"),
        ("What is the highest mountain?", "gpt-4o-mini"),
        ("What is 2+2?", "gpt-4o-mini"),
        ("Name three fruits.", "haiku"),
        ("Name three vegetables.", "llama3.1:latest"),
    ]

# --- Helper function to get content from OpenAI ChatCompletion (or other BaseModels) ---
def get_response_content_preview(response_obj: Any) -> str:
    """Extracts a preview of the content from a response object."""
    if isinstance(response_obj, ChatCompletion):
        # Access content from OpenAI ChatCompletion object
        if response_obj.choices and response_obj.choices[0].message.content is not None:
            return response_obj.choices[0].message.content
        elif response_obj.choices and response_obj.choices[0].message.tool_calls:
            # Handle tool calls if they are present instead of content
            return f"Tool calls: {response_obj.choices[0].message.tool_calls[0].function.name}..."
        else:
            return str(response_obj) # Fallback if no content or tool calls
    elif isinstance(response_obj, BaseModel):
        # If it's a Pydantic model (like Frog from your examples), try to dump it
        return response_obj.model_dump_json()[:50] + "..."
    elif isinstance(response_obj, str):
        return response_obj
    else:
        return str(response_obj) # Generic fallback for other types

# --- Test Functions ---

def test_sync_cache():
    """Test caching for sync Model.query calls with string responses."""
    print("\n" + "="*60)
    print("🔄 Testing Sync Model Cache")
    print("----------------------------------------")
    print("Test 1: String responses")

    cache = ChainCache("test_sync_cache.db")
    cache.clear_cache()
    Model._chain_cache = cache # Set the global cache for Model instances

    test_queries = [
        ("What is the capital of France?", "gpt-4o-mini"),
        ("What is the highest mountain?", "gpt-4o-mini"),
        ("What is 2+2?", "gpt-4o-mini"),
        ("Name three fruits.", "haiku"),
        ("Name three vegetables.", "haiku"),
    ]

    uncached_times = []
    cached_times = []
    
    # Run uncached queries
    print("\n  Running uncached calls (should hit API and cache)...")
    for i, (query_text, model_name) in enumerate(test_queries):
        model = Model(model_name)
        start_time = time.time()
        response = model.query(query_text, cache=True, verbose=False) # Should hit API and cache
        end_time = time.time()
        duration = end_time - start_time
        uncached_times.append(duration)
        
        content_preview = get_response_content_preview(response)
        print(f"  Uncached call {i+1}: {duration:.2f}s - {content_preview[:50]}...")

    # Run cached queries
    print("\n  Running cached calls (should hit cache)...")
    for i, (query_text, model_name) in enumerate(test_queries):
        model = Model(model_name)
        start_time = time.time()
        response = model.query(query_text, cache=True, verbose=False) # Should hit cache
        end_time = time.time()
        duration = end_time - start_time
        cached_times.append(duration)

        content_preview = get_response_content_preview(response)
        print(f"  Cached call {i+1}: {duration:.2f}s - {content_preview[:50]}...")

    avg_uncached = sum(uncached_times) / len(uncached_times)
    avg_cached = sum(cached_times) / len(cached_times)

    print(f"\n  Average uncached time: {avg_uncached:.4f}s")
    print(f"  Average cached time: {avg_cached:.4f}s")

    assert avg_cached < avg_uncached / 2, "Cached calls should be significantly faster than uncached calls."
    assert cache.retrieve_cached_requests(), "Cache should not be empty."

    print("  ✅ Test 1: String responses cache successful!")


# Example of a structured response model (add this if not already present in your project)
class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int = 0
    continent: str = ""

def test_structured_response_cache():
    """Test caching for sync Model.query calls with structured (Pydantic) responses."""
    print("\n" + "="*60)
    print("📋 Testing Structured Response Cache")
    print("----------------------------------------")

    cache = ChainCache("test_structured_cache.db")
    cache.clear_cache()
    Model._chain_cache = cache # Set the global cache for Model instances

    from Chain.parser.parser import Parser
    parser_obj = Parser(CountryInfo) # Assuming Chain.Parser is accessible

    test_queries = [
        ("Provide structured info for France.", "gpt-4o-mini"),
        ("Provide structured info for Japan.", "gpt-4o-mini"),
    ]

    uncached_times = []
    cached_times = []

    # Run uncached queries for structured output
    print("\n  Running uncached structured calls (should hit API and cache)...")
    for i, (query_text, model_name) in enumerate(test_queries):
        model = Model(model_name)
        start_time = time.time()
        # Pass the parser to model.query
        response = model.query(query_text, parser=parser_obj, cache=True, verbose=False)
        end_time = time.time()
        duration = end_time - start_time
        uncached_times.append(duration)
        
        # For structured responses, the 'response' itself is the Pydantic object
        print(f"  Uncached structured call {i+1}: {duration:.2f}s - Type: {type(response).__name__}, Data: {response.model_dump_json()[:50]}...")

    # Run cached queries for structured output
    print("\n  Running cached structured calls (should hit cache)...")
    for i, (query_text, model_name) in enumerate(test_queries):
        model = Model(model_name)
        start_time = time.time()
        response = model.query(query_text, parser=parser_obj, cache=True, verbose=False)
        end_time = time.time()
        duration = end_time - start_time
        cached_times.append(duration)

        print(f"  Cached structured call {i+1}: {duration:.2f}s - Type: {type(response).__name__}, Data: {response.model_dump_json()[:50]}...")

    avg_uncached = sum(uncached_times) / len(uncached_times)
    avg_cached = sum(cached_times) / len(cached_times)

    print(f"\n  Average uncached structured time: {avg_uncached:.4f}s")
    print(f"  Average cached structured time: {avg_cached:.4f}s")

    assert avg_cached < avg_uncached / 2, "Cached structured calls should be significantly faster."
    assert cache.retrieve_cached_requests(), "Structured cache should not be empty."
    assert isinstance(response, CountryInfo), "Response should be a CountryInfo Pydantic model."

    print("  ✅ Structured responses cache successful!")


# --- Main execution block ---
async def main():
    print("🧪 ChainCache Test Suite")
    print("=" * 50)
    
    test_sync_cache()
    test_structured_response_cache() # Make sure this test is called

    # You can add tests for AsyncChain caching here if needed
    # For example:
    # from Chain import AsyncChain, ModelAsync
    # async_cache = ChainCache("test_async_cache.db")
    # async_cache.clear_cache()
    # ModelAsync._chain_cache = async_cache # Set global cache for AsyncModel instances
    #
    # async_model = ModelAsync("gpt-4o-mini")
    # async_chain = AsyncChain(model=async_model)
    #
    # print("\n" + "="*60)
    # print("⚡️ Testing Async Chain Cache")
    # print("----------------------------------------")
    # print("Test 3: Async string responses")
    # prompt_strings = ["Async what is 1+1?", "Async what is 2+2?"]
    # async_responses = await async_chain.run(prompt_strings=prompt_strings, cache=True, verbose=False)
    # print(f"  Async responses: {len(async_responses)} received.")
    # for res in async_responses:
    #     print(f"    Content: {get_response_content_preview(res.content)[:50]}...")
    #
    # async_responses_cached = await async_chain.run(prompt_strings=prompt_strings, cache=True, verbose=False)
    # print(f"  Async cached responses: {len(async_responses_cached)} received (should be faster).")
    #
    # print("  ✅ Async responses cache successful!")

    print("\n" + "="*50)
    print("🎉 All Cache Tests Completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
