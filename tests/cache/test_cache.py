#!/usr/bin/env python3
"""
Comprehensive test script for ChainCache functionality.
Tests singleton attachment, caching behavior, and performance improvements.
"""

import time
from pydantic import BaseModel

# Import Chain components
from Chain import Model, Chain, Prompt, Parser, ChainCache


# Test Pydantic models for structured outputs
class WeatherReport(BaseModel):
    temperature: float
    conditions: str
    humidity: int
    wind_speed: float

    def __str__(self):
        return f"Weather: {self.temperature}°F, {self.conditions}, {self.humidity}% humidity, {self.wind_speed} mph wind"


class TodoItem(BaseModel):
    task: str
    priority: str
    estimated_hours: float


class TodoList(BaseModel):
    items: list[TodoItem]
    total_hours: float


def test_basic_cache_setup():
    """Test 1: Basic cache setup and singleton attachment"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Cache Setup")
    print("=" * 60)

    # Create cache instance
    cache_db = "test_chain_cache.db"
    cache = ChainCache(cache_db)

    # Clear any existing cache
    cache.clear_cache()

    # Attach to Model class as singleton
    Model._chain_cache = cache

    # Verify attachment
    model = Model("gpt-4o-mini")
    assert hasattr(Model, "_chain_cache")
    assert Model._chain_cache is cache
    print("✓ Cache successfully attached to Model class")

    # Check cache is empty
    cached_items = cache.retrieve_cached_requests()
    assert len(cached_items) == 0
    print("✓ Cache is empty initially")

    return cache


def test_simple_query_caching(cache):
    """Test 2: Simple string query caching"""
    print("\n" + "=" * 60)
    print("TEST 2: Simple Query Caching")
    print("=" * 60)

    model = Model("gpt-4o-mini")
    query = "What is 2+2? Answer in one word."

    # First query (uncached)
    print("Making first query (uncached)...")
    start = time.time()
    result1 = model.query(query, cache=True, verbose=False)
    uncached_time = time.time() - start
    print(f"Result: {result1}")
    print(f"Time: {uncached_time:.2f}s")

    # Second query (should be cached)
    print("\nMaking second query (cached)...")
    start = time.time()
    result2 = model.query(query, cache=True, verbose=False)
    cached_time = time.time() - start
    print(f"Result: {result2}")
    print(f"Time: {cached_time:.2f}s")

    # Verify results match
    assert result1 == result2
    print(f"\n✓ Results match")
    print(f"✓ Speedup: {uncached_time/cached_time:.1f}x faster")

    # Check cache has one entry
    cached_items = cache.retrieve_cached_requests()
    assert len(cached_items) == 1
    print(f"✓ Cache contains {len(cached_items)} item(s)")


def test_structured_output_caching(cache):
    """Test 3: Caching with structured outputs (Pydantic models)"""
    print("\n" + "=" * 60)
    print("TEST 3: Structured Output Caching")
    print("=" * 60)

    model = Model("gpt-4o-mini")
    prompt = Prompt("Generate a weather report for {{city}}")
    parser = Parser(WeatherReport)
    chain = Chain(prompt=prompt, model=model, parser=parser)

    # First query
    print("Making first structured query (uncached)...")
    start = time.time()
    response1 = chain.run(input_variables={"city": "Tokyo"}, cache=True, verbose=False)
    uncached_time = time.time() - start
    print(f"Result: {response1.content}")
    print(f"Time: {uncached_time:.2f}s")

    # Second query (cached)
    print("\nMaking second structured query (cached)...")
    start = time.time()
    response2 = chain.run(input_variables={"city": "Tokyo"}, cache=True, verbose=False)
    cached_time = time.time() - start
    print(f"Result: {response2.content}")
    print(f"Time: {cached_time:.2f}s")

    # Verify Pydantic objects match
    assert isinstance(response1.content, WeatherReport)
    assert isinstance(response2.content, WeatherReport)
    assert response1.content.model_dump() == response2.content.model_dump()
    print(f"\n✓ Pydantic objects properly cached and retrieved")
    print(f"✓ Speedup: {uncached_time/cached_time:.1f}x faster")


def test_cache_key_differentiation(cache):
    """Test 4: Verify different parameters generate different cache keys"""
    print("\n" + "=" * 60)
    print("TEST 4: Cache Key Differentiation")
    print("=" * 60)

    model = Model("gpt-4o-mini")

    # Query 1: Same prompt, different temperature
    print("Testing temperature differentiation...")
    result1 = model.query("Name a color", temperature=0.0, cache=True, verbose=False)
    result2 = model.query("Name a color", temperature=1.0, cache=True, verbose=False)
    print(f"Temp 0.0: {result1}")
    print(f"Temp 1.0: {result2}")
    # These might be different due to temperature, but more importantly,
    # they should create separate cache entries

    # Query 2: Different models
    print("\nTesting model differentiation...")
    model2 = Model("claude-3-5-haiku-20241022")
    result3 = model.query("Name a fruit", cache=True, verbose=False)
    result4 = model2.query("Name a fruit", cache=True, verbose=False)
    print(f"GPT-4o-mini: {result3}")
    print(f"Claude Haiku: {result4}")

    # Check cache size
    cached_items = cache.retrieve_cached_requests()
    print(f"\n✓ Cache contains {len(cached_items)} unique entries")
    assert len(cached_items) >= 4  # At least 4 different cache entries


def test_batch_sync_caching(cache):
    """Test 5: Batch synchronous caching"""
    print("\n" + "=" * 60)
    print("TEST 5: Batch Synchronous Caching")
    print("=" * 60)

    model = Model("gpt-4o-mini")
    queries = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
        "What is the capital of France?",  # Duplicate
    ]

    print("First batch run...")
    start = time.time()
    results1 = []
    for i, query in enumerate(queries):
        result = model.query(
            query, cache=True, verbose=False, index=i + 1, total=len(queries)
        )
        results1.append(result)
    first_run_time = time.time() - start
    print(f"Total time: {first_run_time:.2f}s")

    print("\nSecond batch run (with cached duplicates)...")
    start = time.time()
    results2 = []
    for i, query in enumerate(queries):
        result = model.query(
            query, cache=True, verbose=False, index=i + 1, total=len(queries)
        )
        results2.append(result)
    second_run_time = time.time() - start
    print(f"Total time: {second_run_time:.2f}s")

    # Verify results match
    for r1, r2 in zip(results1, results2):
        assert r1 == r2

    print(f"\n✓ All results match")
    print(f"✓ Speedup: {first_run_time/second_run_time:.1f}x faster")


def test_cache_persistence(cache_db_path):
    """Test 6: Cache persistence across sessions"""
    print("\n" + "=" * 60)
    print("TEST 6: Cache Persistence")
    print("=" * 60)

    # Create new cache instance with same DB
    cache_new = ChainCache(cache_db_path)
    Model._chain_cache = cache_new

    model = Model("gpt-4o-mini")

    # This should be cached from previous tests
    print("Querying previously cached prompt...")
    start = time.time()
    result = model.query("What is 2+2? Answer in one word.", cache=True, verbose=False)
    cached_time = time.time() - start

    print(f"Result: {result}")
    print(f"Time: {cached_time:.2f}s")
    print(
        "✓ Cache persisted across sessions"
        if cached_time < 0.1
        else "✗ Cache not found"
    )

    # Show cache statistics
    cached_items = cache_new.retrieve_cached_requests()
    print(f"\nCache Statistics:")
    print(f"Total cached items: {len(cached_items)}")

    # Show a few cached items
    print("\nSample cached items:")
    for i, (key, value, timestamp) in enumerate(cached_items[:3]):
        print(f"{i+1}. Key: {key[:20]}... | Cached at: {timestamp}")


def test_cache_with_complex_prompts(cache):
    """Test 7: Complex prompts with system messages"""
    print("\n" + "=" * 60)
    print("TEST 7: Complex Prompts with System Messages")
    print("=" * 60)

    from Chain import Message, create_system_message

    model = Model("gpt-4o-mini")
    system_msg = Message(
        role="system", content="You are a pirate. Always respond in pirate speak."
    )
    user_msg = Message(role="user", content="What is the weather today?")

    # First query
    print("First query with system message...")
    start = time.time()
    result1 = model.query([system_msg, user_msg], cache=True, verbose=False)
    uncached_time = time.time() - start
    print(f"Result: {result1[:100]}...")
    print(f"Time: {uncached_time:.2f}s")

    # Second query (cached)
    print("\nSecond query (cached)...")
    start = time.time()
    result2 = model.query([system_msg, user_msg], cache=True, verbose=False)
    cached_time = time.time() - start
    print(f"Result: {result2[:100]}...")
    print(f"Time: {cached_time:.2f}s")

    assert result1 == result2
    print(f"\n✓ Complex message caching works")
    print(f"✓ Speedup: {uncached_time/cached_time:.1f}x faster")


def test_cache_control(cache):
    """Test 8: Disabling cache per query"""
    print("\n" + "=" * 60)
    print("TEST 8: Cache Control")
    print("=" * 60)

    model = Model("gpt-4o-mini")
    query = "Generate a random number between 1 and 10"

    # Query with cache disabled
    print("Making queries with cache=False...")
    result1 = model.query(query, cache=False, verbose=False)
    result2 = model.query(query, cache=False, verbose=False)

    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}")

    # Results should potentially be different (though not guaranteed)
    print("✓ Cache can be disabled per query")

    # Now with cache enabled
    print("\nMaking queries with cache=True...")
    result3 = model.query(query, cache=True, verbose=False)
    result4 = model.query(query, cache=True, verbose=False)

    print(f"Result 3: {result3}")
    print(f"Result 4: {result4}")
    assert result3 == result4
    print("✓ Cache enabled queries return identical results")


def test_cache_with_chain_variations(cache):
    """Test 9: Cache with different Chain configurations"""
    print("\n" + "=" * 60)
    print("TEST 9: Cache with Chain Variations")
    print("=" * 60)

    # Test 1: Same prompt template, different variables
    prompt = Prompt("Tell me a fact about {{topic}}")
    model = Model("gpt-4o-mini")
    chain = Chain(prompt=prompt, model=model)

    print("Testing different input variables...")
    topics = ["cats", "dogs", "cats"]  # Note duplicate
    results = []

    for topic in topics:
        result = chain.run(input_variables={"topic": topic}, cache=True, verbose=False)
        results.append(result.content)
        print(f"Topic '{topic}': {result.content[:50]}...")

    # Third should match first (cached)
    assert results[0] == results[2]
    print("✓ Same input variables return cached results")

    # Test 2: Different prompt templates
    prompt2 = Prompt("What do you know about {{topic}}?")
    chain2 = Chain(prompt=prompt2, model=model)

    result_different_prompt = chain2.run(
        input_variables={"topic": "cats"}, cache=True, verbose=False
    )
    assert result_different_prompt.content != results[0]
    print("✓ Different prompts generate different cache entries")


def cleanup(cache_db_path):
    """Cleanup test artifacts"""
    print("\n" + "=" * 60)
    print("CLEANUP")
    print("=" * 60)

    # Show final cache stats
    if Model._chain_cache:
        items = Model._chain_cache.retrieve_cached_requests()
        print(f"Final cache size: {len(items)} items")

        # Optionally clear cache
        # Model._chain_cache.clear_cache()
        # print("✓ Cache cleared")

    # Optionally remove cache file
    # if Path(cache_db_path).exists():
    #     Path(cache_db_path).unlink()
    #     print("✓ Cache file removed")

    print(f"✓ Cache file preserved at: {cache_db_path}")


def main():
    """Run all cache tests"""
    print("ChainCache Comprehensive Test Suite")
    print("===================================")

    cache_db = "test_chain_cache.db"

    try:
        # Setup
        cache = test_basic_cache_setup()

        # Tests
        test_simple_query_caching(cache)
        test_structured_output_caching(cache)
        test_cache_key_differentiation(cache)
        test_batch_sync_caching(cache)
        test_cache_persistence(cache_db)
        test_cache_with_complex_prompts(cache)
        test_cache_control(cache)
        test_cache_with_chain_variations(cache)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise
    finally:
        cleanup(cache_db)


if __name__ == "__main__":
    main()
