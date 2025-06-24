#!/usr/bin/env python3
"""
Test script for caching functionality across sync and async contexts.
Tests cache hits, misses, serialization, and cross-context compatibility.
"""

import asyncio
import time
from pathlib import Path
from Chain import Model, ModelAsync, Parser
from Chain.cache.cache import ChainCache
from Chain.result.response import Response
from Chain.result.error import ChainError
from pydantic import BaseModel


class TestFrog(BaseModel):
    name: str
    species: str
    color: str
    legs: int


def setup_cache():
    """Set up a fresh cache for testing"""
    cache_path = "test_cache.db"
    # Remove existing cache file
    if Path(cache_path).exists():
        Path(cache_path).unlink()
    
    cache = ChainCache(cache_path)
    return cache, cache_path


def test_basic_caching():
    """Test basic cache functionality with sync model"""
    print("\n" + "="*60)
    print("TESTING BASIC SYNC CACHING")
    print("="*60)

    cache, cache_path = setup_cache()
    
    # Set up model with cache
    model = Model("gpt-4o-mini")
    Model._chain_cache = cache
    
    query = "What is the capital of France?"
    
    # Test 1: First query (cache miss)
    print("\n--- Test 1: Cache Miss (First Query) ---")
    start_time = time.time()
    result1 = model.query(query, cache=True, verbose=False)
    first_duration = time.time() - start_time
    
    if isinstance(result1, Response):
        print(f"✅ First query successful: {result1.content[:50]}...")
        print(f"   Duration: {first_duration:.2f}s")
        print(f"   Query duration: {result1.duration:.2f}s")
    else:
        print(f"❌ First query failed: {result1}")
        return

    # Test 2: Second query (cache hit)
    print("\n--- Test 2: Cache Hit (Second Query) ---")
    start_time = time.time()
    result2 = model.query(query, cache=True, verbose=False)
    second_duration = time.time() - start_time
    
    if isinstance(result2, Response):
        print(f"✅ Second query successful: {result2.content[:50]}...")
        print(f"   Duration: {second_duration:.2f}s")
        print(f"   Query duration: {result2.duration:.2f}s")
        print(f"   Cache speedup: {first_duration/second_duration:.1f}x faster")
        
        # Verify content is the same
        if result1.content == result2.content:
            print("✅ Cache returned identical content")
        else:
            print("❌ Cache returned different content")
    else:
        print(f"❌ Second query failed: {result2}")

    # Test 3: Cache stats
    print("\n--- Test 3: Cache Statistics ---")
    stats = cache.cache_stats()
    print(f"✅ Cache stats: {stats}")
    
    cached_requests = cache.retrieve_cached_requests()
    print(f"✅ Cached requests: {len(cached_requests)}")

    # Cleanup
    cache.close()
    Path(cache_path).unlink()


async def test_async_caching():
    """Test caching with async model"""
    print("\n" + "="*60)
    print("TESTING ASYNC CACHING")
    print("="*60)

    cache, cache_path = setup_cache()
    
    # Set up async model with cache
    model = ModelAsync("gpt-4o-mini")
    ModelAsync._chain_cache = cache
    
    query = "What is the capital of Germany?"
    
    # Test 1: First async query (cache miss)
    print("\n--- Test 1: Async Cache Miss ---")
    start_time = time.time()
    result1 = await model.query_async(query, cache=True, verbose=False)
    first_duration = time.time() - start_time
    
    if isinstance(result1, Response):
        print(f"✅ First async query successful: {result1.content[:50]}...")
        print(f"   Duration: {first_duration:.2f}s")
    else:
        print(f"❌ First async query failed: {result1}")
        return

    # Test 2: Second async query (cache hit)
    print("\n--- Test 2: Async Cache Hit ---")
    start_time = time.time()
    result2 = await model.query_async(query, cache=True, verbose=False)
    second_duration = time.time() - start_time
    
    if isinstance(result2, Response):
        print(f"✅ Second async query successful: {result2.content[:50]}...")
        print(f"   Duration: {second_duration:.2f}s")
        print(f"   Cache speedup: {first_duration/second_duration:.1f}x faster")
    else:
        print(f"❌ Second async query failed: {result2}")

    # Test 3: Concurrent queries with caching
    print("\n--- Test 3: Concurrent Queries with Caching ---")
    queries = [
        "What is 10 + 10?",
        "What is 20 + 20?",
        "What is 10 + 10?",  # Duplicate - should hit cache
        "What is 30 + 30?",
        "What is 20 + 20?",  # Duplicate - should hit cache
    ]
    
    start_time = time.time()
    tasks = [model.query_async(q, cache=True, verbose=False) for q in queries]
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start_time
    
    success_count = sum(1 for r in results if isinstance(r, Response))
    print(f"✅ Concurrent caching test completed:")
    print(f"   Successful: {success_count}/{len(queries)}")
    print(f"   Total duration: {total_duration:.2f}s")
    
    # Check cache stats
    stats = cache.cache_stats()
    print(f"   Cache entries: {stats['total_entries']}")

    # Cleanup
    cache.close()
    Path(cache_path).unlink()


def test_cross_context_caching():
    """Test caching across sync and async contexts"""
    print("\n" + "="*60)
    print("TESTING CROSS-CONTEXT CACHING")
    print("="*60)

    cache, cache_path = setup_cache()
    
    # Set up both models with same cache
    sync_model = Model("gpt-4o-mini")
    async_model = ModelAsync("gpt-4o-mini")
    
    Model._chain_cache = cache
    ModelAsync._chain_cache = cache
    
    query = "What is the capital of Italy?"
    
    # Test 1: Sync model populates cache
    print("\n--- Test 1: Sync Model Populates Cache ---")
    sync_result = sync_model.query(query, cache=True, verbose=False)
    
    if isinstance(sync_result, Response):
        print(f"✅ Sync query cached: {sync_result.content[:50]}...")
    else:
        print(f"❌ Sync query failed: {sync_result}")
        return

    # Test 2: Async model reads from cache
    print("\n--- Test 2: Async Model Reads from Cache ---")
    
    async def test_async_read():
        start_time = time.time()
        async_result = await async_model.query_async(query, cache=True, verbose=False)
        duration = time.time() - start_time
        
        if isinstance(async_result, Response):
            print(f"✅ Async query from cache: {async_result.content[:50]}...")
            print(f"   Duration: {duration:.2f}s (should be very fast)")
            
            # Verify content matches
            if sync_result.content == async_result.content:
                print("✅ Cross-context cache content matches")
            else:
                print("❌ Cross-context cache content differs")
        else:
            print(f"❌ Async query failed: {async_result}")
    
    asyncio.run(test_async_read())

    # Test 3: Cache with structured output
    print("\n--- Test 3: Structured Output Caching ---")
    parser = Parser(TestFrog)
    frog_query = "Create a fictional frog"
    
    # Sync query with parser
    sync_frog = sync_model.query(frog_query, parser=parser, cache=True, verbose=False)
    
    async def test_structured_cache():
        # Async query with same parser (should hit cache)
        async_frog = await async_model.query_async(frog_query, parser=parser, cache=True, verbose=False)
        
        if isinstance(sync_frog, Response) and isinstance(async_frog, Response):
            print("✅ Structured output caching works")
            print(f"   Sync frog: {sync_frog.content}")
            print(f"   Async frog: {async_frog.content}")
            
            # Check if they're the same (they should be from cache)
            if sync_frog.content.name == async_frog.content.name:
                print("✅ Structured cache hit successful")
            else:
                print("❌ Structured cache miss or different content")
        else:
            print("❌ Structured output caching failed")
    
    asyncio.run(test_structured_cache())

    # Final cache stats
    print("\n--- Final Cache Statistics ---")
    stats = cache.cache_stats()
    print(f"✅ Final cache stats: {stats}")
    
    cached_requests = cache.retrieve_cached_requests()
    print(f"✅ Total cached requests: {len(cached_requests)}")
    for i, (key, created_at) in enumerate(cached_requests[:3]):  # Show first 3
        print(f"   {i+1}. {key[:50]}... (created: {created_at})")

    # Cleanup
    cache.close()
    Path(cache_path).unlink()


def test_cache_serialization():
    """Test that complex objects serialize/deserialize correctly"""
    print("\n" + "="*60)
    print("TESTING CACHE SERIALIZATION")
    print("="*60)

    cache, cache_path = setup_cache()
    model = Model("gpt-4o-mini")
    Model._chain_cache = cache
    
    # Test with different types of content
    test_cases = [
        ("Simple string", "What is 1+1?"),
        ("Math question", "Calculate the square root of 144"),
        ("Structured output", "Create a frog", Parser(TestFrog)),
    ]
    
    for name, query, *parser in test_cases:
        print(f"\n--- Testing {name} ---")
        parser_obj = parser[0] if parser else None
        
        # First query (populate cache)
        result1 = model.query(query, parser=parser_obj, cache=True, verbose=False)
        
        # Second query (from cache)
        result2 = model.query(query, parser=parser_obj, cache=True, verbose=False)
        
        if isinstance(result1, Response) and isinstance(result2, Response):
            print(f"✅ {name} caching successful")
            print(f"   Original: {str(result1.content)[:50]}...")
            print(f"   Cached: {str(result2.content)[:50]}...")
            
            # Check equality
            if str(result1.content) == str(result2.content):
                print("✅ Serialization preserved content")
            else:
                print("❌ Serialization changed content")
        else:
            print(f"❌ {name} caching failed")

    # Cleanup
    cache.close()
    Path(cache_path).unlink()


def main():
    """Run all caching tests"""
    print("🚀 Starting Cache Tests")
    
    try:
        # Basic sync caching
        test_basic_caching()
        
        # Async caching
        print("\nRunning async caching tests...")
        asyncio.run(test_async_caching())
        
        # Cross-context caching
        test_cross_context_caching()
        
        # Serialization testing
        test_cache_serialization()
        
        print("\n" + "="*60)
        print("✅ ALL CACHE TESTS COMPLETED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Cache test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
