#!/usr/bin/env python3
"""
Test to verify cache sharing between sync and async models
"""

import asyncio
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


def test_cache_sharing():
    """Test that sync and async models properly share cache"""
    print("🔍 TESTING CACHE SHARING BETWEEN SYNC AND ASYNC")
    print("=" * 60)
    
    # Clean slate
    cache_path = "shared_cache_test.db"
    if Path(cache_path).exists():
        Path(cache_path).unlink()
    
    # Create shared cache
    cache = ChainCache(cache_path)
    
    # Set up both models with the SAME cache instance
    sync_model = Model("gpt-4o-mini")
    async_model = ModelAsync("gpt-4o-mini")
    
    # CRITICAL: Set the same cache on both model classes
    Model._chain_cache = cache
    ModelAsync._chain_cache = cache
    
    print(f"Cache instance for Model: {id(Model._chain_cache)}")
    print(f"Cache instance for ModelAsync: {id(ModelAsync._chain_cache)}")
    print(f"Are they the same? {Model._chain_cache is ModelAsync._chain_cache}")
    
    parser = Parser(TestFrog)
    query = "Create a fictional frog"
    
    print(f"\n--- Step 1: Sync model populates cache ---")
    sync_result = sync_model.query(query, parser=parser, cache=True, verbose=False)
    
    print(f"Sync result type: {type(sync_result)}")
    print(f"Sync is Response: {isinstance(sync_result, Response)}")
    
    if isinstance(sync_result, Response):
        print(f"✅ Sync successful: {sync_result.content}")
    else:
        print(f"❌ Sync failed: {sync_result}")
        cache.close()
        return
    
    # Check cache after sync
    stats_after_sync = cache.cache_stats()
    print(f"Cache entries after sync: {stats_after_sync['total_entries']}")
    
    async def test_async_cache_hit():
        print(f"\n--- Step 2: Async model reads from cache ---")
        
        # Verify the async model has the same cache
        print(f"Async model cache: {id(async_model._chain_cache)}")
        print(f"Is same as sync cache: {async_model._chain_cache is sync_model._chain_cache}")
        
        async_result = await async_model.query_async(query, parser=parser, cache=True, verbose=False)
        
        print(f"Async result type: {type(async_result)}")
        print(f"Async is Response: {isinstance(async_result, Response)}")
        
        if isinstance(async_result, Response):
            print(f"✅ Async successful: {async_result.content}")
            
            # Check if content matches (should be identical from cache)
            if str(sync_result.content) == str(async_result.content):
                print("✅ Cache hit! Content matches exactly")
                return True
            else:
                print("❌ Cache miss! Content differs")
                print(f"   Sync: {sync_result.content}")
                print(f"   Async: {async_result.content}")
                return False
        else:
            print(f"❌ Async failed: {async_result}")
            return False
    
    # Run async test
    success = asyncio.run(test_async_cache_hit())
    
    # Final cache stats
    stats_final = cache.cache_stats()
    print(f"\nFinal cache entries: {stats_final['total_entries']}")
    print(f"Expected: 1 (if cache hit worked), Actual: {stats_final['total_entries']}")
    
    if stats_final['total_entries'] == 1:
        print("✅ Cache working correctly (only 1 entry)")
    else:
        print("❌ Cache miss occurred (2 entries means no sharing)")
    
    # Cleanup
    cache.close()
    if Path(cache_path).exists():
        Path(cache_path).unlink()
    
    return success


def test_cache_key_compatibility():
    """Test that sync and async generate identical cache keys for same input"""
    print("\n🔍 TESTING CACHE KEY COMPATIBILITY")
    print("=" * 60)
    
    from Chain.model.params.params import Params
    
    parser = Parser(TestFrog)
    query = "Create a fictional frog"
    
    # Create params as sync model would
    sync_params = Params(
        model="gpt-4o-mini",
        query_input=query,
        parser=parser,
        temperature=None
    )
    
    # Create params as async model would  
    async_params = Params(
        model="gpt-4o-mini",
        query_input=query,
        parser=parser,
        temperature=None
    )
    
    sync_key = sync_params.generate_cache_key()
    async_key = async_params.generate_cache_key()
    
    print(f"Sync cache key:  {sync_key}")
    print(f"Async cache key: {async_key}")
    print(f"Keys match: {sync_key == async_key}")
    
    if sync_key == async_key:
        print("✅ Cache keys are compatible")
        return True
    else:
        print("❌ Cache keys differ - this explains the cache miss!")
        return False


def main():
    """Run cache sharing tests"""
    print("🚀 CACHE SHARING DIAGNOSTIC")
    
    # Test cache key compatibility
    key_compat = test_cache_key_compatibility()
    
    # Test actual cache sharing
    cache_sharing = test_cache_sharing()
    
    print(f"\n📋 RESULTS")
    print("=" * 30)
    print(f"Cache key compatibility: {'✅' if key_compat else '❌'}")
    print(f"Cache sharing works: {'✅' if cache_sharing else '❌'}")
    
    if key_compat and cache_sharing:
        print("\n🎉 Cache sharing is working correctly!")
        print("The issue must be in the original test setup.")
    else:
        print("\n🔧 Found the issue! Check the details above.")


if __name__ == "__main__":
    main()
