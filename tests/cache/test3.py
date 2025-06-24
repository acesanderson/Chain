#!/usr/bin/env python3
"""
Detailed debug script to find the real issue with structured output caching
"""

import asyncio
from Chain import Model, ModelAsync, Parser
from Chain.cache.cache import ChainCache
from Chain.result.response import Response
from Chain.result.error import ChainError
from pydantic import BaseModel
import traceback


class TestFrog(BaseModel):
    name: str
    species: str
    color: str
    legs: int


def debug_sync_structured():
    """Debug sync structured output"""
    print("🔍 DEBUGGING SYNC STRUCTURED OUTPUT")
    print("-" * 50)
    
    cache = ChainCache("debug_sync.db")
    model = Model("gpt-4o-mini")
    Model._chain_cache = cache
    
    parser = Parser(TestFrog)
    query = "Create a fictional frog"
    
    try:
        print("Making sync query with parser...")
        result = model.query(query, parser=parser, cache=True, verbose=False)
        
        print(f"✅ Sync result type: {type(result)}")
        print(f"✅ Is Response: {isinstance(result, Response)}")
        print(f"✅ Is ChainError: {isinstance(result, ChainError)}")
        
        if isinstance(result, Response):
            print(f"✅ Content type: {type(result.content)}")
            print(f"✅ Content: {result.content}")
            return result
        elif isinstance(result, ChainError):
            print(f"❌ Error occurred: {result.info.message}")
            print(f"❌ Error code: {result.info.code}")
            if result.detail:
                print(f"❌ Stack trace: {result.detail.stack_trace}")
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            print(f"❌ Result: {result}")
            
    except Exception as e:
        print(f"❌ Exception in sync query: {e}")
        traceback.print_exc()
    finally:
        cache.close()
        import os
        if os.path.exists("debug_sync.db"):
            os.remove("debug_sync.db")
    
    return None


async def debug_async_structured():
    """Debug async structured output"""
    print("\n🔍 DEBUGGING ASYNC STRUCTURED OUTPUT")
    print("-" * 50)
    
    cache = ChainCache("debug_async.db")
    model = ModelAsync("gpt-4o-mini")
    ModelAsync._chain_cache = cache
    
    parser = Parser(TestFrog)
    query = "Create a fictional frog"
    
    try:
        print("Making async query with parser...")
        result = await model.query_async(query, parser=parser, cache=True, verbose=False)
        
        print(f"✅ Async result type: {type(result)}")
        print(f"✅ Is Response: {isinstance(result, Response)}")
        print(f"✅ Is ChainError: {isinstance(result, ChainError)}")
        
        if isinstance(result, Response):
            print(f"✅ Content type: {type(result.content)}")
            print(f"✅ Content: {result.content}")
            return result
        elif isinstance(result, ChainError):
            print(f"❌ Error occurred: {result.info.message}")
            print(f"❌ Error code: {result.info.code}")
            if result.detail:
                print(f"❌ Stack trace: {result.detail.stack_trace}")
        else:
            print(f"❌ Unexpected result type: {type(result)}")
            print(f"❌ Result: {result}")
            
    except Exception as e:
        print(f"❌ Exception in async query: {e}")
        traceback.print_exc()
    finally:
        cache.close()
        import os
        if os.path.exists("debug_async.db"):
            os.remove("debug_async.db")
    
    return None


def debug_cache_key_generation():
    """Debug cache key generation with parser"""
    print("\n🔍 DEBUGGING CACHE KEY GENERATION")
    print("-" * 50)
    
    from Chain.model.params.params import Params
    
    try:
        # Test cache key generation with and without parser
        parser = Parser(TestFrog)
        query = "Create a fictional frog"
        
        # Without parser
        params1 = Params(model="gpt-4o-mini", query_input=query)
        key1 = params1.generate_cache_key()
        print(f"✅ Cache key without parser: {key1[:32]}...")
        
        # With parser
        params2 = Params(model="gpt-4o-mini", query_input=query, parser=parser)
        key2 = params2.generate_cache_key()
        print(f"✅ Cache key with parser: {key2[:32]}...")
        
        # They should be different
        if key1 != key2:
            print("✅ Cache keys are different (good)")
        else:
            print("❌ Cache keys are the same (bad - parser not included)")
            
    except Exception as e:
        print(f"❌ Error generating cache keys: {e}")
        traceback.print_exc()


def debug_parser_object():
    """Debug the Parser object itself"""
    print("\n🔍 DEBUGGING PARSER OBJECT")
    print("-" * 50)
    
    try:
        parser = Parser(TestFrog)
        print(f"✅ Parser created: {parser}")
        print(f"✅ Parser pydantic_model: {parser.pydantic_model}")
        print(f"✅ Parser original_spec: {parser.original_spec}")
        
        # Test serialization
        parser_str = str(parser)
        print(f"✅ Parser string representation: {parser_str}")
        
    except Exception as e:
        print(f"❌ Error with parser: {e}")
        traceback.print_exc()


async def main():
    """Run all debug tests"""
    print("🚀 DETAILED STRUCTURED OUTPUT DEBUG")
    print("=" * 60)
    
    # Test parser object
    debug_parser_object()
    
    # Test cache key generation
    debug_cache_key_generation()
    
    # Test sync structured output
    sync_result = debug_sync_structured()
    
    # Test async structured output
    async_result = await debug_async_structured()
    
    # Summary
    print("\n📋 SUMMARY")
    print("-" * 50)
    print(f"Sync successful: {sync_result is not None}")
    print(f"Async successful: {async_result is not None}")
    
    if sync_result and async_result:
        print("✅ Both sync and async structured output working!")
    else:
        print("❌ One or both failed - check the detailed output above")


if __name__ == "__main__":
    asyncio.run(main())
