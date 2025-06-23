#!/usr/bin/env python3
"""
Debug script to isolate the cache issue
"""

from Chain import Model
from Chain.cache.cache import ChainCache
import time

def debug_cache_issue():
    """Step by step debugging of the cache issue"""
    
    print("🔍 Debugging Cache Issue")
    print("=" * 50)
    
    # Setup
    cache = ChainCache("debug_cache.db")
    cache.clear_cache()  # Start fresh
    
    model = Model("gpt-4o-mini")
    Model._chain_cache = cache
    
    print(f"Model type: {type(model)}")
    print(f"Model._chain_cache: {model._chain_cache}")
    print(f"Cache instance: {cache}")
    
    # Test 1: Direct client call (should work)
    print("\n1. Testing direct client call:")
    try:
        from Chain.model.params.params import Params
        params = Params(
            model="gpt-4o-mini",
            query_input="What is 2+2?"
        )
        print(f"Params: {params}")
        
        direct_result = model._client.query(params)
        print(f"Direct client result type: {type(direct_result)}")
        print(f"Direct client result: {direct_result[:50] if isinstance(direct_result, str) else direct_result}")
        
    except Exception as e:
        print(f"Direct client error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Model.query with cache=False (should work)
    print("\n2. Testing Model.query with cache=False:")
    try:
        no_cache_result = model.query("What is 2+2?", cache=False, verbose=False)
        print(f"No cache result type: {type(no_cache_result)}")
        print(f"No cache result: {no_cache_result[:50] if isinstance(no_cache_result, str) else no_cache_result}")
        
    except Exception as e:
        print(f"No cache error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Model.query with cache=True (the failing case)
    print("\n3. Testing Model.query with cache=True:")
    try:
        cache_result = model.query("What is 2+2?", cache=True, verbose=False)
        print(f"Cache result type: {type(cache_result)}")
        print(f"Cache result: {cache_result[:50] if isinstance(cache_result, str) else cache_result}")
        
    except Exception as e:
        print(f"Cache error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check what's in the cache
    print("\n4. Checking cache contents:")
    try:
        # Try to peek at what got stored
        import sqlite3
        with sqlite3.connect("debug_cache.db") as conn:
            cursor = conn.execute("SELECT cache_key, content FROM cache")
            rows = cursor.fetchall()
            print(f"Cache entries: {len(rows)}")
            for cache_key, content in rows:
                print(f"  Key: {cache_key[:50]}...")
                print(f"  Content type: {type(content)}")
                print(f"  Content preview: {content[:100]}...")
                
    except Exception as e:
        print(f"Cache inspection error: {e}")

if __name__ == "__main__":
    debug_cache_issue()
