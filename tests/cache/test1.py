#!/usr/bin/env python3
"""
Test script for Model.query and ModelAsync.query_async methods.
Tests basic functionality, error handling, and return types.
"""

import asyncio
from rich.console import Console
from Chain import Model, ModelAsync, Prompt, Parser
from Chain.result.result import ChainResult
from Chain.result.response import Response
from Chain.result.error import ChainError
from Chain.message.message import Message
from pydantic import BaseModel


class TestAnimal(BaseModel):
    name: str
    species: str
    age: int
    habitat: str


def test_sync_queries():
    """Test synchronous Model.query method"""
    print("\n" + "="*60)
    print("TESTING SYNC MODEL.QUERY")
    print("="*60)

    model = Model("gpt-4o-mini")
    
    # Test 1: Basic string query
    print("\n--- Test 1: Basic String Query ---")
    result = model.query("What is 2+2?", verbose=False)
    
    if isinstance(result, Response):
        print(f"✅ Success: {result.content[:50]}...")
        print(f"   Type: {type(result)}")
        print(f"   Duration: {result.duration:.2f}s")
    elif isinstance(result, ChainError):
        print(f"❌ Error: {result.info.message}")
    else:
        print(f"❓ Unexpected type: {type(result)}")

    # Test 2: Query with parser (structured output)
    print("\n--- Test 2: Structured Output Query ---")
    parser = Parser(TestAnimal)
    result = model.query(
        "Create a fictional animal with name, species, age, and habitat",
        parser=parser,
        verbose=False
    )
    
    if isinstance(result, Response):
        print(f"✅ Success: Structured output received")
        print(f"   Animal: {result.content}")
        print(f"   Type: {type(result.content)}")
    elif isinstance(result, ChainError):
        print(f"❌ Error: {result.info.message}")

    # Test 3: Query with Message objects
    print("\n--- Test 3: Message Objects Query ---")
    messages = [
        Message(role="system", content="You are a helpful math tutor"),
        Message(role="user", content="Explain what 5 * 6 equals")
    ]
    result = model.query(messages, verbose=False)
    
    if isinstance(result, Response):
        print(f"✅ Success: {result.content[:50]}...")
        print(f"   Messages count: {len(result.messages)}")
    elif isinstance(result, ChainError):
        print(f"❌ Error: {result.info.message}")

    # Test 4: Return params for debugging
    print("\n--- Test 4: Return Params ---")
    params = model.query("Test", return_params=True)
    print(f"✅ Params returned: {type(params)}")
    print(f"   Model: {params.model}")
    print(f"   Temperature: {params.temperature}")

async def test_async_queries():
    """Test asynchronous ModelAsync.query_async method"""
    print("\n" + "="*60)
    print("TESTING ASYNC MODEL.QUERY_ASYNC")
    print("="*60)

    model = ModelAsync("gpt-4o-mini")
    
    # Test 1: Basic async query
    print("\n--- Test 1: Basic Async Query ---")
    result = await model.query_async("What is 3+3?", verbose=False)
    
    if isinstance(result, Response):
        print(f"✅ Success: {result.content[:50]}...")
        print(f"   Type: {type(result)}")
        print(f"   Duration: {result.duration:.2f}s")
    elif isinstance(result, ChainError):
        print(f"❌ Error: {result.info.message}")

    # Test 2: Async query with parser
    print("\n--- Test 2: Async Structured Output ---")
    parser = Parser(TestAnimal)
    result = await model.query_async(
        "Create a fictional sea creature",
        parser=parser,
        verbose=False
    )
    
    if isinstance(result, Response):
        print(f"✅ Success: Structured output received")
        print(f"   Animal: {result.content}")
    elif isinstance(result, ChainError):
        print(f"❌ Error: {result.info.message}")

    # Test 3: Multiple concurrent queries
    print("\n--- Test 3: Concurrent Queries ---")
    queries = [
        "What is 1+1?",
        "What is 2+2?", 
        "What is 3+3?",
        "What is 4+4?"
    ]
    
    tasks = [model.query_async(query, verbose=False) for query in queries]
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for r in results if isinstance(r, Response))
    error_count = sum(1 for r in results if isinstance(r, ChainError))
    
    print(f"✅ Concurrent queries completed:")
    print(f"   Successful: {success_count}/{len(queries)}")
    print(f"   Errors: {error_count}/{len(queries)}")
    
    for i, result in enumerate(results):
        if isinstance(result, Response):
            print(f"   Query {i+1}: {result.content[:30]}...")
        else:
            print(f"   Query {i+1}: Error - {result.info.message}")

def test_return_types():
    """Test that return types are correct"""
    print("\n" + "="*60)
    print("TESTING RETURN TYPES")
    print("="*60)

    model = Model("gpt-4o-mini")
    
    # Test ChainResult union type
    result = model.query("Simple test", verbose=False)
    print(f"Sync result type: {type(result)}")
    print(f"Is Response: {isinstance(result, Response)}")
    print(f"Is ChainError: {isinstance(result, ChainError)}")
    
    async def check_async_types():
        model_async = ModelAsync("gpt-4o-mini")
        result = await model_async.query_async("Simple async test", verbose=False)
        print(f"Async result type: {type(result)}")
        print(f"Is Response: {isinstance(result, Response)}")
        print(f"Is ChainError: {isinstance(result, ChainError)}")
    
    asyncio.run(check_async_types())


def main():
    """Run all query tests"""
    print("🚀 Starting Model Query Tests")
    
    try:
        # Test sync queries
        test_sync_queries()
        
        # Test async queries
        print("\nRunning async tests...")
        asyncio.run(test_async_queries())
        
        # Test return types
        test_return_types()
        
        print("\n" + "="*60)
        print("✅ ALL QUERY TESTS COMPLETED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
