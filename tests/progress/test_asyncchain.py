#!/usr/bin/env python3
"""
Test script for Chain progress tracking implementation.
Run this to verify that sync/async progress display works correctly.
"""

import asyncio
from rich.console import Console
from Chain import Chain, AsyncChain, Model, ModelAsync, Prompt

def test_sync_progress():
    """Test 1: Sync Model progress (individual operations)"""
    print("\n" + "="*60)
    print("TEST 1: Sync Model Progress (Plain Text)")
    print("="*60)
    
    # Test without console (plain text)
    model = Model("gpt-4o-mini")
    
    print("\n--- Single sync operation with progress ---")
    response = model.query("What is 2+2?", verbose=True)
    print(f"Result: {response[:50]}...")
    
    print("\n--- Single sync operation without progress ---")
    response = model.query("What is 3+3?", verbose=False)
    print(f"Result: {response[:50]}...")

def test_sync_progress_rich():
    """Test 2: Sync Model progress with Rich console"""
    print("\n" + "="*60)
    print("TEST 2: Sync Model Progress (Rich Console)")
    print("="*60)
    
    # Test with Rich console
    console = Console()
    model = Model("gpt-4o-mini")
    model.console = console
    
    print("\n--- Single sync operation with Rich progress ---")
    response = model.query("What is the capital of France?", verbose=True)
    print(f"Result: {response[:50]}...")

def test_async_individual():
    """Test 3: Individual async operations"""
    print("\n" + "="*60)
    print("TEST 3: Individual Async Operations")
    print("="*60)
    
    async def run_individual_async():
        model = ModelAsync("gpt-4o-mini")
        
        print("\n--- Single async operation with progress ---")
        response = await model.query_async("What is 4+4?", verbose=True)
        print(f"Result: {response[:50]}...")
        
        print("\n--- Single async operation without progress ---")
        response = await model.query_async("What is 5+5?", verbose=False)
        print(f"Result: {response[:50]}...")
    
    asyncio.run(run_individual_async())

def test_async_individual_rich():
    """Test 4: Individual async operations with Rich"""
    print("\n" + "="*60)
    print("TEST 4: Individual Async Operations (Rich Console)")
    print("="*60)
    
    async def run_individual_async_rich():
        console = Console()
        model = ModelAsync("gpt-4o-mini")
        model.console = console
        
        print("\n--- Single async operation with Rich progress ---")
        response = await model.query_async("What is the capital of Germany?", verbose=True)
        print(f"Result: {response[:50]}...")
    
    asyncio.run(run_individual_async_rich())

def test_asyncchain_prompt_strings():
    """Test 5: AsyncChain concurrent operations (prompt_strings)"""
    print("\n" + "="*60)
    print("TEST 5: AsyncChain Concurrent Progress (prompt_strings)")
    print("="*60)
    
    model = ModelAsync("gpt-4o-mini")
    chain = AsyncChain(model=model)
    
    prompt_strings = [
        "What is 2+2?",
        "What is 3+3?", 
        "What is 4+4?",
        "What is 5+5?",
        "What is 6+6?"
    ]
    
    print("\n--- AsyncChain with concurrent progress (Plain) ---")
    responses = chain.run(prompt_strings=prompt_strings, verbose=True)
    print(f"Got {len(responses)} responses")
    
    print("\n--- AsyncChain without progress ---")
    responses = chain.run(prompt_strings=prompt_strings, verbose=False)
    print(f"Got {len(responses)} responses")

def test_asyncchain_prompt_strings_rich():
    """Test 6: AsyncChain concurrent operations with Rich console"""
    print("\n" + "="*60)
    print("TEST 6: AsyncChain Concurrent Progress (Rich Console)")
    print("="*60)
    
    # Set Rich console at class level
    console = Console()
    AsyncChain._console = console
    
    model = ModelAsync("gpt-4o-mini")
    chain = AsyncChain(model=model)
    
    prompt_strings = [
        "Name one color",
        "Name one animal", 
        "Name one country",
        "Name one food"
    ]
    
    print("\n--- AsyncChain with Rich concurrent progress ---")
    responses = chain.run(prompt_strings=prompt_strings, verbose=True)
    print(f"Got {len(responses)} responses")

def test_asyncchain_input_variables():
    """Test 7: AsyncChain concurrent operations (input_variables)"""
    print("\n" + "="*60)
    print("TEST 7: AsyncChain Input Variables Concurrent Progress")
    print("="*60)
    
    model = ModelAsync("gpt-4o-mini")
    prompt = Prompt("What is {{num1}} + {{num2}}?")
    chain = AsyncChain(model=model, prompt=prompt)
    
    input_variables_list = [
        {"num1": 10, "num2": 20},
        {"num1": 15, "num2": 25},
        {"num1": 30, "num2": 40},
        {"num1": 50, "num2": 60}
    ]
    
    print("\n--- AsyncChain input_variables with progress ---")
    responses = chain.run(input_variables_list=input_variables_list, verbose=True)
    print(f"Got {len(responses)} responses")

def test_asyncchain_input_variables_rich():
    """Test 8: AsyncChain input_variables with Rich console"""
    print("\n" + "="*60)
    print("TEST 8: AsyncChain Input Variables (Rich Console)")
    print("="*60)
    
    console = Console()
    AsyncChain._console = console
    
    model = ModelAsync("gpt-4o-mini")
    prompt = Prompt("Name a {{thing}} that is {{color}}")
    chain = AsyncChain(model=model, prompt=prompt)
    
    input_variables_list = [
        {"thing": "fruit", "color": "red"},
        {"thing": "animal", "color": "brown"},
        {"thing": "car", "color": "blue"}
    ]
    
    print("\n--- AsyncChain input_variables with Rich progress ---")
    responses = chain.run(input_variables_list=input_variables_list, verbose=True)
    print(f"Got {len(responses)} responses")

def test_console_hierarchy():
    """Test 9: Console hierarchy resolution"""
    print("\n" + "="*60)
    print("TEST 9: Console Hierarchy Resolution")
    print("="*60)
    
    # Reset console state
    Chain._console = None
    AsyncChain._console = None
    
    model1 = Model("gpt-4o-mini")
    model2 = ModelAsync("gpt-4o-mini")
    
    print(f"Model1 console (should be None): {model1.console}")
    print(f"Model2 console (should be None): {model2.console}")
    
    # Set Chain class console
    console1 = Console()
    Chain._console = console1
    
    print(f"Model1 console (should be Console): {model1.console is console1}")
    
    # Set AsyncChain class console
    console2 = Console()
    AsyncChain._console = console2
    
    print(f"Model2 console (should be Console): {model2.console is console2}")
    
    # Test instance override
    console3 = Console()
    model1.console = console3
    print(f"Model1 console after instance override: {model1.console is console3}")

def run_all_tests():
    """Run all tests in sequence"""
    print("Starting Chain Progress Tracking Tests...")
    print("This will test sync/async individual and concurrent operations")
    
    try:
        test_sync_progress()
        test_sync_progress_rich()
        test_async_individual()
        test_async_individual_rich()
        test_asyncchain_prompt_strings()
        test_asyncchain_prompt_strings_rich()
        test_asyncchain_input_variables()
        test_asyncchain_input_variables_rich()
        test_console_hierarchy()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED!")
        print("="*60)
        print("\nWhat to look for:")
        print("- Individual operations show: [timestamp] [model] Starting/Complete")
        print("- Concurrent operations show: 'Starting: X concurrent requests' and 'All requests complete'")
        print("- Rich console shows colored output with ⠋ ✓ symbols")
        print("- Plain console shows timestamped text output")
        print("- verbose=False suppresses all progress output")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
