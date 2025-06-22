#!/usr/bin/env python3
"""
Comprehensive test script for AsyncChain progress display.
Tests current implementation and shows what concurrent progress should look like.
"""

import asyncio
import time
from rich.console import Console
from Chain import AsyncChain, ModelAsync

def test_current_async_progress():
    """Test current AsyncChain progress implementation"""
    print("\n" + "="*70)
    print("CURRENT ASYNCCHAIN PROGRESS IMPLEMENTATION")
    print("="*70)
    
    # Test 1: Plain console (no Rich)
    print("\n--- Test 1: Plain Console Progress ---")
    AsyncChain._console = None  # Reset to plain
    
    model = ModelAsync("gpt-4o-mini")
    chain = AsyncChain(model=model)
    
    prompt_strings = [
        "What is 2+2?",
        "What is 3+3?", 
        "What is 4+4?",
        "What is 5+5?"
    ]
    
    print("Running with plain console progress...")
    responses = chain.run(prompt_strings=prompt_strings, verbose=True)
    print(f"✓ Completed {len(responses)} requests\n")
    
    # Test 2: Rich console
    print("\n--- Test 2: Rich Console Progress ---")
    console = Console()
    AsyncChain._console = console
    
    chain = AsyncChain(model=model)
    
    prompt_strings = [
        "Name one color",
        "Name one animal", 
        "Name one country",
        "Name one food"
    ]
    
    print("Running with Rich console progress...")
    responses = chain.run(prompt_strings=prompt_strings, verbose=True)
    print(f"✓ Completed {len(responses)} requests\n")
    
    # Test 3: Larger batch to see timing
    print("\n--- Test 3: Larger Batch (8 requests) ---")
    
    prompt_strings = [
        "What is 1+1?", "What is 2+2?", "What is 3+3?", "What is 4+4?",
        "What is 5+5?", "What is 6+6?", "What is 7+7?", "What is 8+8?"
    ]
    
    print("Running larger batch...")
    start_time = time.time()
    responses = chain.run(prompt_strings=prompt_strings, verbose=True)
    total_time = time.time() - start_time
    print(f"✓ Completed {len(responses)} requests in {total_time:.1f}s total\n")

def simulate_ideal_concurrent_progress():
    """Show what ideal concurrent progress should look like"""
    print("\n" + "="*70)
    print("IDEAL CONCURRENT PROGRESS (SIMULATION)")
    print("="*70)
    
    console = Console()
    
    print("\n--- What Rich Concurrent Progress Should Look Like ---")
    
    # Simulate the ideal progress display
    console.print("[bold blue]⠋ Starting: 5 concurrent requests[/bold blue]")
    
    # Simulate progress updates
    for i in range(1, 6):
        time.sleep(0.3)  # Simulate work
        if i < 5:
            console.print(f"⠋ Progress: {i}/5 complete | {5-i} running | 0 failed | {i*0.3:.1f}s elapsed", end="\r")
        else:
            console.print(f"[green]✓[/green] All requests complete: 5/5 successful in {i*0.3:.1f}s")
    
    print("\n--- Alternative: Simple Spinner with Final Summary ---")
    
    with console.status("[bold blue]Running 8 concurrent requests...", spinner="dots"):
        time.sleep(2.0)  # Simulate longer work
    
    console.print("[green]✓[/green] All requests complete: 8/8 successful in 2.0s")
    
    print("\n--- What Plain Console Should Look Like ---")
    
    print("[12:34:56] Starting: 5 concurrent requests")
    time.sleep(1.0)
    print("[12:34:57] All requests complete: 5/5 successful in 1.0s")

async def test_individual_vs_concurrent():
    """Compare individual vs concurrent progress display"""
    print("\n" + "="*70)
    print("INDIVIDUAL vs CONCURRENT PROGRESS COMPARISON")
    print("="*70)
    
    console = Console()
    AsyncChain._console = console
    
    model = ModelAsync("gpt-4o-mini")
    
    # Test individual operations (should show each operation)
    print("\n--- Individual Operations (verbose=True) ---")
    for i, prompt in enumerate(["What is 1+1?", "What is 2+2?", "What is 3+3?"]):
        response = await model.query_async(prompt, verbose=True)
        print(f"Individual {i+1}: {response[:30]}...")
    
    # Test concurrent operations (should suppress individual, show batch)
    print("\n--- Concurrent Operations (batch display) ---")
    chain = AsyncChain(model=model)
    prompt_strings = ["What is 4+4?", "What is 5+5?", "What is 6+6?"]
    responses = chain.run(prompt_strings=prompt_strings, verbose=True)
    
    print("\n--- Concurrent with Progress Disabled ---")
    responses = chain.run(prompt_strings=prompt_strings, verbose=False)
    print("Silent mode: no progress shown")

def test_error_handling():
    """Test how progress handles errors and cancellation"""
    print("\n" + "="*70)
    print("ERROR HANDLING IN CONCURRENT OPERATIONS")
    print("="*70)
    
    console = Console()
    AsyncChain._console = console
    
    # Simulate what error display should look like
    console.print("[bold blue]Starting: 4 concurrent requests[/bold blue]")
    
    # Simulate mixed success/failure
    time.sleep(0.5)
    console.print("⠋ Progress: 1/4 complete | 3 running | 0 failed | 0.5s elapsed", end="\r")
    time.sleep(0.3)
    console.print("⠋ Progress: 2/4 complete | 1 running | 1 failed | 0.8s elapsed", end="\r")
    time.sleep(0.2)
    console.print("[yellow]✓[/yellow] All requests complete: 3/4 successful in 1.0s")
    
    print("\nNote: Current implementation doesn't track individual failures in concurrent mode")
    print("This is what we should implement in Phase 5")

async def run_async_tests():
    """Run the async portions of the tests"""
    await test_individual_vs_concurrent()

def main():
    """Run all tests"""
    print("ASYNCCHAIN PROGRESS DISPLAY TEST SUITE")
    print("This will show current implementation and ideal target behavior")
    
    try:
        # Test current implementation
        test_current_async_progress()
        
        # Show ideal behavior
        simulate_ideal_concurrent_progress()
        
        # Test individual operations comparison
        print("\nRunning async individual operation tests...")
        asyncio.run(run_async_tests())
        
        # Show error handling simulation
        test_error_handling()
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print("✓ Current AsyncChain progress: Basic start/complete messages")
        print("✓ Individual operations: Working with spinners/checkmarks") 
        print("🎯 Need to implement: Real-time concurrent progress tracking")
        print("🎯 Need to implement: Error tracking in concurrent operations")
        print("🎯 Need to implement: Live progress updates during batch operations")
        
        print("\nNext steps:")
        print("1. Create ConcurrentProgressTracker class")
        print("2. Add real-time progress updates during asyncio.gather()")
        print("3. Track individual operation status (success/failed)")
        print("4. Integrate with existing Rich/Plain console hierarchy")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

