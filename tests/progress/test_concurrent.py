"""
Test script for concurrent progress tracking functionality.
Tests the new ConcurrentTracker, handlers, and wrappers without AsyncChain integration.
"""

import asyncio
import time
from datetime import datetime

# Import the new progress components
from Chain.progress.tracker import ConcurrentTracker, ConcurrentSummaryEvent, AsyncEvent
from Chain.progress.handlers import PlainProgressHandler, RichProgressHandler
from Chain.progress.wrappers import concurrent_wrapper, create_concurrent_progress_tracker

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich not available - testing PlainProgressHandler only")


async def mock_llm_operation(operation_id: int, duration: float = 1.0, should_fail: bool = False):
    """Mock LLM operation that takes some time and optionally fails"""
    await asyncio.sleep(duration)
    if should_fail:
        raise Exception(f"Mock operation {operation_id} failed")
    return f"Result from operation {operation_id}"


def test_plain_progress_handler():
    """Test PlainProgressHandler with concurrent operations"""
    print("\n" + "="*60)
    print("Testing PlainProgressHandler")
    print("="*60)
    
    handler = PlainProgressHandler()
    tracker = ConcurrentTracker(handler, total=5)
    
    # Test concurrent start
    tracker.emit_concurrent_start()
    
    # Simulate some operations completing
    for i in range(5):
        tracker.operation_started()
        time.sleep(0.1)  # Small delay to show progression
        if i == 3:  # Simulate one failure
            tracker.operation_failed()
        else:
            tracker.operation_completed()
    
    # Test concurrent complete
    tracker.emit_concurrent_complete()


def test_rich_progress_handler():
    """Test RichProgressHandler with concurrent operations"""
    if not RICH_AVAILABLE:
        print("Skipping Rich handler test - Rich not available")
        return
        
    print("\n" + "="*60)
    print("Testing RichProgressHandler")
    print("="*60)
    
    console = Console()
    handler = RichProgressHandler(console)
    tracker = ConcurrentTracker(handler, total=5)
    
    # Test individual operations (should show since not in concurrent mode)
    handler.emit_started("gpt-4o", "What is 2+2?")
    time.sleep(0.5)
    handler.emit_complete("gpt-4o", "What is 2+2?", 0.5)
    
    # Test concurrent operations
    tracker.emit_concurrent_start()
    
    # Simulate some operations completing (should be suppressed)
    for i in range(5):
        handler.emit_started("gpt-4o", f"Operation {i}")  # Should be suppressed
        tracker.operation_started()
        time.sleep(0.1)
        if i == 3:  # Simulate one failure
            handler.emit_failed("gpt-4o", f"Operation {i}", "Mock error")  # Should be suppressed
            tracker.operation_failed()
        else:
            handler.emit_complete("gpt-4o", f"Operation {i}", 0.1)  # Should be suppressed
            tracker.operation_completed()
    
    # Test concurrent complete
    tracker.emit_concurrent_complete()


async def test_concurrent_wrapper():
    """Test the concurrent_wrapper function with real async operations"""
    print("\n" + "="*60)
    print("Testing concurrent_wrapper with async operations")
    print("="*60)
    
    # Create tracker
    tracker = create_concurrent_progress_tracker(None, total=4)  # Plain handler
    
    # Start concurrent operations
    tracker.emit_concurrent_start()
    
    # Create mock operations with different durations and one failure
    operations = [
        mock_llm_operation(1, 0.5),
        mock_llm_operation(2, 0.3), 
        mock_llm_operation(3, 0.7, should_fail=True),  # This one will fail
        mock_llm_operation(4, 0.4)
    ]
    
    # Wrap operations with concurrent tracking
    wrapped_operations = [
        concurrent_wrapper(op, tracker) for op in operations
    ]
    
    # Run all operations concurrently
    results = await asyncio.gather(*wrapped_operations, return_exceptions=True)
    
    # Complete tracking
    tracker.emit_concurrent_complete()
    
    # Show results
    print(f"\nResults: {len([r for r in results if not isinstance(r, Exception)])} successful, "
          f"{len([r for r in results if isinstance(r, Exception)])} failed")
    
    return results


async def test_full_simulation():
    """Full simulation of how AsyncChain would use the progress system"""
    print("\n" + "="*60)
    print("Testing full concurrent progress simulation")
    print("="*60)
    
    # Test with Rich if available, otherwise Plain
    console = Console() if RICH_AVAILABLE else None
    tracker = create_concurrent_progress_tracker(console, total=6)
    
    tracker.emit_concurrent_start()
    
    # Simulate multiple concurrent LLM calls
    prompt_strings = [
        "What is 2+2?",
        "Name five animals", 
        "Explain gravity",
        "What is Python?",
        "Define machine learning",
        "How does the internet work?"
    ]
    
    operations = [
        mock_llm_operation(i, duration=0.2 + (i * 0.1), should_fail=(i == 2))  # Operation 2 will fail
        for i, prompt in enumerate(prompt_strings)
    ]
    
    wrapped_operations = [
        concurrent_wrapper(op, tracker) for op in operations
    ]
    
    results = await asyncio.gather(*wrapped_operations, return_exceptions=True)
    tracker.emit_concurrent_complete()
    
    # Summary
    successful = len([r for r in results if not isinstance(r, Exception)])
    failed = len([r for r in results if isinstance(r, Exception)])
    print(f"\nFinal summary: {successful}/{len(operations)} successful, {failed} failed")


def main():
    """Run all tests"""
    print("Testing Concurrent Progress Tracking System")
    print("This simulates how AsyncChain will use the progress system")
    
    # Test handlers directly
    test_plain_progress_handler()
    
    if RICH_AVAILABLE:
        test_rich_progress_handler()
    
    # Test async functionality
    print("\n" + "="*60)
    print("Testing async concurrent operations")
    print("="*60)
    
    asyncio.run(test_concurrent_wrapper())
    asyncio.run(test_full_simulation())
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("Next step: Integrate this into AsyncChain._run_prompt_strings()")
    print("="*60)


if __name__ == "__main__":
    main()
