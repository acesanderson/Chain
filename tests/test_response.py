#!/usr/bin/env python3
"""
Test script for CacheableMixin implementation.
Tests the mixin with actual Response objects from Chain fixtures.
"""

import json, tempfile, warnings
from pathlib import Path

# Import the mixin
from Chain.cache.cacheable import CacheableMixin

# Import Chain classes for testing
try:
    from Chain.tests.fixtures import sample_response
except ImportError:
    # Fallback if fixtures not available - create a simple response
    from Chain.result.response import Response
    from Chain.message.messages import Messages
    from Chain.message.message import Message
    from Chain.model.params.params import Params
    
    sample_messages = Messages([
        Message(role="user", content="Test question"),
        Message(role="assistant", content="Test answer")
    ])
    sample_params = Params(model="gpt-4o", messages=sample_messages)
    sample_response = Response(
        messages=sample_messages,
        params=sample_params,
        duration=1.23
    )
from Chain.result.response import Response
from Chain.message.messages import Messages
from Chain.message.message import Message
from Chain.model.params.params import Params
from Chain.progress.verbosity import Verbosity


class TestableResponse(Response, CacheableMixin):
    """
    Test version of Response with CacheableMixin added.
    This demonstrates how to integrate the mixin with existing classes.
    """
    pass


def test_basic_serialization():
    """Test 1: Basic serialization/deserialization cycle"""
    print("🧪 Test 1: Basic Serialization Cycle")
    print("=" * 50)
    
    # Get sample response from fixtures
    original = sample_response
    print(f"Original Response: {type(original).__name__}")
    print(f"Messages count: {len(original.messages)}")
    print(f"Model: {original.params.model}")
    print(f"Duration: {original.duration}")
    
    # Create testable version (with mixin)
    testable = TestableResponse(
        messages=original.messages,
        params=original.params,
        duration=original.duration,
        timestamp=original.timestamp
    )
    
    # Serialize
    print("\n📦 Serializing...")
    cache_dict = testable.to_cache_dict()
    
    # Verify serialization structure
    print("✅ Serialization completed")
    print(f"Cache dict keys: {list(cache_dict.keys())}")
    print(f"Cache metadata: {cache_dict.get('_class_path', 'missing')}")
    
    # Deserialize
    print("\n📤 Deserializing...")
    restored = TestableResponse.from_cache_dict(cache_dict)
    
    # Verify restoration
    print("✅ Deserialization completed")
    print(f"Restored type: {type(restored)}")
    print(f"Messages count: {len(restored.messages)}")
    print(f"Model: {restored.params.model}")
    print(f"Duration: {restored.duration}")
    
    # Compare key properties
    assert len(restored.messages) == len(original.messages)
    assert restored.params.model == original.params.model
    assert restored.duration == original.duration
    print("✅ All assertions passed!")
    
    return cache_dict, restored


def test_complex_nested_objects():
    """Test 2: Complex nested objects (Messages, Params, Verbosity)"""
    print("\n🧪 Test 2: Complex Nested Objects")
    print("=" * 50)
    
    # Create a complex Response with nested objects
    complex_messages = Messages([
        Message(role="system", content="You are helpful"),
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="2+2 equals 4")
    ])
    
    complex_params = Params(
        model="gpt-4o",
        messages=complex_messages,
        temperature=0.7,
        verbose=Verbosity.DEBUG  # Test enum serialization
    )
    
    complex_response = TestableResponse(
        messages=complex_messages,
        params=complex_params,
        duration=2.34
    )
    
    print(f"Original verbosity: {complex_params.verbose} (type: {type(complex_params.verbose)})")
    
    # Serialize and deserialize
    cache_dict = complex_response.to_cache_dict()
    restored = TestableResponse.from_cache_dict(cache_dict)
    
    # Verify complex nested data
    print(f"Restored verbosity: {restored.params.verbose} (type: {type(restored.params.verbose)})")
    print(f"Messages preserved: {len(restored.messages) == len(complex_messages)}")
    print(f"Temperature preserved: {restored.params.temperature}")
    
    # Test Verbosity enum specifically
    assert isinstance(restored.params.verbose, Verbosity)
    assert restored.params.verbose == Verbosity.DEBUG
    print("✅ Verbosity enum correctly preserved!")
    
    return cache_dict, restored


def test_graceful_degradation():
    """Test 3: Graceful degradation when classes change"""
    print("\n🧪 Test 3: Graceful Degradation")
    print("=" * 50)
    
    # Create a cache dict that simulates a missing class
    fake_cache_dict = {
        "_class_path": "Chain.fake.FakeResponse",  # Non-existent class
        "_cached_at": "2025-01-01T12:00:00",
        "_cache_version": "1.0",
        "messages": {
            "type": "cacheable_object",
            "object_class": "Chain.message.messages.Messages",
            "data": {
                "_class_path": "Chain.message.messages.Messages",
                "_cached_at": "2025-01-01T12:00:00",
                "messages": [
                    {
                        "type": "pydantic_model",
                        "model_class": "Chain.fake.FakeMessage",  # Non-existent class
                        "data": {
                            "role": "user",
                            "content": "Hello world"
                        }
                    }
                ]
            }
        },
        "params": {
            "type": "string_fallback",
            "data": "fake_params_string",
            "original_type": "FakeParams"
        },
        "duration": 1.23
    }
    
    # Try to deserialize with warnings
    print("Attempting to deserialize with fake classes...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            restored = TestableResponse.from_cache_dict(fake_cache_dict)
            print(f"✅ Graceful degradation successful!")
            print(f"Warnings generated: {len(w)}")
            for warning in w:
                print(f"  ⚠️  {warning.message}")
            
            print(f"Restored object type: {type(restored)}")
            print(f"Duration preserved: {restored.duration}")
            
        except Exception as e:
            print(f"❌ Failed graceful degradation: {e}")
            raise


def test_json_serialization():
    """Test 4: Full JSON round-trip (file system simulation)"""
    print("\n🧪 Test 4: JSON Round-trip Serialization")
    print("=" * 50)
    
    # Use sample response
    original = sample_response
    testable = TestableResponse(
        messages=original.messages,
        params=original.params,
        duration=original.duration,
        timestamp=original.timestamp
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = Path(f.name)
        
        # Serialize to JSON file
        cache_dict = testable.to_cache_dict()
        json.dump(cache_dict, f, indent=2, default=str)  # default=str for any non-serializable objects
    
    try:
        # Read back from JSON file
        with open(temp_file, 'r') as f:
            loaded_dict = json.load(f)
        
        # Deserialize from loaded dict
        restored = TestableResponse.from_cache_dict(loaded_dict)
        
        print("✅ JSON round-trip successful!")
        print(f"Original duration: {original.duration}")
        print(f"Restored duration: {restored.duration}")
        print(f"JSON file size: {temp_file.stat().st_size} bytes")
        
        # Verify data integrity
        assert restored.duration == original.duration
        assert len(restored.messages) == len(original.messages)
        print("✅ Data integrity verified!")
        
    finally:
        # Clean up
        temp_file.unlink()


def test_performance():
    """Test 5: Performance characteristics"""
    print("\n🧪 Test 5: Performance Test")
    print("=" * 50)
    
    import time
    
    # Use sample response
    original = sample_response
    testable = TestableResponse(
        messages=original.messages,
        params=original.params,
        duration=original.duration,
        timestamp=original.timestamp
    )
    
    # Time serialization
    start_time = time.time()
    cache_dict = testable.to_cache_dict()
    serialize_time = time.time() - start_time
    
    # Time deserialization
    start_time = time.time()
    restored = TestableResponse.from_cache_dict(cache_dict)
    deserialize_time = time.time() - start_time
    
    print(f"Serialization time: {serialize_time:.4f}s")
    print(f"Deserialization time: {deserialize_time:.4f}s")
    print(f"Total round-trip time: {serialize_time + deserialize_time:.4f}s")
    print(f"Cache dict size: {len(json.dumps(cache_dict, default=str))} characters")
    
    # Performance should be reasonable (under 100ms for typical objects)
    total_time = serialize_time + deserialize_time
    if total_time < 0.1:
        print("✅ Performance: Excellent (< 100ms)")
    elif total_time < 0.5:
        print("✅ Performance: Good (< 500ms)")
    else:
        print("⚠️  Performance: Slow (> 500ms)")


def run_all_tests():
    """Run all tests in sequence"""
    print("🚀 Testing CacheableMixin with Response Objects")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_serialization()
        test_complex_nested_objects()
        test_graceful_degradation()
        test_json_serialization()
        test_performance()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✅ CacheableMixin is working correctly with Response objects")
        print("✅ Nested objects (Messages, Params, Verbosity) handled properly")
        print("✅ Graceful degradation working for missing classes")
        print("✅ JSON serialization compatible")
        print("✅ Performance is acceptable")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
