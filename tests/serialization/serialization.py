#!/usr/bin/env python3
"""
Comprehensive test script for Chain serialization/deserialization logic.
Tests Message, ImageMessage, AudioMessage, Messages, and Response classes.

Run this script to verify that caching serialization works correctly.
"""

import json
import tempfile
import base64
from pathlib import Path
from pydantic import BaseModel

# Import Chain classes
from Chain.message.message import Message
from Chain.message.messages import Messages
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.result.response import Response
from Chain.model.params.params import Params


# Test Pydantic models for structured content
class TestAnimal(BaseModel):
    name: str
    species: str
    age: int
    habitat: str

class TestFrog(BaseModel):
    species: str
    name: str
    legs: int
    color: str


def create_test_image_file() -> Path:
    """Create a small test PNG file for ImageMessage testing."""
    # Create a minimal 1x1 PNG file (base64 encoded)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_file.write(png_data)
    temp_file.close()
    return Path(temp_file.name)


def create_test_audio_file() -> Path:
    """Create a minimal test audio file for AudioMessage testing."""
    # Create a minimal WAV file header (empty audio)
    wav_data = (
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    )
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(wav_data)
    temp_file.close()
    return Path(temp_file.name)


def test_message_serialization():
    """Test Message class serialization/deserialization."""
    print("🧪 Testing Message Serialization...")
    
    # Test 1: String content
    msg1 = Message(role="user", content="Hello, world!")
    cache_dict1 = msg1.to_cache_dict()
    restored1 = Message.from_cache_dict(cache_dict1)
    
    assert msg1.role == restored1.role
    assert msg1.content == restored1.content
    print("✅ String content: PASSED")
    
    # Test 2: Pydantic model content
    animal = TestAnimal(name="Fluffy", species="Cat", age=3, habitat="House")
    msg2 = Message(role="assistant", content=animal)
    cache_dict2 = msg2.to_cache_dict()
    restored2 = Message.from_cache_dict(cache_dict2)
    
    assert msg2.role == restored2.role
    assert isinstance(restored2.content, TestAnimal)
    assert restored2.content.name == "Fluffy"
    assert restored2.content.species == "Cat"
    print("✅ Pydantic model content: PASSED")
    
    # Test 3: List of Pydantic models
    frog1 = TestFrog(species="Tree Frog", name="Kermit", legs=4, color="Green")
    frog2 = TestFrog(species="Poison Dart", name="Poison", legs=4, color="Blue")
    msg3 = Message(role="assistant", content=[frog1, frog2])
    cache_dict3 = msg3.to_cache_dict()
    restored3 = Message.from_cache_dict(cache_dict3)
    
    assert msg3.role == restored3.role
    assert isinstance(restored3.content, list)
    assert len(restored3.content) == 2
    assert isinstance(restored3.content[0], TestFrog)
    assert restored3.content[0].name == "Kermit"
    print("✅ List of Pydantic models: PASSED")
    
    # Test 4: Graceful degradation (simulate class not found)
    # Manually create a cache dict with a non-existent class
    bad_cache_dict = {
        "message_type": "Message",
        "role": "assistant",
        "content": {
            "type": "pydantic_model",
            "model_class": "non.existent.FakeModel",
            "data": {"fake_field": "fake_value"}
        }
    }
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        restored_bad = Message.from_cache_dict(bad_cache_dict)
        assert len(w) == 1
        assert "Cache deserialization failed" in str(w[0].message)
        assert isinstance(restored_bad.content, str)  # Should be JSON string
        assert "fake_field" in restored_bad.content
    print("✅ Graceful degradation: PASSED")


def test_imagemessage_serialization():
    """Test ImageMessage class serialization/deserialization."""
    print("\n🖼️ Testing ImageMessage Serialization...")
    
    # Create test image file
    image_file = create_test_image_file()
    
    try:
        # Test with file path
        img_msg = ImageMessage(
            role="user",
            text_content="What's in this image?",
            file_path=image_file
        )
        
        cache_dict = img_msg.to_cache_dict()
        restored = ImageMessage.from_cache_dict(cache_dict)
        
        assert img_msg.role == restored.role
        assert img_msg.text_content == restored.text_content
        assert img_msg.image_content == restored.image_content
        assert img_msg.mime_type == restored.mime_type
        assert restored.mime_type == "image/png"  # Should be converted to PNG
        print("✅ ImageMessage with file: PASSED")
        
    finally:
        # Clean up test file
        image_file.unlink()


def test_audiomessage_serialization():
    """Test AudioMessage class serialization/deserialization."""
    print("\n🔊 Testing AudioMessage Serialization...")
    
    # Create test audio file
    audio_file = create_test_audio_file()
    
    try:
        # Test with file path
        audio_msg = AudioMessage(
            role="user",
            text_content="Transcribe this audio",
            audio_file=audio_file
        )
        
        cache_dict = audio_msg.to_cache_dict()
        restored = AudioMessage.from_cache_dict(cache_dict)
        
        assert audio_msg.role == restored.role
        assert audio_msg.text_content == restored.text_content
        assert audio_msg.audio_content == restored.audio_content
        assert audio_msg.format == restored.format
        assert restored.format == "wav"
        print("✅ AudioMessage with file: PASSED")
        
    finally:
        # Clean up test file
        audio_file.unlink()


def test_messages_serialization():
    """Test Messages class serialization/deserialization."""
    print("\n📝 Testing Messages Serialization...")
    
    # Create test image file for mixed message types
    image_file = create_test_image_file()
    audio_file = create_test_audio_file()
    
    try:
        # Create Messages with mixed message types
        messages = Messages()
        
        # Add different types of messages
        messages.append(Message(role="system", content="You are a helpful assistant"))  # system
        messages.add_new("user", "Hello!")  # user #1
        
        frog = TestFrog(species="Tree Frog", name="Kermit", legs=4, color="Green")
        messages.append(Message(role="assistant", content=frog))  # assistant
        
        # Add ImageMessage
        img_msg = ImageMessage(
            role="user",  # user #2
            text_content="What's in this image?",
            file_path=image_file
        )
        messages.append(img_msg)
        
        # Add AudioMessage
        audio_msg = AudioMessage(
            role="user",  # user #3
            text_content="Transcribe this",
            audio_file=audio_file
        )
        messages.append(audio_msg)
        
        # Test serialization/deserialization
        cache_dict = messages.to_cache_dict()
        restored = Messages.from_cache_dict(cache_dict)
        
        # Verify structure
        assert len(restored) == len(messages)
        assert isinstance(restored, Messages)
        
        # Check individual messages
        assert restored[0].role == "system"
        assert restored[1].content == "Hello!"
        assert isinstance(restored[2].content, TestFrog)
        assert restored[2].content.name == "Kermit"
        assert isinstance(restored[3], ImageMessage)
        assert isinstance(restored[4], AudioMessage)
        
        # Test Messages methods
        user_msgs = restored.user_messages()
        assert len(user_msgs) == 3  # "Hello!", ImageMessage, AudioMessage
        
        # Verify the user messages are the right types
        assert user_msgs[0].content == "Hello!"  # Regular Message
        assert isinstance(user_msgs[1], ImageMessage)  # ImageMessage
        assert isinstance(user_msgs[2], AudioMessage)  # AudioMessage
        
        last_msg = restored.last()
        assert isinstance(last_msg, AudioMessage)
        
        print("✅ Mixed Messages types: PASSED")
        
    finally:
        # Clean up test files
        image_file.unlink()
        audio_file.unlink()

def test_response_serialization():
    """Test Response class serialization/deserialization with debugging."""
    print("\n📊 Testing Response Serialization...")
    
    # Create test messages
    messages = Messages()
    messages.add_new("user", "What is 2+2?")
    
    animal = TestAnimal(name="Calculator", species="Robot", age=1, habitat="Computer")
    messages.append(Message(role="assistant", content=animal))
    
    print(f"Original messages type: {type(messages)}")
    print(f"Is Messages instance: {isinstance(messages, Messages)}")
    
    # Create test params (simplified)
    params = Params(
        model="gpt-4o",
        query_input="What is 2+2?",
        temperature=0.7
    )
    
    # Create Response
    response = Response(
        messages=messages,
        params=params,
        duration=1.234
    )
    
    print(f"Response messages type after creation: {type(response.messages)}")
    
    # Test serialization
    cache_dict = response.to_cache_dict()
    print(f"Serialized messages structure: {type(cache_dict.get('messages'))}")
    
    # Test deserialization
    restored = Response.from_cache_dict(cache_dict)
    
    # Debug output
    print(f"Restored messages type: {type(restored.messages)}")
    print(f"Restored is Messages: {isinstance(restored.messages, Messages)}")
    
    # Verify Response attributes
    assert len(restored.messages) == len(response.messages)
    assert restored.duration == response.duration
    assert restored.params.model == response.params.model
    assert restored.params.temperature == response.params.temperature
    
    # Verify messages were properly restored
    # Use a more flexible check for now
    if not isinstance(restored.messages, Messages):
        print(f"WARNING: Expected Messages, got {type(restored.messages)}")
        # Convert if needed
        if isinstance(restored.messages, Messages):
            restored.messages = Messages(restored.messages)
    
    assert isinstance(restored.messages, Messages), f"Expected Messages, got {type(restored.messages)}"
    assert restored.messages[0].content == "What is 2+2?"
    assert isinstance(restored.messages[1].content, TestAnimal)
    assert restored.messages[1].content.name == "Calculator"
    
    # Test Response properties
    assert restored.model == "gpt-4o"
    assert isinstance(restored.content, TestAnimal)
    
    print("✅ Response with nested objects: PASSED")


def test_response_serialization():
    """Test Response class serialization/deserialization."""
    print("\n📊 Testing Response Serialization...")
    
    # Create test messages
    messages = Messages()
    messages.add_new("user", "What is 2+2?")
    
    animal = TestAnimal(name="Calculator", species="Robot", age=1, habitat="Computer")
    messages.append(Message(role="assistant", content=animal))
    
    # Create test params (simplified)
    params = Params(
        model="gpt-4o",
        query_input="What is 2+2?",
        temperature=0.7
    )
    
    # Create Response
    response = Response(
        messages=messages,
        params=params,
        duration=1.234
    )
    
    # Test serialization/deserialization
    cache_dict = response.to_cache_dict()
    restored = Response.from_cache_dict(cache_dict)
    
    # Verify Response attributes
    assert len(restored.messages) == len(response.messages)
    assert restored.duration == response.duration
    assert restored.params.model == response.params.model
    assert restored.params.temperature == response.params.temperature
    
    # Verify messages were properly restored
    assert isinstance(restored.messages, Messages)
    assert restored.messages[0].content == "What is 2+2?"
    assert isinstance(restored.messages[1].content, TestAnimal)
    assert restored.messages[1].content.name == "Calculator"
    
    # Test Response properties
    assert restored.model == "gpt-4o"
    assert isinstance(restored.content, TestAnimal)
    
    print("✅ Response with nested objects: PASSED")

def run_all_tests():
    """Run all serialization tests."""
    print("🚀 Starting Chain Serialization Tests\n")
    
    try:
        test_message_serialization()
        test_imagemessage_serialization()
        test_audiomessage_serialization()
        test_messages_serialization()
        test_response_serialization()
        
        print("\n🎉 All tests PASSED!")
        print("✅ Serialization/deserialization is working correctly")
        print("✅ Cache system is ready to replace cloudpickle")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
