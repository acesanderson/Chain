#!/usr/bin/env python3
"""
Simple serialization test for Chain classes.
Tests to_cache_dict() and from_cache_dict() methods.
"""

import tempfile
from pathlib import Path
from datetime import datetime

# Import all the classes to test
from Chain.message.message import Message
from Chain.message.audiomessage import AudioMessage
from Chain.message.imagemessage import ImageMessage
from Chain.message.messages import Messages
from Chain.model.params.params import Params
from Chain.result.response import Response
from Chain.result.error import ChainError, ErrorInfo, ErrorDetail


def test_message():
    print("Testing Message...")
    
    # Create original
    original = Message(role="user", content="Hello world")
    
    # Serialize
    cache_dict = original.to_cache_dict()
    
    # Deserialize
    restored = Message.from_cache_dict(cache_dict)
    
    # Check
    assert restored.role == original.role
    assert restored.content == original.content
    print("✅ Message passed")


def test_audiomessage():
    print("Testing AudioMessage...")
    
    # Create a dummy audio file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio data")
        audio_file = Path(f.name)
    
    try:
        # Create original
        original = AudioMessage(
            role="user",
            text_content="Transcribe this",
            audio_file=audio_file
        )
        
        # Serialize
        cache_dict = original.to_cache_dict()
        
        # Deserialize
        restored = AudioMessage.from_cache_dict(cache_dict)
        
        # Check
        assert restored.role == original.role
        assert restored.text_content == original.text_content
        assert restored.format == original.format
        print("✅ AudioMessage passed")
        
    finally:
        audio_file.unlink()


def test_imagemessage():
    print("Testing ImageMessage...")
    
    # Create a minimal PNG file
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        image_file = Path(f.name)
    
    try:
        # Create original
        original = ImageMessage(
            role="user",
            text_content="What's in this image?",
            image_file=image_file
        )
        
        # Serialize
        cache_dict = original.to_cache_dict()
        
        # Deserialize
        restored = ImageMessage.from_cache_dict(cache_dict)
        
        # Check
        assert restored.role == original.role
        assert restored.text_content == original.text_content
        assert restored.mime_type == original.mime_type
        print("✅ ImageMessage passed")
        
    finally:
        image_file.unlink()


def test_messages():
    print("Testing Messages...")
    
    # Create original
    msg1 = Message(role="user", content="Hello")
    msg2 = Message(role="assistant", content="Hi there")
    original = Messages([msg1, msg2])
    
    # Serialize
    cache_dict = original.to_cache_dict()
    
    # Deserialize
    restored = Messages.from_cache_dict(cache_dict)
    
    # Check
    assert len(restored) == len(original)
    assert restored[0].role == original[0].role
    assert restored[0].content == original[0].content
    assert restored[1].role == original[1].role
    assert restored[1].content == original[1].content
    print("✅ Messages passed")


def test_params():
    print("Testing Params...")
    
    # Create original
    messages = Messages([Message(role="user", content="Test")])
    original = Params(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )
    
    # Serialize
    cache_dict = original.to_cache_dict()
    
    # Deserialize
    restored = Params.from_cache_dict(cache_dict)
    
    # Check
    assert restored.model == original.model
    assert restored.temperature == original.temperature
    assert len(restored.messages) == len(original.messages)
    print("✅ Params passed")


def test_response():
    print("Testing Response...")
    
    # Create original
    messages = Messages([
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4")
    ])
    params = Params(model="gpt-4o", messages=messages)
    original = Response(
        messages=messages,
        params=params,
        duration=1.23
    )
    
    # Serialize
    cache_dict = original.to_cache_dict()
    
    # Deserialize
    restored = Response.from_cache_dict(cache_dict)
    
    # Check
    assert restored.duration == original.duration
    assert restored.params.model == original.params.model
    assert len(restored.messages) == len(original.messages)
    print("✅ Response passed")


def test_chainerror():
    print("Testing ChainError...")
    
    # Create original
    error_info = ErrorInfo(
        code="test_error",
        message="Test error message",
        category="test",
        timestamp=datetime.now()
    )
    error_detail = ErrorDetail(
        exception_type="ValueError",
        stack_trace="test stack trace",
        raw_response=None,
        request_params={"test": "params"},
        retry_count=0
    )
    original = ChainError(info=error_info, detail=error_detail)
    
    # Serialize
    cache_dict = original.to_cache_dict()
    
    # Deserialize
    restored = ChainError.from_cache_dict(cache_dict)
    
    # Check
    assert restored.info.code == original.info.code
    assert restored.info.message == original.info.message
    assert restored.detail.exception_type == original.detail.exception_type
    print("✅ ChainError passed")


def main():
    print("🧪 Simple Serialization Tests")
    print("=" * 40)
    
    try:
        test_message()
        test_audiomessage()
        test_imagemessage()
        test_messages()
        test_params()
        test_response()
        test_chainerror()
        
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("All classes can serialize and deserialize correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
