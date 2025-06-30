# tests/serialization/test_cacheable_serialization.py

import base64
import pytest
import warnings
from pydantic import BaseModel

# Import classes that should now be Cacheable
from Chain.result.response import Response
from Chain.model.params.params import Params
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.message.messages import Messages
from Chain.progress.verbosity import Verbosity # Assuming Verbosity also gets CacheableMixin if it needs custom serialization beyond simple enum. Your old code had `to_cache_dict` on it.
from Chain.tests.fixtures.sample_models import ( # ADD THIS LINE
    MyTestPydanticContent,
    TestAnimal,
    TestFrog
)

# Import sample objects from fixtures
from Chain.tests.fixtures.sample_objects import (
    sample_message,
    sample_audio_message,
    sample_image_message,
    sample_messages,
    sample_params,
    sample_response,
    sample_image_file, # Needed for ImageMessage recreation if file_path is involved
    sample_audio_file, # Needed for AudioMessage recreation if audio_file is involved
)

# Helper to create temporary files for ImageMessage/AudioMessage tests
@pytest.fixture(scope="module")
def temp_image_file(tmp_path_factory):
    # Create a minimal 1x1 PNG file (base64 encoded)
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    )
    file_path = tmp_path_factory.mktemp("data") / "test_image.png"
    file_path.write_bytes(png_data)
    return file_path

@pytest.fixture(scope="module")
def temp_audio_file(tmp_path_factory):
    # Create a minimal WAV file header (empty audio)
    wav_data = (
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x10\x00data\x00\x00\x00\x00'
    )
    file_path = tmp_path_factory.mktemp("data") / "test_audio.wav"
    file_path.write_bytes(wav_data)
    return file_path


def run_serialization_test(obj, obj_name: str, expected_class: type):
    print(f"\n🧪 Testing {obj_name} Serialization Cycle")
    print(f"==================================================")
    print(f"Original {obj_name}: {obj}")
    print(f"Original Type: {type(obj)}")

    try:
        # 1. Serialize
        print("\n📦 Serializing...")
        serialized_data = obj.to_cache_dict()
        print("✅ Serialization completed.")
        print(f"Serialized data (keys): {list(serialized_data.keys())}")
        print(f"Stored class path: {serialized_data.get('_class_path')}")
        # print(f"Full serialized data (first 500 chars): {json.dumps(serialized_data, indent=2)[:500]}...") # Uncomment for detailed debug

        # 2. Deserialize
        print("\n📤 Deserializing...")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            restored_obj = expected_class.from_cache_dict(serialized_data)
            for warning_message in w:
                print(f"⚠️ Warning during deserialization: {warning_message.message}")

        # 3. Assertions
        print("✅ Deserialization completed.")
        print(f"Restored Type: {type(restored_obj)}")
        print(f"Restored {obj_name}: {restored_obj}")

        # Basic type assertion
        assert isinstance(restored_obj, expected_class)
        print(f"Type check: PASSED (Restored object is an instance of {expected_class.__name__})")

        # More specific assertions based on object type
        if isinstance(obj, Message):
            assert restored_obj.role == obj.role
            # For content, deep comparison might be needed for BaseModel/list
            if isinstance(obj.content, BaseModel):
                assert isinstance(restored_obj.content, BaseModel)
                assert restored_obj.content.model_dump() == obj.content.model_dump()
            elif isinstance(obj.content, list):
                assert isinstance(restored_obj.content, list)
                assert len(restored_obj.content) == len(obj.content)
                for i in range(len(obj.content)):
                    if isinstance(obj.content[i], BaseModel):
                        assert isinstance(restored_obj.content[i], BaseModel)
                        assert restored_obj.content[i].model_dump() == obj.content[i].model_dump()
                    else:
                        assert restored_obj.content[i] == obj.content[i]
            else: # string content
                assert restored_obj.content == obj.content

            if isinstance(obj, ImageMessage):
                assert restored_obj.text_content == obj.text_content
                assert restored_obj.image_content == obj.image_content # Raw base64 comparison
                assert restored_obj.mime_type == obj.mime_type
                # Note: file_path might not be identical if it was converted to base64 during init
                # but image_content should be
            if isinstance(obj, AudioMessage):
                assert restored_obj.text_content == obj.text_content
                assert restored_obj.audio_content == obj.audio_content # Raw base64 comparison
                assert restored_obj.format == obj.format
                # Note: audio_file might not be identical if it was converted to base64 during init
        elif isinstance(obj, Messages):
            assert len(restored_obj) == len(obj)
            for i in range(len(obj)):
                # Recursively call test for each message
                print(f"  -> Validating nested Message {i+1} in Messages...")
                # To avoid re-serializing, we'll just do direct content checks
                assert restored_obj[i].role == obj[i].role
                if isinstance(obj[i].content, BaseModel):
                    assert isinstance(restored_obj[i].content, BaseModel)
                    assert restored_obj[i].content.model_dump() == obj[i].content.model_dump()
                elif isinstance(obj[i].content, list):
                    assert isinstance(restored_obj[i].content, list)
                    assert len(restored_obj[i].content) == len(obj[i].content)
                    # This could go deeper for lists of models if needed
                    for j in range(len(obj[i].content)):
                        if isinstance(obj[i].content[j], BaseModel):
                            assert isinstance(restored_obj[i].content[j], BaseModel)
                            assert restored_obj[i].content[j].model_dump() == obj[i].content[j].model_dump()
                        else:
                            assert restored_obj[i].content[j] == obj[i].content[j]
                else:
                    assert restored_obj[i].content == obj[i].content
                print(f"  -> Nested Message {i+1} PASSED.")

        elif isinstance(obj, Params):
            assert restored_obj.model == obj.model
            assert restored_obj.temperature == obj.temperature
            assert restored_obj.stream == obj.stream
            assert restored_obj.provider == obj.provider
            # Check nested client_params (if it's not None)
            if obj.client_params:
                assert isinstance(restored_obj.client_params, type(obj.client_params))
                assert restored_obj.client_params.model_dump() == obj.client_params.model_dump()
            else:
                assert restored_obj.client_params is None # Or potentially a default one if model_post_init filled it

            # Check parser
            if obj.parser:
                assert restored_obj.parser is not None
                assert restored_obj.parser.pydantic_model.__name__ == obj.parser.pydantic_model.__name__
            else:
                assert restored_obj.parser is None

            # Check messages field in Params if it's there
            if obj.messages:
                assert len(restored_obj.messages) == len(obj.messages)
                assert isinstance(restored_obj.messages, Messages) # Should be converted to Messages type
                # You might need a deeper comparison for Messages content as well if they contain Pydantic models.
                # For now, a basic check that it's a Messages object is sufficient.

        elif isinstance(obj, Response):
            assert restored_obj.duration == obj.duration
            assert restored_obj.timestamp == obj.timestamp

            # Check nested messages
            assert isinstance(restored_obj.messages, Messages)
            assert len(restored_obj.messages) == len(obj.messages)
            # You might want to do a deep check on messages content here too.
            # Example: assert restored_obj.messages[0].content == obj.messages[0].content

            # Check nested params
            assert isinstance(restored_obj.params, Params)
            assert restored_obj.params.model == obj.params.model
            assert restored_obj.params.temperature == obj.params.temperature


        print(f"🥳 {obj_name} Serialization Test: PASSED!\n")
        return True

    except Exception as e:
        print(f"❌ {obj_name} Serialization Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Starting Chain Cacheable Serialization Tests")
    print("============================================================")

    results = []

    # Test Message
    results.append(run_serialization_test(sample_message, "Message (string content)", Message))
    # Add a Message with Pydantic content to fixtures if you don't have one
    class MyTestPydanticContent(BaseModel):
        value: str
        number: int
    msg_with_pydantic_content = Message(role="assistant", content=MyTestPydanticContent(value="test", number=123))
    results.append(run_serialization_test(msg_with_pydantic_content, "Message (Pydantic content)", Message))
    msg_with_list_pydantic_content = Message(role="assistant", content=[MyTestPydanticContent(value="a", number=1), MyTestPydanticContent(value="b", number=2)])
    results.append(run_serialization_test(msg_with_list_pydantic_content, "Message (List[Pydantic] content)", Message))


    # Test ImageMessage (requires file to be present or mocked base64)
    # Ensure sample_image_message has its internal base64 and mime_type populated
    # Or reconstruct it using a temporary file fixture
    # For now, let's assume sample_image_message is fully formed for direct serialization.
    # If sample_image_message relies on a file_path, ensure that file exists at test runtime.
    # Or, modify sample_image_message to directly contain base64 for testing purposes,
    # as the cacheable mixin won't re-read files.
    # If sample_image_message is initialized from file, ensure the file is present.
    # Otherwise, you can manually create an ImageMessage with pre-encoded base64 for testing.
    # Example:
    image_base64_content = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    test_image_message = ImageMessage(
        role="user",
        text_content="Describe this tiny image",
        image_content=image_base64_content,
        mime_type="image/png"
    )
    results.append(run_serialization_test(test_image_message, "ImageMessage", ImageMessage))


    # Test AudioMessage (similar considerations as ImageMessage for file_path)
    audio_base64_content = base64.b64encode(
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x10\x00data\x00\x00\x00\x00'
    ).decode("utf-8")
    test_audio_message = AudioMessage(
        role="user",
        text_content="Transcribe this silent audio",
        audio_content=audio_base64_content,
        format="wav"
    )
    results.append(run_serialization_test(test_audio_message, "AudioMessage", AudioMessage))


    # Test Messages (contains nested Message/ImageMessage/AudioMessage)
    # Ensure sample_messages contains fully formed Message, ImageMessage, AudioMessage instances
    # with their content (including base64) properly set, not relying on file paths post-init.
    # Create a fresh sample_messages with pre-encoded data for robust testing.
    test_messages = Messages([
        Message(role="system", content="You are a helpful bot."),
        Message(role="user", content="Tell me a joke."),
        Message(role="assistant", content="Why did the scarecrow win an award? Because he was outstanding in his field!"),
        test_image_message, # Re-use the pre-encoded image message
        test_audio_message  # Re-use the pre-encoded audio message
    ])
    results.append(run_serialization_test(test_messages, "Messages", Messages))

    # Test Params
    results.append(run_serialization_test(sample_params, "Params", Params))

    # Test Response
    # Create a response that uses the fully formed messages and params
    response_for_test = Response(
        messages=test_messages, # Use the robust test_messages
        params=sample_params,   # Use the sample_params (assuming it's solid)
        duration=1.234,
        # timestamp is set in model_post_init, so don't pass it here unless for specific override test
    )
    results.append(run_serialization_test(response_for_test, "Response", Response))


    print("\n" + "="*60)
    if all(results):
        print("🎉 ALL CACHEABLE SERIALIZATION TESTS PASSED!")
        print("✅ This means your CacheableMixin and class implementations are working for these types.")
        return 0
    else:
        print("❌ SOME CACHEABLE SERIALIZATION TESTS FAILED.")
        print("Review the output for detailed error messages.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
