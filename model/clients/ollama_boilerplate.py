"""
Ollama Python Library Boilerplate Code
Comprehensive examples for common use cases
"""

import asyncio
import base64
from pathlib import Path
from ollama import chat, ChatResponse, Client, AsyncClient


# =============================================================================
# 1. STRAIGHTFORWARD QUERY (with optional parameters)
# =============================================================================

def simple_query():
    """Basic chat query with optional parameters like num_ctx."""
    
    response: ChatResponse = chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': 'Explain quantum computing in simple terms.',
            }
        ],
        options={
            'num_ctx': 4096,        # Context window size
            'temperature': 0.7,     # Creativity (0.0-1.0)
            'top_p': 0.9,          # Nucleus sampling
            'top_k': 40,           # Top-k sampling
            'repeat_penalty': 1.1,  # Repetition penalty
            'num_predict': 500,     # Max tokens to generate
        }
    )
    
    print("Response:", response['message']['content'])
    # Alternative access:
    print("Response (direct):", response.message.content)
    
    return response


# =============================================================================
# 2. STREAMING QUERY
# =============================================================================

def streaming_query():
    """Query that returns a Stream object for real-time responses."""
    
    print("Streaming response:")
    stream = chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': 'Write a short story about a robot learning to paint.',
            }
        ],
        stream=True,
        options={
            'num_ctx': 4096,
            'temperature': 0.8,
        }
    )
    
    full_response = ""
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        full_response += content
    
    print("\n\nFull response collected:", full_response)
    return full_response


# =============================================================================
# 3. AUDIO MESSAGE QUERY (Note: Ollama doesn't directly support audio)
# =============================================================================

def audio_analysis_query():
    """
    Audio analysis query. Note: Ollama doesn't natively support audio.
    You would need to:
    1. Use a separate speech-to-text service (like Whisper)
    2. Send the transcribed text to Ollama for analysis
    """
    
    # Simulated transcribed audio content
    transcribed_audio = "Hello, this is a test audio message. Can you analyze the sentiment of this speech?"
    
    response: ChatResponse = chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'You are an expert audio content analyzer. Analyze the following transcribed audio for sentiment, tone, and key topics.'
            },
            {
                'role': 'user',
                'content': f'Analyze this transcribed audio: "{transcribed_audio}"'
            }
        ],
        options={
            'temperature': 0.3,  # Lower temperature for analysis
        }
    )
    
    print("Audio Analysis:", response.message.content)
    return response


# Alternative: If you have audio file, convert to text first
def audio_with_whisper_example():
    """
    Example of how you might integrate with Whisper for audio transcription
    before sending to Ollama for analysis.
    """
    
    # Pseudo-code for audio transcription (requires separate Whisper setup)
    # import whisper
    # model = whisper.load_model("base")
    # result = model.transcribe("audio_file.mp3")
    # transcribed_text = result["text"]
    
    transcribed_text = "[Simulated transcription] The weather today is beautiful and sunny."
    
    response = chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': f'Analyze the sentiment and extract key information from this audio transcription: {transcribed_text}'
            }
        ]
    )
    
    return response


# =============================================================================
# 4. IMAGE MESSAGE QUERY (Vision/Multimodal)
# =============================================================================

def image_analysis_query():
    """Image analysis using multimodal models."""
    
    # Method 1: Using image file path
    image_path = "path/to/your/image.jpg"  # Replace with actual path
    
    try:
        response: ChatResponse = chat(
            model='llava',  # or 'llama3.2-vision' or other vision model
            messages=[
                {
                    'role': 'user',
                    'content': 'What do you see in this image? Describe it in detail.',
                    'images': [image_path]
                }
            ],
            options={
                'temperature': 0.1,  # Lower for more factual descriptions
            }
        )
        
        print("Image Analysis:", response.message.content)
        return response
        
    except Exception as e:
        print(f"Error with image file: {e}")
        return image_analysis_with_base64()


def image_analysis_with_base64():
    """Image analysis using base64 encoded image."""
    
    # Method 2: Using base64 encoded image
    try:
        # Read and encode image
        image_path = Path("path/to/your/image.jpg")  # Replace with actual path
        if image_path.exists():
            image_data = image_path.read_bytes()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        else:
            # Simulated base64 image data (this won't work, just for structure)
            image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        response: ChatResponse = chat(
            model='llava',
            messages=[
                {
                    'role': 'user',
                    'content': 'Analyze this image and tell me what objects you can identify.',
                    'images': [image_base64]
                }
            ]
        )
        
        print("Image Analysis (Base64):", response.message.content)
        return response
        
    except Exception as e:
        print(f"Error with base64 image: {e}")
        return None


def image_analysis_with_file_object():
    """Image analysis using file object."""
    
    try:
        with open('path/to/your/image.jpg', 'rb') as image_file:  # Replace with actual path
            response: ChatResponse = chat(
                model='llava',
                messages=[
                    {
                        'role': 'user',
                        'content': 'What is interesting or unusual about this image?',
                        'images': [image_file.read()]
                    }
                ]
            )
        
        print("Image Analysis (File Object):", response.message.content)
        return response
        
    except Exception as e:
        print(f"Error with file object: {e}")
        return None


# =============================================================================
# 5. ASYNCHRONOUS QUERY
# =============================================================================

async def async_query():
    """Asynchronous chat query."""
    
    client = AsyncClient()
    
    response = await client.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': 'What are the benefits of asynchronous programming?'
            }
        ],
        options={
            'num_ctx': 4096,
            'temperature': 0.6,
        }
    )
    
    print("Async Response:", response.message.content)
    return response


async def async_streaming_query():
    """Asynchronous streaming query."""
    
    client = AsyncClient()
    
    print("Async streaming response:")
    full_response = ""
    
    async for chunk in await client.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': 'Explain the concept of machine learning in detail.'
            }
        ],
        stream=True,
        options={
            'temperature': 0.7,
        }
    ):
        content = chunk['message']['content']
        print(content, end='', flush=True)
        full_response += content
    
    print("\n\nAsync full response collected:", full_response)
    return full_response


async def async_multiple_queries():
    """Multiple asynchronous queries running concurrently."""
    
    client = AsyncClient()
    
    # Define multiple queries
    queries = [
        "What is artificial intelligence?",
        "Explain quantum computing.",
        "What are the applications of blockchain?",
    ]
    
    # Create tasks for concurrent execution
    tasks = []
    for i, query in enumerate(queries):
        task = client.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': query}],
            options={'temperature': 0.5}
        )
        tasks.append(task)
    
    # Wait for all queries to complete
    responses = await asyncio.gather(*tasks)
    
    # Print results
    for i, response in enumerate(responses):
        print(f"\nQuery {i+1}: {queries[i]}")
        print(f"Response: {response.message.content[:100]}...")
    
    return responses


# =============================================================================
# CUSTOM CLIENT CONFIGURATION
# =============================================================================

def custom_client_example():
    """Using custom client with specific configuration."""
    
    client = Client(
        host='http://localhost:11434',  # Default Ollama host
        headers={
            'x-custom-header': 'custom-value',
            'user-agent': 'my-app/1.0'
        },
        timeout=30.0,  # Request timeout
    )
    
    response = client.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'user',
                'content': 'Hello from custom client!'
            }
        ]
    )
    
    print("Custom Client Response:", response.message.content)
    return response


# =============================================================================
# MAIN EXECUTION EXAMPLES
# =============================================================================

def main():
    """Run synchronous examples."""
    
    print("=== 1. Simple Query ===")
    simple_query()
    
    print("\n=== 2. Streaming Query ===")
    streaming_query()
    
    print("\n=== 3. Audio Analysis ===")
    audio_analysis_query()
    
    print("\n=== 4. Image Analysis ===")
    # Note: These will fail without actual image files
    # image_analysis_query()
    
    print("\n=== 5. Custom Client ===")
    custom_client_example()


async def async_main():
    """Run asynchronous examples."""
    
    print("\n=== Async Examples ===")
    
    print("\n--- Single Async Query ---")
    await async_query()
    
    print("\n--- Async Streaming ---")
    await async_streaming_query()
    
    print("\n--- Multiple Concurrent Queries ---")
    await async_multiple_queries()


if __name__ == "__main__":
    # Run synchronous examples
    main()
    
    # Run asynchronous examples
    print("\n" + "="*50)
    asyncio.run(async_main())
