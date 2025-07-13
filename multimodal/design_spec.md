# Design Specification: Image Generation and TTS Extensions

## Overview

This specification extends the Chain framework to support image generation and text-to-speech (TTS) capabilities through the existing client architecture. The implementation maintains consistency with current patterns while adding multimodal generation support for OpenAI, Google, and Ollama providers.

## Core Design Principles

1. **Extend existing clients** rather than creating separate multimodal clients
2. **Leverage existing Request/Response flow** with new request types
3. **Maintain provider-agnostic interface** through Request class orchestration
4. **Auto-generate appropriate Message objects** (ImageMessage/AudioMessage)
5. **Preserve caching and serialization** capabilities

## Implementation Requirements

### 1. Request Class Extensions

#### New Fields
```python
class Request(BaseModel, ...):
    # NEW: Request type discrimination
    request_type: Literal["text", "image_gen", "audio_gen"] = "text"
    
    # NEW: Generation-specific content
    generation_prompt: Optional[str] = None  # For image generation
    tts_text: Optional[str] = None  # For TTS input
```

#### New Constructor Methods
```python
@classmethod
def for_image_generation(
    cls, 
    model: str, 
    prompt: str,
    client_params: Optional[dict] = None,
    **kwargs
) -> "Request":
    """Create Request for image generation"""

@classmethod  
def for_tts(
    cls,
    model: str,
    text: str,
    client_params: Optional[dict] = None,
    **kwargs
) -> "Request":
    """Create Request for TTS generation"""
```

#### Updated Format Methods
Extend existing `to_openai()`, `to_google()`, `to_ollama()` methods to handle generation request types with provider-specific parameter mapping.

### 2. ClientParams Extensions

#### OpenAIParams
```python
class OpenAIParams(ClientParams):
    # Existing text parameters...
    
    # Image generation parameters
    image_size: Optional[str] = None  # e.g., "1024x1024", "1792x1024"
    image_quality: Optional[str] = None  # "standard", "hd" 
    image_style: Optional[str] = None  # "vivid", "natural"
    image_n: Optional[int] = None  # Number of images
    
    # TTS parameters
    voice: Optional[str] = None  # e.g., "alloy", "echo", "fable"
    tts_model: Optional[str] = None  # "tts-1", "tts-1-hd"
    speed: Optional[float] = None  # 0.25 to 4.0
```

#### GoogleParams  
```python
class GoogleParams(OpenAIParams):
    # Inherits OpenAI params for compatibility
    
    # Google-specific TTS parameters
    voice_name: Optional[str] = None  # e.g., "Kore", "Aoede"
    # (Google image generation parameters TBD based on actual API)
```

#### OllamaParams
```python
class OllamaParams(OpenAIParams):
    # Inherits OpenAI params for compatibility
    # (Local model parameters TBD based on actual capabilities)
```

### 3. Client Method Extensions

#### New Methods for All Clients
```python
def generate_image(self, request: Request) -> tuple[ImageMessage, Usage]:
    """Generate image from text prompt"""

def generate_speech(self, request: Request) -> tuple[AudioMessage, Usage]:
    """Generate speech from text input"""
```

#### Implementation Pattern
- Validate `request.request_type` matches method
- Convert request to provider-specific format
- Call provider API
- Create appropriate Message object with generated content
- Return `(Message, Usage)` tuple

### 4. Model Class Integration

#### Primary Interface (Option B)
```python
# Through existing query method with Request objects
model = Model("dall-e-3")
request = Request.for_image_generation(
    model="dall-e-3", 
    prompt="a red sports car",
    client_params={"image_size": "1024x1024", "image_quality": "hd"}
)
response = model.query(request=request)
# response.content contains ImageMessage
```

#### Future Enhancement (TBD)
```python
# Convenience methods on Model class
model = Model("dall-e-3")
response = model.generate_image("a red car", size="1024x1024")
```

### 5. Provider-Specific Implementation Notes

#### OpenAI
- **Image Models**: DALL-E 3, DALL-E 2
- **TTS Models**: tts-1, tts-1-hd  
- **API Endpoints**: `/v1/images/generations`, `/v1/audio/speech`

#### Google  
- **TTS Models**: gemini-2.5-flash-preview-tts
- **Image Models**: (TBD - likely through Gemini models)
- **Integration**: Through existing Google client using OpenAI SDK compatibility

#### Ollama
- **Models**: (TBD - depends on pulled local models with generation capabilities)
- **Integration**: Through existing Ollama client using OpenAI SDK compatibility

### 6. Response Integration

The existing `Response` class accommodates generated content through its support for Message objects. Generated ImageMessage/AudioMessage objects will be returned in the response content, maintaining consistency with current text response handling.

### 7. Caching Integration

No changes required to caching system. Generated content will be cached through existing mechanisms:
- Request cache keys include generation parameters
- ImageMessage/AudioMessage serialization already implemented
- Cache validation works through existing hash comparison

## Implementation Order

1. **Extend ClientParams classes** with generation parameters
2. **Add Request class fields and constructors** for generation types  
3. **Update Request format methods** (to_openai, to_google, to_ollama)
4. **Implement client generation methods** starting with OpenAI
5. **Update Model.query()** to handle generation request types
6. **Test integration** with existing Response/caching systems
7. **Extend to Google and Ollama** clients

## Validation & Error Handling

- Validate generation capabilities against ModelSpec when available
- Graceful fallback when generation not supported by provider
- Maintain existing ChainError patterns for generation failures
- Provider-specific parameter validation through existing ClientParams system

This specification maintains architectural consistency while cleanly extending multimodal capabilities through the existing request/response flow.
