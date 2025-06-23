---
# Version 3.0
## Major New Features
### 1. Progress Display System (Entirely New)
- Rich Console Integration: Full progress reporting with spinners, colors, and real-time updates
- Plain Text Fallback: Simple timestamp-based progress for environments without Rich
- Console Hierarchy: Instance → Chain class → None console resolution
- Async Progress Support: Progress tracking for concurrent operations
- Event-Based Architecture: Standardized progress events with timestamps, duration tracking

### 2. Multimodal Support (Major Expansion)

#### ImageMessage Class: Full support for image analysis across all providers
- Base64 image processing with automatic PNG conversion and resizing
- OpenAI and Anthropic format conversion
- Terminal image display with chafa integration
- Support for JPEG, PNG, GIF, WEBP, etc.

#### AudioMessage Class: Audio processing capabilities
- MP3/WAV audio file handling with base64 encoding
- OpenAI audio model support (gpt-4o-audio-preview)
- Gemini multimodal audio support
- Audio playback functionality

#### Image Conversion Pipeline: Automatic image optimization for LLM consumption

### 3. Enhanced Model Support

#### New Providers:
- Perplexity client with citation support
- HuggingFace client for local models
- DeepSeek integration

#### Improved Existing Providers:
- Gemini: Full audio/image support via OpenAI SDK
- Ollama: Better context size management, streaming support
- All providers: Temperature validation, better error handling

### 4. Advanced Caching System
- **ChainCache**: Persistent caching with a database backend
- Cache Integration: Automatic cache lookup/storage in all query operations
- Pydantic Object Caching: Proper serialization/deserialization of structured outputs

### 5. Comprehensive Async Architecture
- AsyncChain: Full async workflow support with semaphore control
- ModelAsync: Dedicated async model class
- Concurrent Processing: Batch processing with progress tracking
- async/await: Proper async implementation across all clients

### 6. Chain Server & Client Architecture
- FastAPI Server: Complete REST API for Chain operations
- ServerClient: Client for distributed Chain operations
- Distributed Computing: Load balancing across multiple Chain servers
- Remote Model Access: Access powerful models on remote GPU servers

### 7. Enhanced CLI Framework
- ChainCLI: Extensible command-line interface framework
- Dynamic Argument Registration: Decorator-based CLI argument definition
- Rich Integration: Beautiful CLI output with markdown rendering
- Persistent History: Message history with file-based persistence

### 8. Developer Experience Improvements
- @llm Decorator: Function decorator for easy Chain integration with XML parameter wrapping
- Better Error Handling: Comprehensive error messages and validation
- Rich Documentation: Extensive docstrings and type hints
- Testing Framework: Comprehensive test suite with fixtures

### 9. Params Class (Major Refactor)
- Centralized Parameter Management: Consolidates all model and client-specific parameters into a single `Params` Pydantic class.
- Unified API: Simplifies `Model.query` and `ModelAsync.query_async` signatures by accepting a `Params` object.
- Enhanced Validation: Provides robust validation for parameters like `temperature` based on provider-specific ranges.
- Cache Key Generation: Implements a deterministic method for generating cache keys from `Params` objects, ensuring reliable caching.

---
## TBD
### 1. TBD: Workflow Composition System
- ChainML: JSON-based DSL for defining LLM workflows as DAGs
- Mermaid Integration: Automatic diagram generation from workflows
- Conditional Logic: Step-level and prompt-level conditional execution
- Variable Flow: Jinja2 templating for data passing between steps

### 2. Configuration & Model Management
- Model Aliases: User-friendly model name mapping
- Dynamic Model Discovery: Automatic Ollama model detection and updates
- Validation System: Model and provider validation
- Context Size Management: Per-model context window configuration
