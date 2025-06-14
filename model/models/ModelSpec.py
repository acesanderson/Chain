from pydantic import BaseModel, Field
from typing import Optional, Literal, Union
from datetime import date
from enum import Enum


# Provider-specific endpoint enums
class OpenAIEndpoint(str, Enum):
    CHAT_COMPLETIONS = "chat/completions"
    COMPLETIONS = "completions"
    EMBEDDINGS = "embeddings"
    FINE_TUNING = "fine_tuning/jobs"
    IMAGES_GENERATIONS = "images/generations"
    IMAGES_EDITS = "images/edits"
    IMAGES_VARIATIONS = "images/variations"
    AUDIO_SPEECH = "audio/speech"
    AUDIO_TRANSCRIPTIONS = "audio/transcriptions"
    AUDIO_TRANSLATIONS = "audio/translations"
    MODERATIONS = "moderations"


class AnthropicEndpoint(str, Enum):
    MESSAGES = "messages"
    COMPLETIONS = "complete"


class GoogleEndpoint(str, Enum):
    GENERATE_CONTENT = "generateContent"
    STREAM_GENERATE_CONTENT = "streamGenerateContent"
    COUNT_TOKENS = "countTokens"
    EMBED_CONTENT = "embedContent"
    BATCH_EMBED_CONTENTS = "batchEmbedContents"


class GroqEndpoint(str, Enum):
    CHAT_COMPLETIONS = "chat/completions"
    EMBEDDINGS = "embeddings"


class HuggingFaceEndpoint(str, Enum):
    INFERENCE = "inference"
    FEATURE_EXTRACTION = "feature-extraction"
    TEXT_GENERATION = "text-generation"
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_TEXT = "image-to-text"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"


class OllamaEndpoint(str, Enum):
    GENERATE = "generate"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    CREATE = "create"
    PULL = "pull"
    PUSH = "push"
    LIST = "list"
    SHOW = "show"


# Provider-specific configuration classes
class OpenAIConfig(BaseModel):
    endpoints: list[OpenAIEndpoint] = Field(default_factory=list)
    max_tokens_default: Optional[int] = Field(
        None, description="Default max tokens for generation"
    )
    max_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on max tokens"
    )
    supports_system_messages: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.0, 1.0)
    frequency_penalty_range: tuple[float, float] = (-2.0, 2.0)
    presence_penalty_range: tuple[float, float] = (-2.0, 2.0)
    supports_seed: bool = False
    supports_response_format: bool = False


class AnthropicConfig(BaseModel):
    endpoints: list[AnthropicEndpoint] = Field(default_factory=list)
    max_tokens_required: bool = True
    max_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on max tokens"
    )
    system_message_separate: bool = True
    supports_streaming: bool = True
    temperature_range: tuple[float, float] = (0.0, 1.0)
    top_p_range: tuple[float, float] = (0.0, 1.0)
    top_k_range: Optional[tuple[int, int]] = None
    supports_stop_sequences: bool = True


class GoogleConfig(BaseModel):
    endpoints: list[GoogleEndpoint] = Field(default_factory=list)
    max_output_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on output tokens"
    )
    supports_system_instruction: bool = True
    supports_function_calling: bool = False
    supports_grounding: bool = False
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.0, 1.0)
    top_k_range: tuple[int, int] = (1, 40)
    candidate_count_max: int = 1
    supports_safety_settings: bool = True


class GroqConfig(BaseModel):
    endpoints: list[GroqEndpoint] = Field(default_factory=list)
    max_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on max tokens"
    )
    supports_streaming: bool = True
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.0, 1.0)
    supports_json_mode: bool = False
    ultra_fast_inference: bool = True


class OllamaConfig(BaseModel):
    endpoints: list[OllamaEndpoint] = Field(default_factory=list)
    # Ollama-specific parameters
    num_ctx: Optional[int] = Field(
        None, description="Context window size (number of tokens)"
    )
    num_batch: Optional[int] = Field(
        None, description="Batch size for prompt processing"
    )
    num_gqa: Optional[int] = Field(None, description="Number of GQA groups")
    num_gpu: Optional[int] = Field(None, description="Number of GPU layers to use")
    main_gpu: Optional[int] = Field(
        None, description="Which GPU to use for main processing"
    )
    low_vram: Optional[bool] = Field(None, description="Enable low VRAM mode")
    f16_kv: Optional[bool] = Field(
        None, description="Use 16-bit floats for key/value cache"
    )
    logits_all: Optional[bool] = Field(None, description="Return logits for all tokens")
    vocab_only: Optional[bool] = Field(None, description="Only load vocabulary")
    use_mmap: Optional[bool] = Field(None, description="Use memory mapping")
    use_mlock: Optional[bool] = Field(None, description="Lock memory")
    embedding_only: Optional[bool] = Field(None, description="Only use for embeddings")
    rope_frequency_base: Optional[float] = Field(
        None, description="RoPE frequency base"
    )
    rope_frequency_scale: Optional[float] = Field(
        None, description="RoPE frequency scale"
    )
    num_thread: Optional[int] = Field(None, description="Number of threads to use")
    # Runtime parameters
    temperature: Optional[float] = Field(None, description="Default temperature")
    top_k: Optional[int] = Field(None, description="Top-k sampling")
    top_p: Optional[float] = Field(None, description="Top-p sampling")
    repeat_last_n: Optional[int] = Field(
        None, description="How many tokens to consider for repetition"
    )
    repeat_penalty: Optional[float] = Field(None, description="Repetition penalty")
    seed: Optional[int] = Field(None, description="Random seed")


class HuggingFaceConfig(BaseModel):
    endpoints: list[HuggingFaceEndpoint] = Field(default_factory=list)
    model_type: Optional[str] = Field(
        None, description="HuggingFace model type/architecture"
    )
    pipeline_tag: Optional[str] = Field(None, description="Primary pipeline/task")
    requires_auth_token: bool = False
    supports_inference_api: bool = True
    supports_hosted_inference: bool = False
    max_new_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on new tokens"
    )
    temperature_range: tuple[float, float] = (0.0, 1.0)
    top_p_range: tuple[float, float] = (0.0, 1.0)
    top_k_range: tuple[int, int] = (1, 50)
    supports_streaming: bool = False


# Union type for all provider configs
ProviderConfig = Union[
    OpenAIConfig,
    AnthropicConfig,
    GoogleConfig,
    GroqConfig,
    OllamaConfig,
    HuggingFaceConfig,
]


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"
    BMP = "bmp"
    HEIC = "heic"
    HEIF = "heif"
    AVIF = "avif"
    TIFF = "tiff"
    SVG = "svg"


class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    AAC = "aac"
    OGG = "ogg"
    WEBM = "webm"
    M4A = "m4a"
    FLAC = "flac"
    AIFF = "aiff"
    WMA = "wma"
    AMR = "amr"


class VideoFormat(str, Enum):
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"
    MKV = "mkv"
    FLV = "flv"


class ModelCapabilities(BaseModel):
    # Input modalities
    text_input: bool
    image_input: bool
    audio_input: bool
    video_input: bool

    # Output modalities
    text_output: bool
    image_output: bool
    audio_output: bool  # TTS
    video_output: bool

    # Specialized capabilities
    function_calling: bool
    reasoning: bool  # Chain-of-thought, o1-style reasoning
    code_generation: bool
    structured_output: bool  # JSON, Pydantic models
    streaming: bool

    # Advanced features
    multimodal_reasoning: bool  # Can reason across modalities
    real_time: bool  # Real-time audio/video processing
    fine_tuning: bool
    embeddings: bool


class ModelFormats(BaseModel):
    image_input: list[ImageFormat] = Field(default_factory=list)
    audio_input: list[AudioFormat] = Field(default_factory=list)
    video_input: list[VideoFormat] = Field(default_factory=list)
    image_output: list[ImageFormat] = Field(default_factory=list)
    audio_output: list[AudioFormat] = Field(default_factory=list)
    video_output: list[VideoFormat] = Field(default_factory=list)


class ModelUseCases(BaseModel):
    # Primary use cases - helps with model selection
    chat: bool = False
    completion: bool = False
    analysis: bool = False
    creative_writing: bool = False
    code_assistance: bool = False
    research: bool = False
    summarization: bool = False
    translation: bool = False
    reasoning_tasks: bool = False


class ModelSpec(BaseModel):
    # Core identifiers
    model_id: str = Field(
        ..., description="Official API/CLI identifier (e.g., 'gpt-4o', 'llama3.1:70b')"
    )
    display_name: str = Field(
        ..., description="Human-readable name (e.g., 'GPT-4o', 'Llama 3.1 70B')"
    )
    family: str = Field(
        ..., description="Model family (e.g., 'gpt-4', 'llama', 'claude')"
    )

    # Provider information
    provider: Literal[
        "openai", "anthropic", "google", "groq", "ollama", "huggingface"
    ] = Field(..., description="Provider of the model")
    deployment: Literal["local", "api", "hybrid"] = Field(
        ..., description="Deployment type of the model"
    )

    # Model characteristics
    parameter_count: Optional[str] = Field(
        None, description="Parameter count (e.g., '7b', '70b', '405b')"
    )
    context_window: int = Field(..., description="Maximum context window in tokens")
    knowledge_cutoff: Optional[date] = Field(
        default=None,
        description="Date when the model's knowledge was last updated; None if unknown or continuously updated",
    )

    # Capability specifications
    capabilities: ModelCapabilities = Field(description="What the model can do")
    formats: ModelFormats = Field(description="Supported input/output formats")
    use_cases: ModelUseCases = Field(description="Recommended use cases")

    # Provider-specific configurations
    provider_config: ProviderConfig = Field(
        description="Provider-specific settings and constraints"
    )

    # API endpoints this model supports (derived from provider_config)
    @property
    def endpoints(self) -> list[str]:
        """Get list of supported endpoints from provider config"""
        if hasattr(self.provider_config, "endpoints"):
            return [endpoint.value for endpoint in self.provider_config.endpoints]
        return []

    # Additional metadata
    description: str = Field(default="", description="Brief description of the model")
    documentation_url: Optional[str] = Field(
        None, description="Link to official documentation"
    )
    license: Optional[str] = Field(
        None, description="Model license (e.g., 'Apache 2.0', 'Custom')"
    )

    def __str__(self):
        return f"{self.display_name} ({self.provider})"

    def __repr__(self):
        return f"ModelSpec(model_id={self.model_id}, provider={self.provider}, deployment={self.deployment})"

    @property
    def is_multimodal(self) -> bool:
        """Check if model supports multiple input/output modalities"""
        caps = self.capabilities
        input_modalities = sum(
            [caps.text_input, caps.image_input, caps.audio_input, caps.video_input]
        )
        output_modalities = sum(
            [caps.text_output, caps.image_output, caps.audio_output, caps.video_output]
        )
        return input_modalities > 1 or output_modalities > 1

    @property
    def is_reasoning_model(self) -> bool:
        """Check if this is a reasoning-focused model"""
        return self.capabilities.reasoning

    @property
    def supports_structured_output(self) -> bool:
        """Check if model supports structured outputs"""
        return self.capabilities.structured_output or self.capabilities.function_calling

    def supports_format(self, format_type: str, file_format: str) -> bool:
        """Check if model supports a specific file format"""
        formats = getattr(self.formats, format_type, [])
        return file_format.lower() in [f.value.lower() for f in formats]


class ModelSpecs(BaseModel):
    specs: list[ModelSpec] = Field(description="Collection of model specifications")
