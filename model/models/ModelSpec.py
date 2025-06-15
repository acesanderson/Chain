from pydantic import BaseModel, Field
from typing import Optional, Literal, Union
from datetime import date
from enum import Enum


# Provider-specific configuration classes
class OpenAIConfig(BaseModel):
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
    temperature_range: list[float] = [0.0, 2.0]
    top_p_range: list[float] = [0.0, 1.0]
    frequency_penalty_range: list[float] = [-2.0, 2.0]
    presence_penalty_range: list[float] = [-2.0, 2.0]
    supports_seed: bool = False
    supports_response_format: bool = False


class AnthropicConfig(BaseModel):
    max_tokens_required: bool = True
    max_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on max tokens"
    )
    system_message_separate: bool = True
    supports_streaming: bool = True
    temperature_range: list[float] = [0.0, 1.0]
    top_p_range: list[float] = [0.0, 1.0]
    top_k_range: Optional[list[int]] = None
    supports_stop_sequences: bool = True


class GoogleConfig(BaseModel):
    max_output_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on output tokens"
    )
    supports_system_instruction: bool = True
    supports_function_calling: bool = False
    supports_grounding: bool = False
    temperature_range: list[float] = [0.0, 2.0]
    top_p_range: list[float] = [0.0, 1.0]
    top_k_range: list[int] = [1, 40]
    candidate_count_max: int = 1
    supports_safety_settings: bool = True


class GroqConfig(BaseModel):
    max_tokens_limit: Optional[int] = Field(
        None, description="Hard limit on max tokens"
    )
    supports_streaming: bool = True
    temperature_range: list[float] = [0.0, 2.0]
    top_p_range: list[float] = [0.0, 1.0]
    supports_json_mode: bool = False
    ultra_fast_inference: bool = True


class OllamaConfig(BaseModel):
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
    temperature_range: list[float] = [0.0, 1.0]
    top_p_range: list[float] = [0.0, 1.0]
    top_k_range: list[int] = [1, 50]
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



class ModelSpec(BaseModel):
    # Core identifiers
    model_id: str = Field(
        ..., description="Official API/CLI identifier (e.g., 'gpt-4o', 'llama3.1:70b')"
    )
    # Provider information
    provider: Literal[
        "openai", "anthropic", "google", "groq", "ollama", "huggingface"
    ] = Field(..., description="Provider of the model")

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

    # Provider-specific configurations
    provider_config: ProviderConfig = Field(
        description="Provider-specific settings and constraints"
    )

    def __str__(self):
        return f"{self.model_id} ({self.provider})"

    def __repr__(self):
        return f"ModelSpec(model_id={self.model_id}, provider={self.provider})"

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
    def accepts_image_input(self) -> bool:
        """Check if model accepts image input"""
        return self.capabilities.image_input

    @property
    def accepts_audio_input(self) -> bool:
        """Check if model accepts audio input"""
        return self.capabilities.audio_input

    @property
    def accepts_video_input(self) -> bool:
        """Check if model accepts video input"""
        return self.capabilities.video_input

    @property
    def accepts_text_input(self) -> bool:
        """Check if model accepts text input"""
        return self.capabilities.text_input

    @property
    def supports_image_output(self) -> bool:
        """Check if model supports image output"""
        return self.capabilities.image_output

    @property
    def supports_audio_output(self) -> bool:
        """Check if model supports audio output (TTS)"""
        return self.capabilities.audio_output

    @property
    def supports_video_output(self) -> bool:
        """Check if model supports video output"""
        return self.capabilities.video_output

    @property
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling"""
        return self.capabilities.function_calling

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

    def card(self):
        from rich.console import Console
        from rich.tree import Tree

        def display_model(model: BaseModel) -> None:
            console = Console()
            
            tree = Tree(f"[bold blue]{model.__class__.__name__}[/bold blue]")
            
            for field_name, field_value in model.model_dump().items():
                _add_to_tree(tree, field_name, field_value)
            
            console.print(tree)

        def _add_to_tree(parent: Tree, key: str, value) -> None:
            if isinstance(value, dict):
                branch = parent.add(f"[green]{key}[/green]")
                for k, v in value.items():
                    _add_to_tree(branch, k, v)
            elif isinstance(value, list):
                branch = parent.add(f"[green]{key}[/green] [dim]({len(value)} items)[/dim]")
                for i, item in enumerate(value):
                    _add_to_tree(branch, f"[{i}]", item)
            else:
                parent.add(f"[green]{key}[/green]: [yellow]{repr(value)}[/yellow]")

        display_model(self)



class ModelSpecs(BaseModel):
    specs: list[ModelSpec] = Field(description="Collection of model specifications")
