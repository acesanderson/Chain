classDiagram
    class ModelSpecs {
        +List~ModelSpec~ specs
        +find_by_capability()
        +find_by_use_case()
        +find_multimodal()
    }

    class ModelSpec {
        +str model_id
        +str display_name
        +str family
        +str provider
        +str deployment
        +str description
        +ModelCapabilities capabilities
        +ModelFormats formats
        +ModelUseCases use_cases
        +ProviderConfig provider_config
        +endpoints : List~str~
        +is_multimodal() bool
        +is_reasoning_model() bool
        +supports_structured_output() bool
        +supports_format()
    }

    class ModelCapabilities {
        +bool text_input
        +bool image_input
        +bool audio_input
        +bool video_input
        +bool text_output
        +bool image_output
        +bool audio_output
        +bool video_output
        +bool function_calling
        +bool reasoning
        +bool code_generation
        +bool structured_output
        +bool streaming
        +bool multimodal_reasoning
        +bool real_time
        +bool fine_tuning
        +bool embeddings
    }

    class ModelFormats {
        +List~ImageFormat~ image_input
        +List~AudioFormat~ audio_input
        +List~VideoFormat~ video_input
        +List~ImageFormat~ image_output
        +List~AudioFormat~ audio_output
        +List~VideoFormat~ video_output
    }

    class ModelUseCases {
        +bool chat
        +bool completion
        +bool analysis
        +bool creative_writing
        +bool code_assistance
        +bool research
        +bool summarization
        +bool translation
        +bool reasoning_tasks
    }

    class ProviderConfig

    class OpenAIConfig
    class AnthropicConfig
    class GoogleConfig
    class GroqConfig
    class OllamaConfig
    class HuggingFaceConfig

    class ImageFormat
    class AudioFormat
    class VideoFormat

    ProviderConfig <|-- OpenAIConfig
    ProviderConfig <|-- AnthropicConfig
    ProviderConfig <|-- GoogleConfig
    ProviderConfig <|-- GroqConfig
    ProviderConfig <|-- OllamaConfig
    ProviderConfig <|-- HuggingFaceConfig

    ModelSpec --> ModelCapabilities
    ModelSpec --> ModelFormats
    ModelSpec --> ModelUseCases
    ModelSpec --> ProviderConfig
    ModelSpecs --> ModelSpec
    ModelFormats --> ImageFormat
    ModelFormats --> AudioFormat
    ModelFormats --> VideoFormat
