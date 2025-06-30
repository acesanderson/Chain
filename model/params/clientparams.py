from pydantic import BaseModel, Field
from typing import Optional, Any, ClassVar, Literal

Provider = Literal["openai", "ollama", "anthropic", "google", "perplexity"]

class ClientParams(BaseModel):
    """
    Parameters that are specific to a client.
    """
    provider: Provider


class OpenAIParams(ClientParams):
    """
    Parameters specific to OpenAI API spec-using clients.
    NOTE: This is a generic OpenAI client, not the official OpenAI API.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)

    # Core parameters
    provider: Provider = "openai"
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    safety_settings: Optional[dict[str, Any]] = None


class GoogleParams(OpenAIParams):
    """
    Parameters specific to Gemini clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    # Core parameters
    provider: Provider = "google"

class OllamaParams(OpenAIParams):
    """
    Parameters specific to Ollama clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    # Core parameters -- note for Instructor we will need to embed these in "extra_body":{"options": {}}
    provider: Provider = "ollama"
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[list[str]] = None

    # SORT THIS OUT
    # def model_dump(self, *args, **kwargs):
    #     """
    #     Override model_dump to embed dicts as needed for our Instructor/OpenAI Spec/Ollama compatibility.
    #     Why is this needed? Because Ollama expects "options" to be nested under "extra_body" in the API call.
    #     """
    #     # Call the parent method to get the base dict
    #     base_dict = super().model_dump(*args, **kwargs)
    #
    #     # Ollama expects options to be nested under "extra_body"
    #     extra_body = {"options": base_dict}
    #     return {"extra_body": extra_body}
    #

class AnthropicParams(ClientParams):
    """
    Parameters specific to Anthropic clients.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    # Core parameters
    provider: Provider = "anthropic"
    max_tokens: int = (
        4000  # TBD: three defaults: 256, 1024, 4000, depending on use case
    )
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None

    # Excluded from serialization in API calls
    model: str = Field(
        default="",
        description="The model identifier to use for inference.",
        exclude=True,
    )


class PerplexityParams(OpenAIParams):
    """
    Parameters specific to Perplexity clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class var
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)

    # Core parameters
    provider: Provider = "perplexity"


# All client parameters types, for validation
ClientParamsModels = [OpenAIParams, OllamaParams, AnthropicParams, GoogleParams, PerplexityParams]


