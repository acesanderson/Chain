from pydantic import BaseModel, Field
from typing import Optional, Any, ClassVar, TYPE_CHECKING
from Chain.message.message import Message
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.parser.parser import Parser
from Chain.model.models.models import ModelStore


# Sub classes for specialized params
## Special instructor-native params
class RetryConfig(BaseModel):
    """
    Configuration for retry logic in the instructor framework.
    """

    max_retries: int = 3
    timeout: Optional[float] = None
    exponential_backoff: bool = True
    retry_on_validation_error: bool = True
    circuit_breaker_threshold: int = 5


class MockConfig(BaseModel):
    """
    Configuration for mocking responses in the instructor framework.
    """

    enabled: bool = False
    responses: dict[str, Any] = Field(default_factory=dict)


class ClientParams(BaseModel):
    """
    Parameters that are specific to a client.
    """

    pass


class OpenAIParams(ClientParams):
    """
    Parameters specific to OpenAI API spec-using clients.
    NOTE: This is a generic OpenAI client, not the official OpenAI API.
    """

    # Class vars
    provider: ClassVar[str] = "openai"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)

    # Excluded from serialization in API calls
    model: str = Field(..., description="The model identifier to use for inference.", exclude=True)

    # Core parameters
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    safety_settings: Optional[dict[str, Any]] = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class GeminiParams(OpenAIParams):
    """
    Parameters specific to Gemini clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class vars
    provider: ClassVar[str] = "gemini"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)


class OllamaParams(OpenAIParams):
    """
    Parameters specific to Ollama clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class vars (excluded by default)
    provider: ClassVar[str] = "ollama"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    # Core parameters
    num_ctx: Optional[int] = None  # Number of context tokens
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[list[str]] = None

    # Methods
    def model_post_init(self, __context) -> None:
        """
        Steps:
        - Call super method to initialize OpenAIParams
        - Get the number of context tokens for the Ollama model
        """
        super().model_post_init(__context)
        # Get the number of context tokens for the Ollama model
        if not self.num_ctx:
            # If num_ctx is not set, try to get it from the model store
            if ModelStore.is_supported(self.model):
                self.num_ctx = ModelStore.get_num_ctx(self.model)
            else:
                # If model is not supported, set num_ctx to None
                self.num_ctx = None
        self.num_ctx = self._get_num_ctx()

    def _get_num_ctx(self) -> Optional[int]:
        """
        Get the number of context tokens for the Ollama model.
        This is a placeholder method; actual implementation may vary.
        """
        from pathlib import Path
        import json
        dir_path = Path(__file__).parent
        ollama_context_sizes_file = dir_path.parent / "clients" / "ollama_context_sizes.json"
        if ollama_context_sizes_file.exists():
            with open(ollama_context_sizes_file, "r") as f:
                context_sizes = json.load(f)
            # Return the context size for the current model if it exists
            return context_sizes.get(self.model, None)


        # Ollama does not provide a direct way to get context size, so we return None
        return None


class AnthropicParams(ClientParams):
    """
    Parameters specific to Anthropic clients.
    """

    # Class vars
    provider: ClassVar[str] = "anthropic"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    max_tokens: Optional[int] = (
        None  # Anthropic uses max_tokens, not max_tokens_to_sample
    )
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[list[str]] = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class PerplexityParams(OpenAIParams):
    """
    Parameters specific to Perplexity clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class vars
    provider: ClassVar[str] = "perplexity"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)


# Union type for all client-specific parameters
ClientParamsTypes = (
    OpenAIParams | OllamaParams | AnthropicParams | GeminiParams | PerplexityParams
)


# Main Params class
class Params(BaseModel):
    """
    Parameters that are constructed by Model and are sent to Clients.
    """

    # Core parameters
    model: str = Field(..., description="The model identifier to use for inference.")
    messages: list[Message] = Field(
        default_factory=list,
        description="List of messages to send to the model. Can include text, images, audio, etc.",
    )

    # Optional parameters
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for sampling. If None, defaults to provider-specific value.",
    )
    stream: bool = False
    verbose: bool = True

    # Post model init parameters
    parser: Optional[Any] = Field(
        default=None,
        description="Parser to convert messages to a specific format. Not intended for direct use.",
    )
    query_input: str | Message | list[Message] | None = Field(
        default=None,
        description="Various possible input types that we coerce to a list of Messages. Not intended for direct use.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Provider of the model, populated post init. Not intended for direct use.",
    )

    # Client parameters (embedded in dict for now)
    client_params: Optional[ClientParams] = Field(
        default=None,
        description="Client-specific parameters. Can be OpenAIParams, OllamaParams, AnthropicParams, etc.",
    )

    # New features (available but not implemented)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    mock_config: MockConfig = Field(default_factory=MockConfig)

    def model_post_init(self, __context) -> None:
        """
        Steps:
        - coerce query_input to a list of Messages
        - validate model
        - set provider
        - validate temperature (provider-specific ranges)
        """
        # 1. Coerce query_input to a list of Messages
        self._coerce_query_input()
        # 2. Validate model
        if not ModelStore.is_supported(self.model):
            raise ValueError(f"Model '{self.model}' is not supported.")
        # 3. Set provider
        if self.provider is None:
            for provider in ModelStore.models().keys():
                if self.model in ModelStore.models()[provider]:
                    self.provider = provider
                    break
        if self.provider is None:
            raise ValueError(f"Provider not identified for model: {self.model}")
        # 4. Validate temperature
        self.validate_temperature()
        # 5. Validate Parser object in self.parser
        if self.parser is not None and not isinstance(self.parser, Parser):
            raise TypeError("parser must be an instance of Parser")
        # 6. Validate client_params TBD

    def validate_temperature(self):
        """
        Validate temperature against provider-specific ranges.
        """
        if self.temperature is None:
            return
        temperature_range: Optional[tuple[float, float]] = None
        for client_param_type in ClientParamsTypes.__args__:
            if self.provider == client_param_type.provider:
                temperature_range = client_param_type.temperature_range
        if not temperature_range:
            raise ValueError(
                f"Temperature range not found for provider: {self.provider}"
            )
        if temperature_range[0] <= self.temperature <= temperature_range[1]:
            return
        else:
            raise ValueError(
                f"Temperature {self.temperature} is out of range {temperature_range} for provider: {self.provider}"
            )

    def _coerce_query_input(self):
        """
        Coerce query_input to a list of Messages.
        """
        if self.query_input is None:
            return
        input_messages = []
        if isinstance(self.query_input, Message):
            input_messages.append(self.query_input)
        elif isinstance(self.query_input, str):
            message = Message(role="user", content=self.query_input)
            input_messages.append(message)
        elif isinstance(self.query_input, list):
            if not all(isinstance(item, Message) for item in self.query_input):
                raise ValueError(
                    "All items in query_input list must be of type Message."
                )
            input_messages = self.query_input
        else:
            raise ValueError(
                "query_input must be a Message, a string, or a list of Messages."
            )
        self.query_input = None
        if self.messages != []:
            input_messages = self.messages + input_messages
            self.messages = input_messages
        else:
            self.messages = input_messages

    def generate_cache_key(self) -> str:
        """
        Generate a reliable cache key for the Params instance.
        Only includes fields that would affect the LLM response.
        """
        from hashlib import sha256
        import json

        # Use sort_keys for deterministic JSON ordering
        messages_str = json.dumps(self.convert_messages(), sort_keys=True)

        # Include parser since it affects response format
        parser_str = self.parser.pydantic_model.__name__ if self.parser else "none"

        # Handle None temperature gracefully
        temp_str = str(self.temperature) if self.temperature is not None else "none"

        params_str = "|".join([messages_str, self.model, temp_str, parser_str])

        return sha256(params_str.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        """
        Generate a string representation of the Params instance.
        """
        return f"Params(model={self.model}, messages={self.messages}, temperature={self.temperature}, provider={self.provider})"

    def __repr__(self) -> str:
        """
        Generate a detailed string representation of the Params instance.
        """
        return (
            f"Params(model={self.model!r}, messages={self.messages!r}, "
            f"temperature={self.temperature!r}, provider={self.provider!r}, "
            f"client_params={self.client_params!r}, parser={self.parser!r})"
        )

    # Convert to dict -- clients use this to send to the API
    def convert_messages(self) -> list[dict]:
        """
        Convert messages to a list of dictionaries.
        This is used by clients to send messages to the API.
        """
        # If no messages at all, that's an error
        if not self.messages:
            raise ValueError("No messages to convert. Messages list cannot be empty.")

        converted_messages = []
        for message in self.messages:
            if isinstance(message, ImageMessage):
                match self.provider:
                    case "openai" | "ollama" | "gemini" | "perplexity":
                        converted_messages.append(message.to_openai().model_dump())
                    case "anthropic":
                        converted_messages.append(message.to_anthropic().model_dump())
            elif isinstance(message, AudioMessage):
                # Currently we only use GPT for this, specifically the gpt-4o-audio-preview model.
                if not self.model == "gpt-4o-audio-preview":
                    raise ValueError(
                        "AudioMessage can only be used with the gpt-4o-audio-preview model."
                    )
                converted_messages.append(message.to_openai().model_dump())
            elif isinstance(message, Message) and self.provider == "anthropic":
                # For Anthropic, we need to convert the message to the appropriate format.
                converted_messages.append(message.model_dump())
            else:
                # For other message types, we use the default model_dump method.
                if not isinstance(message, Message):
                    raise ValueError(f"Unsupported message type: {type(message)}")
                converted_messages.append(message.model_dump())

        return converted_messages

    def to_openai(self) -> dict:
        base_params = {
            "model": self.model,
            "messages": self.convert_messages(),
            "response_model": self.parser.pydantic_model if self.parser else None,
            "temperature": self.temperature,
            "stream": self.stream,
        }

        # Automatically include all client params
        if self.client_params:
            client_dict = self.client_params.model_dump(exclude_none=True)
            base_params.update(client_dict)

        # Filter out None values and return
        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }  # Actually filter None values EXCEPT for response_model, as Instructor expects it to be present

    def to_ollama(self) -> dict:
        return self.to_openai()

    def to_anthropic(self) -> dict:
        """
        Convert parameters to Anthropic format.
        Key differences from OpenAI:
        1. System messages become a separate 'system' parameter
        2. max_tokens is required
        3. No response_model in the API call params
        """
        # Start with converted messages
        converted_messages = self.convert_messages()

        # Extract system message if present
        system_content = ""
        filtered_messages = []

        for message in converted_messages:
            if message.get("role") == "system":
                system_content = message.get("content", "")
            else:
                filtered_messages.append(message)

        # Also check if any remaining messages have system role (Anthropic quirk)
        for message in filtered_messages:
            if message.get("role") == "system":
                # Convert system role to user role (as per your AnthropicClient logic)
                message["role"] = "user"

        # Build base parameters
        base_params = {
            "model": self.model,
            "messages": filtered_messages,
            "max_retries": 0,  # As per your client implementation
            "response_model": (
                self.parser.pydantic_model if self.parser else None
            ),  # Include response_model for instructor
            "temperature": (
                self.temperature if self.temperature is not None else 1.0
            ),  # Default to 1.0 if not set
        }

        # Add system parameter if we have system content
        if system_content:
            base_params["system"] = system_content

        # Set max_tokens based on model (as per your client logic)
        if self.model == "claude-3-5-sonnet-20240620":
            base_params["max_tokens"] = 8192
        else:
            base_params["max_tokens"] = 8192  # Default for other models

        # Add temperature if specified and validate range
        if self.temperature is not None:
            if not (0 <= self.temperature <= 1):
                raise ValueError(
                    "Temperature for Anthropic models needs to be between 0 and 1."
                )
            base_params["temperature"] = self.temperature

        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }

    def to_gemini(self) -> dict:
        return self.to_openai()

    def to_perplexity(self) -> dict:
        return self.to_openai()
