from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Any, ClassVar, override
from Chain.message.message import Message
from Chain.message.textmessage import TextMessage
from Chain.message.messages import Messages
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.progress.verbosity import Verbosity
from Chain.progress.display_mixins import (
    RichDisplayParamsMixin,
    PlainDisplayParamsMixin,
)
from Chain.parser.parser import Parser
from Chain.model.models.models import ModelStore
import json


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

    provider: ClassVar[str]


class OpenAIParams(ClientParams):
    """
    Parameters specific to OpenAI API spec-using clients.
    NOTE: This is a generic OpenAI client, not the official OpenAI API.
    """

    # Class vars
    provider: ClassVar[str] = "openai"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)

    # Excluded from serialization in API calls
    model: str = Field(
        default="",
        description="The model identifier to use for inference.",
        exclude=True,
    )

    # Core parameters
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

    # Class vars
    provider: ClassVar[str] = "google"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)


class OllamaParams(OpenAIParams):
    """
    Parameters specific to Ollama clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """

    # Class vars (excluded by default)
    provider: ClassVar[str] = "ollama"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    # Core parameters -- note for Instructor we will need to embed these in "extra_body":{"options": {}}
    num_ctx: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[list[str]] = None

    @override
    def model_dump(self, *args, **kwargs):
        """
        Override model_dump to embed dicts as needed for our Instructor/OpenAI Spec/Ollama compatibility.
        Why is this needed? Because Ollama expects "options" to be nested under "extra_body" in the API call.
        """
        # Call the parent method to get the base dict
        base_dict = super().model_dump(*args, **kwargs)

        # Ollama expects options to be nested under "extra_body"
        extra_body = {"options": base_dict}
        return {"extra_body": extra_body}

    @override
    def model_dump_json(self, *args, **kwargs):
        """
        Override model_dump_json to ensure OllamaParams is serialized correctly.
        """
        # Use the custom model_dump method to get the dict
        base_dict = self.model_dump(*args, **kwargs)
        return json.dumps(base_dict)


class AnthropicParams(ClientParams):
    """
    Parameters specific to Anthropic clients.
    """

    # Class vars
    provider: ClassVar[str] = "anthropic"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    # Core parameters
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

    # Class vars
    provider: ClassVar[str] = "perplexity"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)


# Union type for all client-specific parameters
ClientParamsTypes = (
    OpenAIParams | OllamaParams | AnthropicParams | GoogleParams | PerplexityParams
)


# Main Params class
class Params(BaseModel, RichDisplayParamsMixin, PlainDisplayParamsMixin):
    """
    Parameters that are constructed by Model and are sent to Clients.
    Note: we mixin our DisplayParamsMixin classes to provide rich and plain display methods.
    """

    # Core parameters
    model: str = Field(..., description="The model identifier to use for inference.")
    messages: Messages | list[Message] = Field(
        default_factory=list,
        description="List of messages to send to the model. Can include text, images, audio, etc.",
    )

    # Optional parameters
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for sampling. If None, defaults to provider-specific value.",
    )
    stream: bool = False
    verbose: Verbosity = Verbosity.PROGRESS

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
        - validate that we have EITHER messages or query_input
        - coerce query_input to a list of Messages
        - validate model
        - set provider
        - validate temperature (provider-specific ranges)
        - validate Parser object in self.parser
        - validate ClientParams against provider
        - set client_params based on provider if not already set
        """
        super().model_post_init(__context)  # Call the mixin's post_init first!
        # 1. Validate that we have EITHER messages or query_input
        if not self.messages and not self.query_input:
            raise ValidationError(
                "Either messages or query_input must be provided. Both cannot be empty."
            )
        # 2. Coerce query_input to a list of Messages
        self._coerce_query_input()
        # 3. Validate model
        if not ModelStore.is_supported(self.model):
            raise ValueError(f"Model '{self.model}' is not supported.")
        # 4. Set provider
        if self.provider is None:
            for provider in ModelStore.models().keys():
                if self.model in ModelStore.models()[provider]:
                    self.provider = provider
                    break
        if self.provider is None:
            raise ValueError(f"Provider not identified for model: {self.model}")
        # 5. Validate temperature
        self.validate_temperature()
        # 6. Validate Parser object in self.parser
        if self.parser is not None and not isinstance(self.parser, Parser):
            raise TypeError("parser must be an instance of Parser")
        # 7. Validate ClientParams against provider.
        if self.client_params:
            if self.client_params.provider != self.provider:
                raise ValidationError(
                    f"ClientParams provider '{self.client_params.provider}' does not match Params provider '{self.provider}'."
                )
        # 8. Set client_params based on provider if not already set
        if self.client_params is None:
            for client_param_type in ClientParamsTypes.__args__:
                # Use getattr with a default to safely check for 'provider' on ClassVar
                if self.provider == getattr(client_param_type, "provider", None):
                    self.client_params = client_param_type()
                    break
            if self.client_params is None:
                raise ValidationError(
                    f"ClientParams could not be determined for model provider: {self.provider}. This may indicate an unsupported provider or a missing client_params initialization logic."
                )

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
            message = TextMessage(role="user", content=self.query_input)
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

    # Serialization methods for different providers
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
            elif isinstance(message, TextMessage) and self.provider == "anthropic":
                # For Anthropic, we need to convert the message to the appropriate format.
                converted_messages.append(message.model_dump())
            else:
                # For other message types, we use the default model_dump method.
                if not isinstance(message, Message):
                    raise ValueError(f"Unsupported message type: {type(message)}")
                converted_messages.append(message.model_dump())

        return converted_messages

    # Each serialization method first operates on client_params, then constructs base parameters.
    def _to_openai_spec(self) -> dict:
        assert isinstance(
            self.client_params, OpenAIParams
        ), f"OpenAIParams (and subclasses) expected for OpenAI client, not {type(self.client_params)}. This is a bug in the code."
        """
        We use OpenAI spec with OpenAI, Gemini, Ollama, and Perplexity clients.
        """
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

    def to_openai(self) -> dict:
        assert (
            type(self.client_params) is OpenAIParams
        ), f"OpenAIParams expected for OpenAI client, not {type(self.client_params)}."
        return self._to_openai_spec()

    def to_ollama(self) -> dict:
        """
        We should set num_ctx so we have maximal context window.
        Recall that Ollama with Instructor/OpenAI spec expects options to be nested under extra_body, so we have overriden model_dump.
        """
        assert isinstance(
            self.client_params, OllamaParams
        ), f"OllamaParams expected for Ollama client, not {type(self.client_params)}."
        # Set num_ctx to the maximum context window for the model.
        num_ctx = ModelStore.get_num_ctx(self.model)
        if num_ctx is None:
            raise ValueError(
                f"Model '{self.model}' does not have a defined context window."
            )
        self.client_params.num_ctx = num_ctx
        return self._to_openai_spec()

    def to_anthropic(self) -> dict:
        """
        Convert parameters to Anthropic format.
        Key differences from OpenAI:
        1. System messages become a separate 'system' parameter
        2. max_tokens is required
        3. No response_model in the API call params
        """
        assert isinstance(
            self.client_params, AnthropicParams
        ), f"AnthropicParams expected for Anthropic client, not {type(self.client_params)}."
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

        # Add temperature if specified and validate range
        if self.temperature is not None:
            if not (0 <= self.temperature <= 1):
                raise ValueError(
                    "Temperature for Anthropic models needs to be between 0 and 1."
                )
            base_params["temperature"] = self.temperature

        # Add client_params to base_params
        if self.client_params:
            client_dict = self.client_params.model_dump(exclude_none=True)
            base_params.update(client_dict)

        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }

    def to_google(self) -> dict:
        assert isinstance(
            self.client_params, GoogleParams
        ), f"GeminiParams expected for Gemini client, not {type(self.client_params)}."
        return self._to_openai_spec()

    def to_perplexity(self) -> dict:
        assert isinstance(
            self.client_params, PerplexityParams
        ), f"PerplexityParams expected for Perplexity client, not {type(self.client_params)}."
        return self._to_openai_spec()
