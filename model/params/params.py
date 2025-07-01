from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from Chain.message.message import Message
from Chain.message.textmessage import TextMessage
from Chain.message.imagemessage import ImageMessage
from Chain.message.audiomessage import AudioMessage
from Chain.message.messages import Messages
from Chain.parser.parser import Parser
from Chain.progress.verbosity import Verbosity
from Chain.progress.display_mixins import (
    RichDisplayParamsMixin,
    PlainDisplayParamsMixin,
)
from Chain.model.params.clientparams import Provider, ClientParamsModels, OpenAIParams, OllamaParams, AnthropicParams, GoogleParams, PerplexityParams
from Chain.parser.parser import Parser
from Chain.model.models.models import ModelStore

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
    verbose: Verbosity = Field(default = Verbosity.PROGRESS, exclude=True, description="Verbosity level for logging and progress display.")
    response_model: type[BaseModel] | dict | None = Field(
        default=None,
        description="Pydantic model to convert messages to a specific format. If dict, this is a schema for the model for serialization / caching purposes."
    )
    # Post model init parameters
    provider: Optional[Provider] = Field(
        default=None,
        description="Provider of the model, populated post init. Not intended for direct use.",
    )
    # Client parameters (embedded in dict for now)
    client_params: Optional[dict] = Field(
        default=None,
        description="Client-specific parameters. Validated against the provider-approved params, through our ClientParams.",
    )

    def model_post_init(self, __context) -> None:
        """
        Post-initialization method to validate and set parameters.
        Validates:
        - Model is supported by at least one provider
        - Temperature is within range for the provider
        - Client parameters match the provider
        Sets:
        - Provider based on the model
        - Client parameters based on the provider
        Finally, validates that we have our required parameters.
        """
        self._validate_model() # Raise error if model is not supported
        self._set_provider() # Set provider based on model
        self._validate_temperature() # Raise error if temperature is out of range for provider
        self._set_client_params() # Set client_params based on provider
        # Validate the whole lot
        if self.messages is None or len(self.messages) == 0:
            raise ValueError("Messages cannot be empty. Likely a code error.")
        if self.provider == "" or self.provider is None:
            raise ValueError("Provider must be set based on the model. Likely a code error.")

    # Constructor methods
    @classmethod
    def from_query_input(cls, query_input: str, **kwargs) -> "Params":
        """
        If you have a prompt string, input it this way. Intercepts query string and coerces it into Messages.
        """
        messages = kwargs.pop("messages", [])
        user_message = TextMessage(role = "user", content=query_input)
        modified_messages = messages + [user_message]
        kwargs.update({"messages": modified_messages})
        return cls(
            **kwargs,
        )

    # Validation methods
    def _set_provider(self) -> str:
        for provider in ModelStore.models().keys():
            if self.model in ModelStore.models()[provider]:
                self.provider = provider
                return
        if not self.provider:
            raise ValueError(f"Model '{self.model}' is not supported by any provider.")

    def _set_client_params(self):
        """
        User should add extra client params as a dict called client_params.
        """
        # If client_params is already set, we assume it's a dict and validate it.
        if self.client_params is None:
            return
        # If client_params is a dict, we need to validate it against the provider's client params.
        for client_param_type in ClientParamsModels:
            if self.provider == client_param_type.provider:
                # Validate the dict against the provider's client params
                try:
                    client_param_type.model_validate(self.client_params)
                except ValidationError as e:
                    raise ValidationError(
                        f"Client parameters do not match the expected format for provider '{self.provider}': {e}"
                    )

    def _validate_temperature(self):
        """
        Validate temperature against provider-specific ranges.
        """
        if self.temperature is None:
            return
        # Find the temperature range for the provider
        temperature_range: Optional[tuple[float, float]] = None
        for client_param_type in ClientParamsModels.__args__:
            if self.provider == client_param_type.provider:
                temperature_range = client_param_type.temperature_range
        if not temperature_range:
            raise ValueError(
                f"Temperature range not found for provider: {self.provider}"
            )
        # Check if the temperature is within the range
        if temperature_range[0] <= self.temperature <= temperature_range[1]:
            return
        else:
            raise ValidationError(
                f"Temperature {self.temperature} is out of range {temperature_range} for provider: {self.provider}"
            )

    def _validate_model(self):
        """
        Validate that the model is supported by at least one provider.
        """
        if not ModelStore.is_supported(self.model):
            raise ValidationError(f"Model '{self.model}' is not supported.")
        return

    # For Caching
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
        schema_str = Parser.as_string(self.response_model),
        # Handle None temperature gracefully
        temp_str = str(self.temperature) if self.temperature is not None else "none"
        params_str = "|".join([messages_str, self.model, temp_str, schema_str])
        return sha256(params_str.encode("utf-8")).hexdigest()

    # Dunder methods for string representation
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
            f"client_params={self.client_params!r}, parser={self.response_model!r})"
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

    # Serialization methods (for Cache, MessageStore, etc.)
    def to_cache_dict(self) -> dict:
        pass

    @classmethod
    def from_cache_dict(cls, cache_dict: dict) -> "Params":
        pass

    # Each customization method first operates on client_params, then constructs base parameters.
    def _to_openai_spec(self) -> dict:
        """
        We use OpenAI spec with OpenAI, Gemini, Ollama, and Perplexity clients.
        """
        if self.client_params:
            assert OpenAIParams.model_validate(self.client_params), f"OpenAIParams expected for OpenAI client, not {type(self.client_params)}."
        base_params = {
            "model": self.model,
            "messages": self.convert_messages(),
            "response_model": self.response_model,
            "temperature": self.temperature,
            "stream": self.stream,
        }

        # Automatically include all client params
        if self.provider == "ollama":
            if self.client_params:
                # Ollama expects options to be nested under "extra_body"
                base_params.update({"extra_body": {"options": self.client_params}})
        else:
            if self.client_params:
                base_params.update(self.client_params)

        # Filter out None values and return
        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }  # Actually filter None values EXCEPT for response_model, as Instructor expects it to be present

    def to_openai(self) -> dict:
        if self.client_params:
            assert OpenAIParams.model_validate(self.client_params), f"OpenAIParams expected for OpenAI client, not {type(self.client_params)}."
        return self._to_openai_spec()

    def to_ollama(self) -> dict:
        """
        We should set num_ctx so we have maximal context window.
        Recall that Ollama with Instructor/OpenAI spec expects options to be nested under extra_body, so we have a special case within to_openai_spec.
        """
        if self.client_params:
            assert OllamaParams.model_validate(self.client_params), f"OllamaParams expected for Ollama client, not {type(self.client_params)}."
        # Set num_ctx to the maximum context window for the model.
        num_ctx = ModelStore.get_num_ctx(self.model)
        if num_ctx is None:
            raise ValueError(
                f"Model '{self.model}' does not have a defined context window."
            )
        if self.client_params:
            self.client_params["num_ctx"] = num_ctx
        else:
            self.client_params = {"num_ctx": num_ctx}
        return self._to_openai_spec()

    def to_anthropic(self) -> dict:
        """
        Convert parameters to Anthropic format.
        Key differences from OpenAI:
        1. System messages become a separate 'system' parameter
        2. max_tokens is required
        3. No response_model in the API call params
        """
        if self.client_params:
            assert AnthropicParams.model_validate(self.client_params), f"AnthropicParams expected for Anthropic client, not {type(self.client_params)}."
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
            "max_retries": 0,
            "response_model": self.response_model,
            "temperature": (
                self.temperature if self.temperature is not None else 1.0
            ),  # Default to 1.0 if not set
            "max_tokens": 4000,  # Default max tokens for Anthropic
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
            base_params.update(self.client_params)

        return {
            k: v
            for k, v in base_params.items()
            if v is not None or k == "response_model"
        }

    def to_google(self) -> dict:
        if self.client_params:
            assert GoogleParams.model_validate(self.client_params), f"GoogleParams expected for Google client, not {type(self.client_params)}."
            # Remove "frequency_penalty" key if it exists, as Google does not use it (unlike OpenAI).
            self.client_params.pop("frequency_penalty", None)
        return self._to_openai_spec()

    def to_perplexity(self) -> dict:
        if self.client_params:
            assert PerplexityParams.model_validate(self.client_params), f"PerplexityParams expected for Perplexity client, not {type(self.client_params)}."
            # Remove "stop" key if it exists, as Perplexity does not use it (unlike OpenAI).
            self.client_params.pop("stop", None)
        return self._to_openai_spec()
