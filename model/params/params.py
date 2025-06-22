from pydantic import BaseModel, Field
from typing import Optional, Any, ClassVar, TYPE_CHECKING
from Chain.message.message import Message
from Chain.model.models.models import ModelStore

if TYPE_CHECKING:
    from Chain.parser.parser import Parser

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
    # Class vars
    provider: ClassVar[str] = "ollama"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    class OllamaClientParams(BaseModel):
        """
        Parameters specific to Ollama clients.
        """
        temperature: Optional[float] = None
        top_k: Optional[int] = None
        top_p: Optional[float] = None
        repeat_penalty: Optional[float] = None
        stop: Optional[list[str]] = None
    
    client_params: Optional[OllamaClientParams] = Field(default=None, description="Parameters specific to Ollama clients.")

class AnthropicParams(ClientParams):
    """
    Parameters specific to Anthropic clients.
    """
    # Class vars
    provider: ClassVar[str] = "anthropic"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 1.0)

    max_tokens_to_sample: Optional[int] = None

class PerplexityParams(OpenAIParams):
    """
    Parameters specific to Perplexity clients.
    Inherits from OpenAIParams to maintain compatibility with OpenAI API spec.
    """
    # Class vars
    provider: ClassVar[str] = "perplexity"
    temperature_range: ClassVar[tuple[float, float]] = (0.0, 2.0)

# Union type for all client-specific parameters
ClientParamsTypes = OpenAIParams | OllamaParams | AnthropicParams | GeminiParams | PerplexityParams 

# Main Params class
class Params(BaseModel):
    """
    Parameters that are constructed by Model and are sent to Clients.
    """
    # Core parameters
    model: str = Field(..., description="The model identifier to use for inference.")
    messages: list[Message] = Field(default_factory=list, description="List of messages to send to the model. Can include text, images, audio, etc.")
   
    # Optional parameters
    temperature: Optional[float] = Field(default=None, description="Temperature for sampling. If None, defaults to provider-specific value.")
    stream: bool = False
    raw: bool = False
    verbose: bool = True
    
    # Post model init parameters
    parser: Optional[Any] = Field(default=None, description="Parser to convert messages to a specific format. Not intended for direct use.")
    query_input: str | Message | list[Message] | None = Field(default=None, description="Various possible input types that we coerce to a list of Messages. Not intended for direct use.")
    provider: Optional[str] = Field(default=None, description="Provider of the model, populated post init. Not intended for direct use.")
     
    # Client parameters (embedded in dict for now)
    client_params: Optional[ClientParams] = Field(default = None, description="Client-specific parameters. Can be OpenAIParams, OllamaParams, AnthropicParams, etc.")
    
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
            raise ValueError(f"Temperature range not found for provider: {self.provider}")
        if temperature_range[0] <= self.temperature <= temperature_range[1]:
            return 
        else:
            raise ValueError(f"Temperature {self.temperature} is out of range {temperature_range} for provider: {self.provider}")

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
                raise ValueError("All items in query_input list must be of type Message.")
            input_messages = self.query_input
        else:
            raise ValueError("query_input must be a Message, a string, or a list of Messages.")
        self.query_input = None
        if self.messages is not None:
            input_messages = self.messages + input_messages
            self.messages = input_messages
        else:
            self.messages = input_messages

    def __hash__(self) -> str:
        """
        Generate a hash for the Params instance for our caching system.
        """
        pass

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
    def to_openai(self) -> dict:
        pass

    def to_ollama(self) -> dict:
        pass

    def to_anthropic(self) -> dict:
        pass

    def to_gemini(self) -> dict:
        pass

    def to_perplexity(self) -> dict:
        pass
