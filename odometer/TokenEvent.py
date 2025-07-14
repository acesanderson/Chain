from pydantic import BaseModel, Field

class TokenEvent(BaseModel)
    provider: str = Field(..., description = "Model provider (Anthropic, OpenAI, Google, Ollama, Perplexity, etc.).")
    model: str = Field(..., description = "Specific model by name (gpt-4o, claude-3.5-sonnet, etc.).")
    input_tokens: int = Field(..., description = "Prompt tokens as defined and provided in API response.")
    output_tokens: int = Field(..., description = "Output tokens as defined and provided in API response.")
    timestamp: int = Field(..., description = "Unix epoch timestamp.")       # Unix epoch timestamp
    host: str = Field(..., description = "Simple host detection for multi-machine tracking.")
