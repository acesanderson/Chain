from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date

# Our types
image_formats = Literal["jpg", "png", "webp"]
audio_formats = Literal["mp3", "wav", "flac"]


class ModelCapabilities(BaseModel):
    text_generation: bool = Field(
        description="Supports text generation",
    )
    function_calling: bool = Field(
        description="Supports function calling",
    )
    image_input: bool = Field(
        description="Supports image input",
    )
    audio_input: bool = Field(
        description="Supports audio input",
    )
    image_output: bool = Field(
        description="Supports image output",
    )
    audio_output: bool = Field(
        description="Supports audio output, i.e. TTS",
    )
    reasoning: bool = Field(
        description="Supports reasoning tasks",
    )


class ModelFormats(BaseModel):
    image_input: list[image_formats] = Field(
        description="Supported image input formats",
    )
    audio_input: list[audio_formats] = Field(
        description="Supported audio input formats",
    )
    image_output: list[image_formats] = Field(
        description="Supported image output formats",
    )
    audio_output: list[audio_formats] = Field(
        description="Supported audio output formats",
    )


class ModelSpec(BaseModel):
    model_name: str = Field(..., description="Unique identifier for the model")
    provider: Literal[
        "openai",
        "anthropic",
        "google",
        "groq",
        "ollama",
    ] = Field(
        ...,
        description="Provider of the model",
    )
    deployment: Literal["local", "api"] = Field(
        ...,
        description="Deployment type of the model",
    )
    capabilities: ModelCapabilities = Field(
        description="Capabilities of the model",
    )
    formats: ModelFormats = Field(
        description="Supported formats for input and output",
    )
    context_window: int = Field(
        ...,
        description="Maximum context window size in tokens",
    )
    knowledge_cutoff: Optional[date] = Field(
        default=None,
        description="Date when the model's knowledge was last updated; None if unknown or model is updated continuously",
    )

    description: str = Field(
        default="",
        description="A brief description of the model",
    )

    def __str__(self):
        return f"{self.provider} {self.model_name} ({self.deployment})"

    def __repr__(self):
        return f"ModelSpec(model_name={self.model_name}, provider={self.provider}, deployment={self.deployment})"

    @property
    def card(self):
        if "console" in globals():
            console = globals()["console"]
        else:
            from rich.console import Console

            console = Console()

        output = ""
        for attr, value in self.__dict__.items():
            output += f"[bold green]{attr}[/bold green] = [yellow]{value}[/yellow]\n"

        console.print(output)


class ModelSpecs(BaseModel):
    models: list[ModelSpec] = Field(
        description="List of model specifications",
    )
