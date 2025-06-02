# Chain

A lightweight, unified framework for building LLM applications with support for multiple providers, structured outputs, and async operations.

## Quick Start

```bash
pip install git+https://github.com/acesanderson/Chain.git
```

**Basic usage in 30 seconds:**

```python
from Chain import Chain, Model, Prompt

# Simple completion
chain = Chain(
    prompt=Prompt("What is the capital of France?"),
    model=Model("gpt-4o")
)
response = chain.run()
print(response.content)  # "The capital of France is Paris."
```

## Why Chain?

- **Universal LLM Interface**: One API for OpenAI, Anthropic, Google, Groq, Ollama, and more
- **Structured Outputs**: Built-in Pydantic integration via Instructor
- **Async-First**: Native async support for high-throughput applications  
- **Smart Caching**: Automatic response caching to reduce costs and latency
- **Rich Templating**: Jinja2 templates with variable validation

## Installation & Setup

### Prerequisites
- Python 3.8+
- API keys for desired providers

### Environment Setup
Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

For local models, install [Ollama](https://github.com/ollama/ollama):
```bash
# Install Ollama, then pull models
ollama pull llama3.1:latest
ollama pull qwen2.5:latest
```

## Core Examples

### 1. Multi-Provider Model Access

```python
from Chain import Model

# See all available models
print(Model.models())
# Returns: {'openai': ['gpt-4o', 'gpt-4o-mini', ...], 'anthropic': ['claude-3-5-sonnet', ...], ...}

# Use any model with identical syntax
models = [
    Model("gpt-4o"),           # OpenAI
    Model("claude-3-5-sonnet"), # Anthropic  
    Model("llama3.1:latest"),   # Ollama (local)
    Model("gemini-1.5-pro"),    # Google
]
```

### 2. Dynamic Prompts with Variables

```python
from Chain import Chain, Model, Prompt

prompt = Prompt("List 5 {{category}} that are {{color}}")
chain = Chain(prompt=prompt, model=Model("claude-3-5-sonnet"))

response = chain.run(input_variables={
    "category": "animals", 
    "color": "red"
})
print(response.content)
```

### 3. Structured Outputs (Function Calling)

```python
from Chain import Chain, Model, Prompt, Parser
from pydantic import BaseModel

class Animal(BaseModel):
    name: str
    species: str
    habitat: str
    conservation_status: str

prompt = Prompt("Create a detailed profile for a {{animal_type}}")
parser = Parser(Animal)
chain = Chain(prompt=prompt, model=Model("gpt-4o"), parser=parser)

result = chain.run(input_variables={"animal_type": "arctic fox"})
print(result.content)  # Returns Animal object
print(result.content.conservation_status)  # "Least Concern"
```

### 4. Async Batch Processing

```python
from Chain import AsyncChain, ModelAsync

# Process multiple prompts concurrently
prompts = [
    "Explain quantum computing",
    "What is machine learning?", 
    "Define blockchain technology"
]

model = ModelAsync("gpt-4o-mini")
chain = AsyncChain(model=model)
responses = chain.run(prompt_strings=prompts)

for response in responses:
    print(f"Response: {response.content[:100]}...")
```

### 5. Image Analysis

```python
from Chain import Chain, Model, ImageMessage
import base64

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

image_msg = ImageMessage(
    role="user",
    text_content="What's in this image?",
    image_content=image_data,
    mime_type="image/jpeg"
)

chain = Chain(model=Model("gpt-4o"))
response = chain.run(messages=[image_msg])
```

## Advanced Features

### Conversation Management
```python
from Chain import MessageStore, Message

# Persistent chat history
store = MessageStore(history_file="chat.pkl", log_file="chat.log")
store.add_new("system", "You are a helpful assistant")
store.add_new("user", "Hello!")

chain = Chain(model=Model("gpt-4o"))
response = chain.run(messages=store.messages)
```

### Smart Caching
```python
from Chain import ChainCache, Model

# Enable caching for cost savings
Model._chain_cache = ChainCache("cache.db")

# Subsequent identical calls return cached results
model = Model("gpt-4o")
result1 = model.query("What is Python?")  # API call
result2 = model.query("What is Python?")  # Cache hit!
```

### CLI Applications
```python
from Chain import ChainCLI

# Create custom CLI apps
class MyCLI(ChainCLI):
    @arg("-t")
    def arg_translate(self, text):
        """Translate text to French"""
        # Custom functionality
        pass

cli = MyCLI(name="My AI Assistant")
cli.run()
```

## Model Support

| Provider | Models | Features |
|----------|--------|----------|
| **OpenAI** | GPT-4o, GPT-4o-mini, o1-preview | Function calling, vision, reasoning |
| **Anthropic** | Claude 3.5 Sonnet, Haiku | Large context, vision, tool use |
| **Google** | Gemini 1.5/2.0 Pro/Flash | Multimodal, long context |
| **Groq** | Llama, Mixtral, Gemma | Ultra-fast inference |
| **Ollama** | 100+ local models | Privacy, offline, custom models |

## Architecture

```
Chain Framework
├── Model Layer       # Universal LLM interface
├── Prompt System     # Jinja2 templating + validation  
├── Message Handling  # Conversation + multimodal support
├── Response System   # Structured outputs + metadata
├── Async Engine      # Concurrent processing
└── Extensions        # CLI, caching, API server
```

## Contributing

Chain is actively developed and welcomes contributions:

```bash
git clone https://github.com/acesanderson/Chain
cd Chain
pip install -r requirements.txt
pytest  # Run tests
```

## What's Next

- 🔧 **Agentic Support**: Tools and resources via MCP protocol
- 🤗 **HuggingFace Integration**: Fine-tuned model support  
- 💬 **Enhanced Chat**: Improved conversation management
- 🔌 **Plugin System**: Extensible tool ecosystem
