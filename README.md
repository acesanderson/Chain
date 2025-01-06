## Chain

A lightweight framework for building LLM applications

## Installation

### Install the Chain package
```bash
git clone https://github.com/acesanderson/Chain
cd Chain
pip install -r requirements.txt
pip install .
```
### Setup

In a file titled `.env` in the root directory of your project, add the following environment variables for the LLM providers you wish to use:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

If you wish to use local models, you will need to install [Ollama](https://github.com/ollama/ollama). Recommended models (from personal experience): llama3.1:latest, qwen.

## Usage

### Basic usage
```python
from Chain import Prompt, Model, Chain, Parser, Message, MessageStore

# List available models
print(Model.models)

# {'ollama': ['phi3:latest', 'phi:latest', 'llama3.2:latest', 'llama3.1:latest', 'llama3.1:70b-instruct-q2_K', 'dolphin-mixtral:latest', 'dolphin-mixtral:8x7b', 'mistral:latest'], 'openai': ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo-0125', 'gpt-4o-mini', 'o1-preview', 'o1-mini'], 'anthropic': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20240620'], 'google': ['gemini-1.5-pro-latest', 'gemini-1.5-flash-latest', 'gemini-1.0-pro-latest'], 'groq': ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']}

# Simple GPT chain

prompt = Prompt("What is the capital of France?")
model = Model("gpt")
chain = Chain(prompt, model)
response = chain.run()
print(response)

# Response(content='The capital of France is Paris.', status='success', prompt='What is the capitol of France?', model='gpt-4o', duration=0.9836819171905518, messages=[], variables=<built-in function input>)
print(response.content)

# "The capital of France is Paris."
```

### Working with input variables

Chain uses Jinja templating to allow for dynamic prompts. You can pass variables to the prompt and model using the `input_variables` parameter in the `Chain` and `Prompt` classes.

```python
prompt_string = "Name ten {{things}} that are red."
prompt = Prompt(prompt_string)
model = Model("claude-3-5-sonnet-20240620")
chain = Chain(prompt, model)
response = chain.run(input_variables={"things": "frogs"})

'Here are ten different types of frogs:\n\n1. Red-Eyed Tree Frog\n2. Poison Dart Frog\n3. American Bullfrog\n4. African Clawed Frog\n5. Green Tree Frog\n6. Glass Frog\n7. Tomato Frog\n8. Pacman Frog (Horned Frog)\n9. Northern Leopard Frog\n10. Goliath Frog\n\nThese frogs represent a diverse range of species from various habitats around the world, each with unique characteristics and adaptations.'
```

### Working with structured output (function calling)

Chain uses the excellent [Instructor](https://github.com/instructor-ai/instructor) library to get pydantic objects from LLM calls. You can use the `Parser` class to parse the output of a chain into a pydantic object.

```python
from pydantic import BaseModel
from Chain import Parser

class Frog(BaseModel):
    name: str
    color: str
    habitat: str

prompt = Prompt("Come up with a frog that is red and lives in the rainforest")
model = Model("claude-3-5-sonnet-20240620")
parser = Parser(Frog)
chain = Chain(prompt, model, parser) # Note that the parser is passed to the Chain class in initialization
response = chain.run()
print(response.content)

# Frog(name='Fred', color='red', habitat='rainforest')
```

### Advanced
- You can create a messagestore to either: save response objects in a Message format throughout a script; save a persistent chat conversation between sessions; log prompt flows.
- You can set system prompts with the create_messages method in the MessageStore class.
- Every client has an AsyncClient implementation for async calls.

### Upcoming features
- Chatbot class
- HuggingFace/unsloth implementation for working with finetuned models
- Agentic support for tools, resources, using MCP protocol

