from Chain.model.model import Model
from Chain.model.params.params import Params
from Chain.result.response import Response

# OpenAI client params - leveraging unique OpenAI features
openai_params = {
    "max_tokens": 150,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.3,
    "stop": [".", "!", "?"],
}

# Anthropic client params - using Anthropic-specific sampling
anthropic_params = {
    "top_k": 40,
    "top_p": 0.9,
    "stop_sequences": ["Human:", "Assistant:"]
}

# Google client params - inherits OpenAI spec but with safety settings
google_params = {
    "max_tokens": 200,
    "presence_penalty": 0.1,
    "stop": ["\n\n", "END"],
    }

# Perplexity client params - OpenAI spec focused on research/search
perplexity_params = {
    "max_tokens": 300,
    "frequency_penalty": 0.1,
}

# Ollama client params - extensive Ollama-specific options
ollama_params = {
    "num_ctx": 4096,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "stop": ["<|im_end|>", "\n\n"]
}

def test_openai_params():
    model = Model("gpt-3.5-turbo-0125")
    params = Params.from_query_input(query_input = "name ten mammals", model = "gpt-3.5-turbo-0125", client_params=openai_params)
    response = model.query(params = params)
    assert isinstance(response, Response)

def test_anthropic_params():
    model = Model("claude-3-5-haiku-20241022")
    params = Params.from_query_input(query_input = "name ten mammals", model = "claude-3-5-haiku-20241022", client_params=anthropic_params)
    response = model.query(params = params)
    assert isinstance(response, Response)

def test_google_params():
    model = Model("gemini-1.5-flash")
    params = Params.from_query_input(query_input = "name ten mammals", model = "gemini-1.5-flash", client_params=google_params)
    response = model.query(params = params)
    assert isinstance(response, Response)

def test_perplexity_params():
    model = Model("sonar")
    params = Params.from_query_input(query_input = "name ten mammals", model = "sonar", client_params=perplexity_params)
    response = model.query(params = params)
    assert isinstance(response, Response)

def test_ollama_params():
    model = Model("llama3.1:latest")
    params = Params.from_query_input(query_input = "name ten mammals", model = "llama3.1:latest", client_params=ollama_params)
    response = model.query(params = params)
    assert isinstance(response, Response)


