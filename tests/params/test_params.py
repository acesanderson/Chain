from Chain.model.model import Model
from Chain.model.params.params import Params
from Chain.result.response import Response

# OpenAI client params - leveraging unique OpenAI features
openai_params = {
    "max_tokens": 150,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.3,
    "stop": [".", "!", "?"],
    "safety_settings": {"content_policy": "strict"}
}

# Anthropic client params - using Anthropic-specific sampling
anthropic_params = {
    "top_k": 40,
    "top_p": 0.9,
    "stop_sequences": ["Human:", "Assistant:", "\n\n"]
}

# Google client params - inherits OpenAI spec but with safety settings
google_params = {
    "max_tokens": 200,
    "frequency_penalty": 0.2,
    "presence_penalty": 0.1,
    "stop": ["\n\n", "END"],
    "safety_settings": {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
    }
}

# Perplexity client params - OpenAI spec focused on research/search
perplexity_params = {
    "max_tokens": 300,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
    "stop": ["Sources:", "References:"]
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
    params = Params.from_query_input(query_input: "name ten mammals", model = "gpt-3.5-turbo-0125", client_params=openai_params)
    response = model.query(params)
    assert isinstance(response, Response)

def test_anthropic_params():
    model = Model("haikue")
    params = Params.from_query_input(query_input: "name ten mammals", model = "haiku", client_params=anthropic_params)
    response = model.query(params)
    assert isinstance(response, Response)

def test_google_params():
    model = Model("flash")
    params = Params.from_query_input(query_input: "name ten mammals", model = "flash", client_params=google_params)
    response = model.query(params)
    assert isinstance(response, Response)

def test_perplexity_params():
    model = Model("sonar")
    params = Params.from_query_input(query_input: "name ten mammals", model = "sonar", client_params=perplexity_params)
    response = model.query(params)
    assert isinstance(response, Response)

def test_ollama_params():
    model = Model("llama3.1:latest")
    params = Params.from_query_input(query_input: "name ten mammals", model = "llama3.1:latest", client_params=ollama_params)
    response = model.query(params)
    assert isinstance(response, Response)


