**ProxyModel** is our Model implementation to make our ChainServer and ChainClient plug right into the base Chain syntax.

## Why ProxyModel Works

1. **Accurate** - Your Chain server is literally acting as a proxy
2. **Clear distinction** - `Model` = direct API calls, `ProxyModel` = via your server
3. **Familiar pattern** - "Proxy" is a well-understood networking/architecture term
4. **Future-proof** - Could proxy to multiple servers, load balance, etc.

## Clean Implementation

```python
class Model:
    """Direct API calls to LLM providers"""
    pass

class ProxyModel(Model):
    """Model accessed via Chain server proxy"""
    
    def __init__(self, url: str, model: str = "auto"):
        self.url = url
        self.proxy_client = ChainClient(url)
        self.available_models = self._discover_models()
        
        if model == "auto":
            model = self.available_models[0]
            
        # Don't call super().__init__ since we don't want direct client setup
        self.model = self._validate_model(model)
    
    def _discover_models(self):
        """Get available models from proxy server"""
        return self.proxy_client.get_available_models()
    
    def query(self, input, **kwargs):
        """Proxy the query through Chain server"""
        request = ChainRequest(
            prompt=None if isinstance(input, list) else input,
            messages=input if isinstance(input, list) else None,
            model=self.model,
            **kwargs
        )
        response = self.proxy_client.send_request(request)
        return response.content
```

## Usage Examples

```python
# Direct API access
openai_model = Model("gpt-4o")
local_ollama = Model("llama3.1:latest")

# Via your GPU server proxy  
gpu_ollama = ProxyModel("http://gpu-server:8000", "llama3.1:70b")
gpu_openai = ProxyModel("http://gpu-server:8000", "gpt-4o")  # Server calls OpenAI

# All identical interface
response1 = openai_model.query("Hello")
response2 = gpu_ollama.query("Hello") 
```

## Optional: Factory Method

If you want even cleaner syntax:

```python
class Model:
    @classmethod
    def proxy(cls, url: str, model: str = "auto"):
        """Create a proxy model"""
        return ProxyModel(url, model)

# Usage
model = Model.proxy("http://gpu-server:8000", "llama3.1:70b")
```

**ProxyModel** perfectly captures what's happening - your Chain server is proxying requests to various LLM providers, whether they're running locally on the server (Ollama) or remotely (OpenAI/Anthropic APIs).
