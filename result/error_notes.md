Looking at your `Response` class and the `Params -> Response` flow, I'd recommend implementing errors as a **union type** rather than trying to stuff error info into the existing Response model. Here's why and how:

## Recommended Approach: Union Type with Result Pattern

```python
from typing import Union, Optional, Any
from dataclasses import dataclass
import traceback
from datetime import datetime

@dataclass(frozen=True)
class ErrorInfo:
    """Simple error information"""
    code: str                    # "validation_error", "api_error", etc.
    message: str                 # Human-readable error message
    category: str                # "client", "server", "network", "parsing"
    timestamp: datetime
    
@dataclass(frozen=True)
class ErrorDetail:
    """Detailed error information for debugging"""
    exception_type: str          # "ValidationError", "APIException"
    stack_trace: Optional[str]   # Full stack trace if available
    raw_response: Optional[Any]  # Original response that caused error
    request_params: Optional[dict] # Params that led to error
    retry_count: Optional[int]   # If retries were attempted
    
@dataclass(frozen=True)
class ChainError:
    """Complete error information"""
    info: ErrorInfo
    detail: Optional[ErrorDetail] = None
    
    @classmethod
    def from_exception(cls, exc: Exception, code: str, category: str, **context) -> 'ChainError':
        """Create ChainError from an exception with full context"""
        info = ErrorInfo(
            code=code,
            message=str(exc),
            category=category,
            timestamp=datetime.now()
        )
        
        detail = ErrorDetail(
            exception_type=type(exc).__name__,
            stack_trace=traceback.format_exc(),
            raw_response=context.get('raw_response'),
            request_params=context.get('request_params'),
            retry_count=context.get('retry_count')
        )
        
        return cls(info=info, detail=detail)
    
    @classmethod
    def simple(cls, code: str, message: str, category: str) -> 'ChainError':
        """Create simple error without exception details"""
        info = ErrorInfo(
            code=code,
            message=message,
            category=category,
            timestamp=datetime.now()
        )
        return cls(info=info)

# Your Result type
ChainResult = Union[Response, ChainError]
```

## Updated Response Class

Keep your `Response` clean and focused on success cases:

```python
class Response(BaseModel):
    content: Any
    status: str = "success"  # Always success for Response objects
    prompt: str | None
    model: str
    duration: float | None
    messages: Optional[list[Message | ImageMessage]]
    
    # Keep your existing methods
    def __repr__(self): ...
    def __str__(self): ...
    def __len__(self): ...
```

## Usage in Your Chain Methods

```python
class Chain:
    def run(self, **kwargs) -> ChainResult:
        try:
            start_time = time.time()
            
            # Your existing logic
            result = self.model.query(...)
            duration = time.time() - start_time
            
            return Response(
                content=result,
                model=self.model.name,
                duration=duration,
                # ... other fields
            )
            
        except ValidationError as e:
            return ChainError.from_exception(
                e, 
                code="validation_error",
                category="client",
                request_params=kwargs
            )
        except APIException as e:
            return ChainError.from_exception(
                e,
                code="api_error", 
                category="server",
                raw_response=getattr(e, 'response', None)
            )
        except Exception as e:
            return ChainError.from_exception(
                e,
                code="unknown_error",
                category="client"
            )
```

## Pattern Matching for Clean Error Handling

```python
def handle_chain_result(result: ChainResult) -> str:
    match result:
        case Response() as response:
            return str(response)
        
        case ChainError() as error:
            if error.info.category == "client":
                return f"Request error: {error.info.message}"
            elif error.info.category == "server":
                return f"Service error: {error.info.message}"
            else:
                return f"Error: {error.info.message}"

# Or with your monadic approach:
def run_chain_safely(chain: Chain, **kwargs) -> Result[Response, ChainError]:
    result = chain.run(**kwargs)
    match result:
        case Response() as response:
            return Ok(response)
        case ChainError() as error:
            return Err(error)
```

## Integration with Your Clients

```python
class AnthropicClientSync:
    def query(self, params: Params) -> ChainResult:
        try:
            api_params = params.to_anthropic()
            raw_result = self._client.chat.completions.create(**api_params)
            content = self._extract_content(raw_result)
            
            return Response(
                content=content,
                model=params.model,
                # ... other fields
            )
            
        except ValidationError as e:
            return ChainError.from_exception(
                e,
                code="params_validation_error",
                category="client",
                request_params=params.model_dump()
            )
        except AnthropicAPIError as e:
            return ChainError.from_exception(
                e,
                code="anthropic_api_error", 
                category="server",
                raw_response=getattr(e, 'response', None)
            )
```

## Why This Approach?

1. **Clean separation** - Response is only for success, ChainError only for failures
2. **Rich error context** - Stack traces and debug info when needed
3. **Simple error info** - Clean message and categorization for users
4. **Type safety** - Union type makes error handling explicit
5. **Flexible** - Can add more error types or response types later
6. **Debuggable** - Full context preserved for troubleshooting

## Alternative: If You Must Use Single Type

If you really want to keep everything in `Response`, you could do:

```python
class Response(BaseModel):
    content: Any | None = None
    status: str  # "success", "error" 
    error: Optional[ChainError] = None
    prompt: str | None
    model: str
    duration: float | None
    messages: Optional[list[Message | ImageMessage]]
    
    @property
    def is_success(self) -> bool:
        return self.status == "success"
    
    @property 
    def is_error(self) -> bool:
        return self.status == "error"
```

But I'd strongly recommend the union type approach - it's more explicit, type-safe, and follows the functional programming principles you're adopting with monads.
