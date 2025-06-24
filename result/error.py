from typing import Optional, Any
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

    def __str__(self) -> str:
        """
        Print this like a normal error message + stack trace if available.
        """
        base_message = f"[{self.info.timestamp}] {self.info.category.upper()} - {self.info.code}: {self.info.message}"
        if self.detail and self.detail.stack_trace:
            return f"{base_message}\nStack Trace:\n{self.detail.stack_trace}"
        return base_message
    
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

