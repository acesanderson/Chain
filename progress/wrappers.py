import inspect
import time
from functools import wraps

def extract_query_preview(input_data, max_length=30):
    """Extract a preview of the query for display purposes"""
    if isinstance(input_data, str):
        # Strip whitespace and replace newlines with spaces
        preview = input_data.strip().replace('\n', ' ').replace('\r', ' ')
        return preview[:max_length] + "..." if len(preview) > max_length else preview
    elif isinstance(input_data, list):
        # Find last message with role="user"
        for message in reversed(input_data):
            if hasattr(message, 'role') and message.role == "user":
                content = message.content
                # Handle Pydantic objects
                if hasattr(content, 'model_dump_json'):
                    content = content.model_dump_json()
                else:
                    content = str(content)
                # Strip whitespace and replace newlines with spaces
                content = content.strip().replace('\n', ' ').replace('\r', ' ')
                return content[:max_length] + "..." if len(content) > max_length else content
        return "No user message found"
    else:
        preview = str(input_data).strip().replace('\n', ' ').replace('\r', ' ')
        return preview[:max_length] + "..."

def sync_wrapper(model_instance, func, handler, query_preview, index, total, *args, **kwargs):
    """Synchronous wrapper for progress display with in-place updates"""
    model_name = model_instance.model
    display_preview = f"[{index}/{total}] {query_preview}" if index is not None else query_preview
    
    # Show starting state
    if hasattr(handler, 'show_spinner'):
        handler.show_spinner(model_name, display_preview)
    else:
        handler.emit_started(model_name, display_preview)

    start_time = time.time()
    try:
        result = func(model_instance, *args, **kwargs)
        duration = time.time() - start_time
        
        # Update same line with completion
        if hasattr(handler, 'show_complete'):
            handler.show_complete(model_name, display_preview, duration)
        else:
            handler.emit_complete(model_name, display_preview, duration)
        return result
    except KeyboardInterrupt:
        if hasattr(handler, 'show_canceled'):
            handler.show_canceled(model_name, display_preview)
        else:
            handler.emit_canceled(model_name, display_preview)
        raise
    except Exception as e:
        if hasattr(handler, 'show_failed'):
            handler.show_failed(model_name, display_preview, str(e))
        else:
            handler.emit_failed(model_name, display_preview, str(e))
        raise

async def async_wrapper(model_instance, func, handler, query_preview, *args, **kwargs):
    """Asynchronous wrapper for progress display with in-place updates"""
    model_name = model_instance.model
    
    # Show starting state
    if hasattr(handler, 'show_spinner'):
        handler.show_spinner(model_name, query_preview)
    else:
        handler.emit_started(model_name, query_preview)

    start_time = time.time()
    try:
        result = await func(model_instance, *args, **kwargs)
        duration = time.time() - start_time
        
        # Update same line with completion
        if hasattr(handler, 'show_complete'):
            handler.show_complete(model_name, query_preview, duration)
        else:
            handler.emit_complete(model_name, query_preview, duration)
        return result
    except KeyboardInterrupt:
        if hasattr(handler, 'show_canceled'):
            handler.show_canceled(model_name, query_preview)
        else:
            handler.emit_canceled(model_name, query_preview)
        raise
    except Exception as e:
        if hasattr(handler, 'show_failed'):
            handler.show_failed(model_name, query_preview, str(e))
        else:
            handler.emit_failed(model_name, query_preview, str(e))
        raise

def progress_display(func):
    """
    Decorator that adds progress display to Model.query() methods.
    Automatically detects sync vs async and uses appropriate wrapper.
    """
    @wraps(func)
    def sync_decorator(self, *args, **kwargs):
        verbose = kwargs.pop('verbose', True)
        index = kwargs.pop('index', None)
        total = kwargs.pop('total', None)

        # Validate index/total parameters
        if (index is None) != (total is None):
            raise ValueError("Must provide both 'index' and 'total' parameters or neither")

        if not verbose:
            return func(self, *args, **kwargs)

        # Extract query preview from first argument
        query_preview = extract_query_preview(args[0] if args else "")

        if self.console:
            # Lazy import Rich components only when needed
            from Chain.progress.handlers import RichProgressHandler
            handler = RichProgressHandler(self.console)
        else:
            # Built-in PlainProgressHandler - no imports
            from Chain.progress.handlers import PlainProgressHandler
            handler = PlainProgressHandler()

        return sync_wrapper(self, func, handler, query_preview, index, total, *args, **kwargs)

    @wraps(func)
    async def async_decorator(self, *args, **kwargs):
        verbose = kwargs.pop('verbose', True)
        # Note: index/total not supported for async operations

        if not verbose:
            return await func(self, *args, **kwargs)

        # Extract query preview from first argument
        query_preview = extract_query_preview(args[0] if args else "")

        if self.console:
            # Lazy import Rich components only when needed
            from Chain.progress.handlers import RichProgressHandler
            handler = RichProgressHandler(self.console)
        else:
            # Built-in PlainProgressHandler - no imports
            from Chain.progress.handlers import PlainProgressHandler
            handler = PlainProgressHandler()

        return await async_wrapper(self, func, handler, query_preview, *args, **kwargs)

    # Return the appropriate decorator based on function type
    if inspect.iscoroutinefunction(func):
        return async_decorator
    else:
        return sync_decorator


async def concurrent_wrapper(operation, tracker):
    """Wrap individual async operations for concurrent tracking"""
    try:
        tracker.operation_started()
        result = await operation
        tracker.operation_completed()
        return result
    except Exception as e:
        tracker.operation_failed()
        raise  # Re-raise the exception


def create_concurrent_progress_tracker(console, total: int):
    """Factory function to create appropriate concurrent tracker"""
    if console:
        from Chain.progress.handlers import RichProgressHandler
        handler = RichProgressHandler(console)
    else:
        from Chain.progress.handlers import PlainProgressHandler
        handler = PlainProgressHandler()
   
    from Chain.progress.tracker import ConcurrentTracker
    return ConcurrentTracker(handler, total)
