import inspect
import time
from functools import wraps


def extract_query_preview(input_data, max_length=20):
    """Extract a preview of the query for display purposes"""
    if isinstance(input_data, str):
        return (
            input_data[:max_length] + "..."
            if len(input_data) > max_length
            else input_data
        )
    elif isinstance(input_data, list):
        # Find last message with role="user"
        for message in reversed(input_data):
            if hasattr(message, "role") and message.role == "user":
                content = message.content
                # Handle Pydantic objects
                if hasattr(content, "model_dump_json"):
                    content = content.model_dump_json()
                else:
                    content = str(content)
                return (
                    content[:max_length] + "..."
                    if len(content) > max_length
                    else content
                )
        return "No user message found"
    else:
        return str(input_data)[:max_length] + "..."


def sync_wrapper(model_instance, func, handler, query_preview, *args, **kwargs):
    """Synchronous wrapper for progress display"""
    model_name = model_instance.model
    handler.emit_started(model_name, query_preview)

    start_time = time.time()
    try:
        result = func(model_instance, *args, **kwargs)
        duration = time.time() - start_time
        handler.emit_complete(model_name, query_preview, duration)
        return result
    except KeyboardInterrupt:
        handler.emit_canceled(model_name, query_preview)
        raise
    except Exception as e:
        handler.emit_failed(model_name, query_preview, str(e))
        raise


async def async_wrapper(model_instance, func, handler, query_preview, *args, **kwargs):
    """Asynchronous wrapper for progress display"""
    model_name = model_instance.model
    handler.emit_started(model_name, query_preview)

    start_time = time.time()
    try:
        result = await func(model_instance, *args, **kwargs)
        duration = time.time() - start_time
        handler.emit_complete(model_name, query_preview, duration)
        return result
    except KeyboardInterrupt:
        handler.emit_canceled(model_name, query_preview)
        raise
    except Exception as e:
        handler.emit_failed(model_name, query_preview, str(e))
        raise


def progress_display(func):
    """
    Decorator that adds progress display to Model.query() methods.
    Automatically detects sync vs async and uses appropriate wrapper.
    """

    @wraps(func)
    def sync_decorator(self, *args, **kwargs):
        verbose = kwargs.get("verbose", True)

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

        return sync_wrapper(self, func, handler, query_preview, *args, **kwargs)

    @wraps(func)
    async def async_decorator(self, *args, **kwargs):
        verbose = kwargs.get("verbose", True)

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
