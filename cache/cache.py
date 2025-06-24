from pydantic import BaseModel
from typing import Callable, Any, Optional
import sqlite3, json, importlib


class ChainCache:
    def __init__(self, db_path: str = "chain_cache.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_class_name TEXT,
                    content_module TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def store_in_cache(self, cache_key: str, content: Any):
        serialized_content = None
        content_class_name = None
        content_module = None

        if isinstance(content, BaseModel):
            serialized_content = content.model_dump_json()
            content_class_name = content.__class__.__name__
            content_module = content.__class__.__module__
        elif isinstance(content, (str, int, float, bool, list, dict)):
            # Direct JSON serializable types
            serialized_content = json.dumps(content)
            content_class_name = type(content).__name__
            content_module = None
        else:
            # If it's not a BaseModel and not a simple JSON-serializable type,
            # we cannot cache it without specific serialization logic or pickling.
            # For this scenario, we will *not* store it in cache.
            # print(f"Warning: Content of type {type(content)} is not a BaseModel or simple JSON type. Not caching.")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (cache_key, content, content_class_name, content_module) VALUES (?, ?, ?, ?)",
                (cache_key, serialized_content, content_class_name, content_module),
            )
            conn.commit()

    def retrieve_from_cache(self, cache_key: str) -> Optional[Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT content, content_class_name, content_module FROM cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()

        if row:
            content_data, content_class_name, content_module = row

            try:
                # 1. Try to dynamically load the class if class_name is available
                if content_class_name:
                    cls = None
                    if content_module:
                        try:
                            module = importlib.import_module(content_module)
                            cls = getattr(module, content_class_name)
                        except (ImportError, AttributeError):
                            # Fallback for common types that might not be in the stored module
                            pass

                    # Common built-in types or well-known library types
                    if cls is None:
                        if content_class_name == "str":
                            cls = str
                        elif content_class_name == "int":
                            cls = int
                        elif content_class_name == "float":
                            cls = float
                        elif content_class_name == "bool":
                            cls = bool
                        elif content_class_name == "list":
                            cls = list
                        elif content_class_name == "dict":
                            cls = dict
                        elif (
                            content_class_name == "ChatCompletion"
                        ):  # Specific OpenAI type
                            import openai.types.chat.chat_completion as oc

                            cls = oc.ChatCompletion
                        elif (
                            content_class_name == "Completion"
                        ):  # Specific OpenAI type for /v1/completions
                            import openai.types.completion as oac

                            cls = oac.Completion
                        # Add other specific third-party types here if needed (e.g., Anthropic, Google)
                        # Example for Anthropic Message object
                        elif (
                            content_class_name == "Message"
                        ):  # Assumes anthropic.types.messages.Message
                            import anthropic.types.messages as atm

                            cls = atm.Message
                        # Example for Google (Gemini via OpenAI SDK) Message object
                        # elif content_class_name == "GoogleMessage":
                        #    import google.generativeai.types as ggt
                        #    cls = ggt.GoogleMessage # This needs to be precisely correct

                    if cls:
                        # If it's a Pydantic model, use model_validate_json
                        if issubclass(cls, BaseModel):
                            return cls.model_validate_json(content_data)
                        else:  # For other types, parse JSON directly
                            return json.loads(content_data)
                    else:
                        # If class couldn't be resolved, try simple JSON load as a last resort
                        return json.loads(content_data)

                else:  # Fallback for old cache entries without class info, or if class_name is None for some reason
                    return json.loads(content_data)  # Assume JSON for consistency

            except (json.JSONDecodeError, ImportError, AttributeError, Exception) as e:
                # print(f"Error deserializing cached content for key {cache_key} (class: {content_class_name}, module: {content_module}): {e}")
                return None
        return None

    def clear_cache(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()
        # print(f"Cache '{self.db_path}' cleared.") # Suppressed for cleaner test output

    def retrieve_cached_requests(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT cache_key, content, content_class_name FROM cache"
            )
            return cursor.fetchall()


# --- check_cache_and_query and check_cache_and_query_async functions ---
# These functions will be imported by Model and ModelAsync respectively.
# They handle the caching logic by interacting with the ChainCache instance.


def check_cache_and_query(
    model_instance: Any, params: Any, query_func: Callable
) -> Any:
    # Do not cache if streaming is requested, as stream objects are not serializable.
    if params.stream:
        return query_func()

    if model_instance._chain_cache:
        cache_key = params.generate_cache_key()
        cached_result = model_instance._chain_cache.retrieve_from_cache(cache_key)

        if cached_result is not None:
            print(f"Cache HIT for key: {cache_key[:10]}...")  # For debugging
            return cached_result
        else:
            # print(f"Cache MISS for key: {cache_key[:10]}...") # For debugging
            result = query_func()

            # Check if the result is a stream object (chunk).
            # We import these types conditionally to avoid circular imports and unnecessary dependencies.
            is_stream_chunk = False
            try:
                # OpenAI ChatCompletionChunk
                from openai.types.chat.chat_completion import ChatCompletionChunk

                if isinstance(result, ChatCompletionChunk):
                    is_stream_chunk = True
            except ImportError:
                pass
            try:
                # Anthropic MessageStreamEvent (for streaming from Anthropic SDK)
                from anthropic.types.messages import MessageStreamEvent

                if isinstance(result, MessageStreamEvent):
                    is_stream_chunk = True
            except ImportError:
                pass

            if not is_stream_chunk:  # Only store if it's not a stream chunk
                model_instance._chain_cache.store_in_cache(cache_key, result)
            return result
    else:
        return query_func()


async def check_cache_and_query_async(
    model_instance: Any, params: Any, query_func: Callable
) -> Any:
    # Do not cache if streaming is requested, as stream objects are not serializable.
    if params.stream:
        return await query_func()  # Await here as it's an async query_func

    if model_instance._chain_cache:
        cache_key = params.generate_cache_key()
        cached_result = model_instance._chain_cache.retrieve_from_cache(
            cache_key
        )  # retrieve_from_cache is sync

        if cached_result is not None:
            # print(f"Async Cache HIT for key: {cache_key[:10]}...") # For debugging
            return cached_result
        else:
            # print(f"Async Cache MISS for key: {cache_key[:10]}...") # For debugging
            result = await query_func()  # Await the async query function

            # Check if the result is a stream object (chunk).
            is_stream_chunk = False
            try:
                from openai.types.chat.chat_completion import ChatCompletionChunk

                if isinstance(result, ChatCompletionChunk):
                    is_stream_chunk = True
            except ImportError:
                pass
            try:
                from anthropic.types.messages import MessageStreamEvent

                if isinstance(result, MessageStreamEvent):
                    is_stream_chunk = True
            except ImportError:
                pass

            if not is_stream_chunk:  # Only store if it's not a stream chunk
                model_instance._chain_cache.store_in_cache(
                    cache_key, result
                )  # store_in_cache is sync
            return result
    else:
        return await query_func()
