"""
Chain caching system using JSON serialization instead of cloudpickle.
Provides transparent caching for Chain responses with automatic serialization/deserialization.
"""

import json
import sqlite3
import hashlib
from typing import Optional, Any
from datetime import datetime
from pathlib import Path


class ChainCache:
    """
    SQLite-based cache for Chain responses using JSON serialization.
    Automatically handles serialization/deserialization of Response and Params objects.
    """

    def __init__(self, db_path: str = "chain_cache.db"):
        """
        Initialize cache with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        """Create cache table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                response_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.connection.commit()

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached response by key.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Deserialized response object or None if not found
        """
        self.cursor.execute(
            "SELECT response_data FROM cache WHERE cache_key = ?", 
            (cache_key,)
        )
        result = self.cursor.fetchone()
        
        if result:
            response_data = result[0]
            return self._deserialize_from_cache(response_data)
        return None

    def set(self, cache_key: str, response: Any):
        """
        Store response in cache.
        
        Args:
            cache_key: Unique cache key
            response: Response object to cache
        """
        serialized_data = self._serialize_for_cache(response)
        
        self.cursor.execute("""
            INSERT OR REPLACE INTO cache (cache_key, response_data)
            VALUES (?, ?)
        """, (cache_key, serialized_data))
        self.connection.commit()

    def _serialize_for_cache(self, obj: Any) -> str:
        """
        Serialize object using new cache dict methods.
        
        Args:
            obj: Object to serialize (should have to_cache_dict method)
            
        Returns:
            JSON string representation
        """
        if hasattr(obj, 'to_cache_dict'):
            cache_dict = obj.to_cache_dict()
            return json.dumps(cache_dict, default=str, ensure_ascii=False)
        else:
            # Fallback for simple objects
            return json.dumps(str(obj), ensure_ascii=False)

    def _deserialize_from_cache(self, data: str) -> Any:
        """
        Deserialize object using cache dict methods.
        Type information is embedded in the JSON structure.
        
        Args:
            data: JSON string to deserialize
            
        Returns:
            Reconstructed object or None if deserialization fails
        """
        try:
            cache_dict = json.loads(data)
            
            # Determine object type based on structure and deserialize accordingly
            if 'messages' in cache_dict and 'params' in cache_dict and 'duration' in cache_dict:
                # This is a Response object
                from Chain.result.response import Response
                return Response.from_cache_dict(cache_dict)
            elif 'model' in cache_dict and ('provider' in cache_dict or 'messages' in cache_dict):
                # This is a Params object
                from Chain.model.params.params import Params
                return Params.from_cache_dict(cache_dict)
            elif 'messages' in cache_dict and isinstance(cache_dict['messages'], list):
                # This is a Messages object
                from Chain.message.messages import Messages
                return Messages.from_cache_dict(cache_dict)
            elif 'message_type' in cache_dict:
                # This is a specific Message type
                message_type = cache_dict['message_type']
                if message_type == 'ImageMessage':
                    from Chain.message.imagemessage import ImageMessage
                    return ImageMessage.from_cache_dict(cache_dict)
                elif message_type == 'AudioMessage':
                    from Chain.message.audiomessage import AudioMessage
                    return AudioMessage.from_cache_dict(cache_dict)
                else:
                    # Standard Message
                    from Chain.message.message import Message
                    return Message.from_cache_dict(cache_dict)
            else:
                # Unknown structure - return as dict
                return cache_dict
                
        except (json.JSONDecodeError, KeyError, ImportError, AttributeError) as e:
            print(f"Cache deserialization error: {e}")
            return None

    def clear_cache(self):
        """Clear all cached responses."""
        self.cursor.execute("DELETE FROM cache")
        self.connection.commit()

    def retrieve_cached_requests(self):
        """
        Get all cached requests for debugging.
        
        Returns:
            List of tuples: (cache_key, created_at)
        """
        self.cursor.execute("SELECT cache_key, created_at FROM cache ORDER BY created_at DESC")
        return self.cursor.fetchall()

    def cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        self.cursor.execute("SELECT COUNT(*) FROM cache")
        total_entries = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT SUM(LENGTH(response_data)) FROM cache")
        total_size = self.cursor.fetchone()[0] or 0
        
        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "database_path": str(self.db_path)
        }

    def delete_cache_entry(self, cache_key: str):
        """
        Delete a specific cache entry.
        
        Args:
            cache_key: Key of the entry to delete
        """
        self.cursor.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
        self.connection.commit()

    def cleanup_old_entries(self, days: int = 30):
        """
        Remove cache entries older than specified days.
        
        Args:
            days: Number of days to keep entries
        """
        self.cursor.execute("""
            DELETE FROM cache 
            WHERE created_at < datetime('now', '-{} days')
        """.format(days))
        self.connection.commit()

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self):
        """String representation."""
        stats = self.cache_stats()
        return f"ChainCache(entries={stats['total_entries']}, size={stats['total_size_bytes']} bytes, db='{self.db_path}')"


# Cache helper functions
def check_cache(model_instance, params):
    """
    Check if response exists in cache using new JSON serialization.
    
    Args:
        model_instance: Model instance with _chain_cache attribute
        params: Params object containing request parameters
        
    Returns:
        Cached Response object or None if not found
    """
    if not model_instance._chain_cache:
        return None
        
    cache_key = params.generate_cache_key()
    return model_instance._chain_cache.get(cache_key)


def update_cache(model_instance, params, response):
    """
    Store response in cache using new JSON serialization.
    
    Args:
        model_instance: Model instance with _chain_cache attribute
        params: Params object containing request parameters
        response: Response object to cache
    """
    if not model_instance._chain_cache:
        return
        
    cache_key = params.generate_cache_key()
    model_instance._chain_cache.set(cache_key, response)
