import sqlite3
import json
from typing import Optional, Any, Dict
from pathlib import Path
from Chain.result.response import Response
from Chain.model.params.params import Params


class ChainCache:
    """
    SQLite-based cache for Chain responses with JSON serialization.
    Replaces cloudpickle with proper Pydantic serialization.
    """

    def __init__(self, db_path: str | Path = "chain_cache.db"):
        """
        Initialize the cache with a SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """
        Initialize the SQLite database with required tables.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    response_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get(self, cache_key: str) -> Optional[Response]:
        """
        Retrieve a cached response by cache key.
        
        Args:
            cache_key: SHA256 hash of the request parameters
            
        Returns:
            Response object if found, None otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT response_data FROM cache WHERE cache_key = ?",
                    (cache_key,)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                # Deserialize JSON back to Response object
                response_dict = json.loads(row[0])
                return Response.from_cache_dict(response_dict)
                
        except (sqlite3.Error, json.JSONDecodeError, ValueError) as e:
            # Log error but don't crash - cache miss is better than crash
            print(f"Cache retrieval error for key {cache_key}: {e}")
            return None

    def set(self, cache_key: str, response: Response) -> bool:
        """
        Store a response in the cache.
        
        Args:
            cache_key: SHA256 hash of the request parameters
            response: Response object to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Serialize Response to JSON
            response_dict = response.to_cache_dict()
            response_json = json.dumps(response_dict, ensure_ascii=False)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (cache_key, response_data) VALUES (?, ?)",
                    (cache_key, response_json)
                )
                conn.commit()
                return True
                
        except (sqlite3.Error, json.JSONEncodeError, ValueError) as e:
            # Log error but don't crash - cache miss is better than crash
            print(f"Cache storage error for key {cache_key}: {e}")
            return False

    def clear_cache(self) -> bool:
        """
        Clear all cached entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Cache clear error: {e}")
            return False

    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM cache")
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute(
                    "SELECT MIN(created_at), MAX(created_at) FROM cache"
                )
                min_date, max_date = cursor.fetchone()
                
                return {
                    "total_entries": total_entries,
                    "oldest_entry": min_date,
                    "newest_entry": max_date,
                    "db_path": str(self.db_path)
                }
        except sqlite3.Error as e:
            return {"error": str(e)}

    def retrieve_cached_requests(self) -> list[Dict[str, Any]]:
        """
        Retrieve all cached requests for debugging/inspection.
        
        Returns:
            List of dictionaries with cache key and creation time
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT cache_key, created_at FROM cache ORDER BY created_at DESC"
                )
                return [
                    {"cache_key": row[0], "created_at": row[1]}
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            print(f"Error retrieving cached requests: {e}")
            return []

    def __repr__(self):
        stats = self.cache_stats()
        return f"ChainCache(db_path={self.db_path}, entries={stats.get('total_entries', 'unknown')})"


# Cache helper functions for Model.query integration
def check_cache(model_instance, params: Params) -> Optional[Response]:
    """
    Check if a response is cached for the given parameters.
    
    Args:
        model_instance: Model instance with _chain_cache attribute
        params: Params object containing request parameters
        
    Returns:
        Cached Response if found, None otherwise
    """
    if not model_instance._chain_cache:
        return None
    
    cache_key = params.generate_cache_key()
    return model_instance._chain_cache.get(cache_key)


def update_cache(model_instance, params: Params, response: Response) -> bool:
    """
    Update the cache with a new response.
    
    Args:
        model_instance: Model instance with _chain_cache attribute
        params: Params object containing request parameters
        response: Response object to cache
        
    Returns:
        True if successfully cached, False otherwise
    """
    if not model_instance._chain_cache:
        return False
    
    cache_key = params.generate_cache_key()
    return model_instance._chain_cache.set(cache_key, response)


async def check_cache_and_query_async(model_instance, params: Params, execute_query_func):
    """
    Async version of cache check and query execution.
    
    Args:
        model_instance: ModelAsync instance with _chain_cache attribute
        params: Params object containing request parameters
        execute_query_func: Async function that executes the actual query
        
    Returns:
        Response object (from cache or fresh query)
    """
    # Check cache first
    if model_instance._chain_cache:
        cache_key = params.generate_cache_key()
        cached_response = model_instance._chain_cache.get(cache_key)
        if cached_response:
            return cached_response
    
    # Execute query
    response = await execute_query_func()
    
    # Update cache
    if model_instance._chain_cache and isinstance(response, Response):
        cache_key = params.generate_cache_key()
        model_instance._chain_cache.set(cache_key, response)
    
    return response
