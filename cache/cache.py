from pydantic.dataclasses import dataclass
import sqlite3
from functools import wraps


# Define a dataclass to store the cached request
@dataclass(slots=True, frozen=True)
class CachedRequest:
    """
    Simple dataclass to store the cached request.
    slots and frozen are used to make the class immutable / hashable.
    """

    user_input: str
    llm_output: str
    model: str


class ChainCache:
    """
    Class to handle the caching of requests.
    If the same rendered prompt string + model name have been seen before, we return the cached response.
    """

    def __init__(self, db_name: str):
        self.db_name = db_name
        self.conn, self.cursor = self.load_db()
        self.cached_requests = self.retrieve_cached_requests()
        self.cache_dict = self.generate_in_memory_dict(self.cached_requests)

    def load_db(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS cached_requests (user_input TEXT, llm_output TEXT, model TEXT)"
        )
        return conn, cursor

    def insert_cached_request(self, cached_request: CachedRequest):
        self.cursor.execute(
            "INSERT INTO cached_requests (user_input, llm_output, model) VALUES (?, ?, ?)",
            (
                cached_request.user_input,
                cached_request.llm_output,
                cached_request.model,
            ),
        )
        self.conn.commit()
        self.cache_dict[(cached_request.user_input, cached_request.model)] = (
            cached_request.llm_output
        )

    def retrieve_cached_requests(self) -> set[CachedRequest]:
        self.cursor.execute("SELECT * FROM cached_requests")
        data = self.cursor.fetchall()
        return {CachedRequest(*row) for row in data}

    def generate_in_memory_dict(self, cachedrequests: set[CachedRequest]) -> dict:
        return {(cr.user_input, cr.model): cr.llm_output for cr in cachedrequests}

    def cache_lookup(self, user_input: str, model: str) -> str | None:
        """
        Checks if there is a match for the CacheEntry, returns if yes, returns None if no.
        """
        key = (user_input, model)
        try:
            value = self.cache_dict[key]
        except:
            value = None
        return value

    def clear_cache(self):
        self.cursor.execute("DELETE FROM cached_requests")
        self.conn.commit()
        self.cache_dict = {}
        print("Cache cleared.")

    def __bool__(self):
        """
        We want this to return True if the object is initialized.
        We wouldn't need this if we didn't also want a __len__ method.
        (If __len__ returns 0, bool() returns False for your average object).
        """
        return True

    def __len__(self):
        """
        Note: see the __bool__ method above for extra context.
        """
        return len(self.cache_dict)


if __name__ == "__main__":
    cache = ChainCache(db_name=".example.db")
    lookup_examples = [
        {"user_input": "name five mammals", "model": "gpt"},
        {"user_input": "what is the capital of germany?", "model": "gpt"},
        {"user_input": "what is the capital of france?", "model": "gpt"},
        {"user_input": "what is your context cutoff?", "model": "gpt"},
        {"user_input": "is this thing on?", "model": "gpt"},
        {"user_input": "what OS am I using?", "model": "gpt"},
        {"user_input": "what OS am I using?", "model": "claude"},
        {"user_input": "what OS am I using?", "model": "gemini"},
    ]
    for lookup_example in lookup_examples:
        print(lookup_example, cache.cache_lookup(**lookup_example))
