"""
TODO:
- add "raw = True" option to query in Chain.model.client
- use raw response for the actual caching below
- still use ObjectCache however to serialize the pydantic object (perhaps hashing the raw response as the key for retrieval)
"""

from pydantic import field_validator, BaseModel
from pydantic.dataclasses import dataclass
from pathlib import Path
import sqlite3


# Define a dataclass to store the cached request
@dataclass(slots=True, frozen=True)
class CachedRequest:
    """
    Simple dataclass to store the cached request.
    slots and frozen are used to make the class immutable / hashable.
    If user_input is a list of messages, we flatten it to a string.
    """

    user_input: str | list
    llm_output: str | BaseModel
    model: str

    @field_validator("user_input")
    def convert_input_to_str(cls, v):
        """
        Flatten the user_input to a string if it is a list of messages.
        Also stripping to regularize the input.
        """
        if isinstance(v, list):
            v = str(v[-1].content)
        return v.strip()


class ChainCache:
    """
    Class to handle the caching of requests.
    If the same rendered prompt string + model name have been seen before, we return the cached response.
    """

    def __init__(
        self,
        db_name: str | Path = ".cache.db",
    ):
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
        # Create CachedRequest objects
        return {CachedRequest(*row) for row in data}

    def generate_in_memory_dict(self, cachedrequests: set[CachedRequest]) -> dict:
        return {(cr.user_input, cr.model): cr.llm_output for cr in cachedrequests}

    def cache_lookup(self, user_input: str | list, model: str) -> str | None:
        """
        Checks if there is a match for the CacheEntry, returns if yes, returns None if no.
        """
        # Regularize this, like we do with CachedRequest class
        if isinstance(user_input, list):
            user_input = user_input[-1].content
        user_input = str(user_input).strip()
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
