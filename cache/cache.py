from Chain import Response
from pydantic.dataclasses import dataclass
import sqlite3

"""
Response(content='Sure, here are ten examples of mammals:\n\n1. Hum, status='success', prompt='name ten mammals', model='gpt-4o', duration=1.3020920753479004, messages=[Message(role='user', content='name ten mammals'),) 
"""

db_name = ".chain_cache.db"


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

    def __init__(db_name: str):
        self.db_name: str = db_name
        self.cache_dict: dict = dict()

    def create_cached_request(response: Response) -> CachedRequest:
        user_input, llm_output, model = (
            str(response.prompt),
            str(response.content),
            response.model,
        )
        return CachedRequest(user_input, llm_output, model)

    def load_db(db_name: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS cached_requests (user_input TEXT, llm_output TEXT, model TEXT)"
        )
        return conn, cursor

    def insert_cached_request(cursor: sqlite3.Cursor, cached_request: CachedRequest):
        cursor.execute(
            "INSERT INTO cached_requests (user_input, llm_output, model) VALUES (?, ?, ?)",
            (
                cached_request.user_input,
                cached_request.llm_output,
                cached_request.model,
            ),
        )

    def retrieve_cached_requests(
        cursor: sqlite3.Cursor, user_input: str, model: str
    ) -> set[CachedRequest]:
        cursor.execute(
            "SELECT * FROM cached_requests WHERE user_input = ? AND model = ?",
            (user_input, model),
        )
        data = cursor.fetchall()
        return {CachedRequest(*row) for row in data}

    def generate_in_memory_dict(cachedrequests: set[CachedRequest]) -> dict:
        return {(cr.user_input, cr.model): cr.llm_output for cr in cachedrequests}
