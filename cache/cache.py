from Chain import Response
from pydantic.dataclasses import dataclass
import sqlite3


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
    """
    Class to handle the caching of requests.
    If the same rendered prompt string + model name have been seen before, we return the cached response.
    """

    def __init__(db_name: str):
        self.db_name = db_name
        self.conn, self.cursor = self.load_db()
        self.cached_requests = self.retrieve_cached_requests(self.cursor)
        self.cache_dict = generate_in_memory_dict(self.cached_requests)

    def create_cached_request(response: Response) -> CachedRequest:  # type: ignore
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

    def retrieve_cached_requests(cursor: sqlite3.Cursor) -> set[CachedRequest]:
        cursor.execute("SELECT * FROM cached_requests")
        data = cursor.fetchall()
        return {CachedRequest(*row) for row in data}

    def generate_in_memory_dict(cachedrequests: set[CachedRequest]) -> dict:
        return {(cr.user_input, cr.model): cr.llm_output for cr in cachedrequests}

    def cache_lookup(user_input: str, model: str) -> Response | None:
        """
        Checks if there is a match for the CacheEntry, returns if yes, returns None if no.
        """
        key = (user_input, model)
        try:
            value = self.cache_dict[key]
        except:
            value = None
        return value


if __name__ == "__main__":
    cache = ChainCache(db_name=".example.db")
    examples = [
        Response(
            prompt="name five mammals",
            content="(1) lizard (2) dog (3) bird (4) dinosaur (5) human",
            model="gpt",
        ),
        Response(prompt="what is the capital of france?", content="Paris", model="gpt"),
        Response(
            prompt="is this thing on?",
            content="yes, do you have a question?",
            model="gpt",
        ),
        Response(prompt="what OS am I using?", content="macOS", model="gpt"),
    ]
