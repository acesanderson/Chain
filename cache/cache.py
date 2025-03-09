"""
To do: allow llm_output to be a BaseModel. This is non-trivial to implement since deserializing objects back from json loses the pydantic class information.
The solution would involve:
- separating concerns: each class (Curation, Course, etc.) should have an inheritance pattern.
 - Raw_Curation(BaseModel) -- the actual dataclass with fields and types; Curation(Raw_Curation) -- the actual class with methods and properties.
 - on serialization, save not just values but the class name and the schema, something like this:
    - {"class_name": "Curation", "schema": {"title": "string", "description": "string"}, "values": {"title": "foo", "description": "bar"}}
 - on deserialization, check the class name and schema, and create the object accordingly.
 - then, when using Parser and Instructor, use the Raw class and then bless it with the methods and properties of the actual class afterwards.
"""

from pydantic import field_validator, BaseModel
from pydantic.dataclasses import dataclass
import cloudpickle
import hashlib
import json
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


class ObjectCache:
    """
    Class to handle the caching of objects.
    This is only used if you are caching pydantic models (as Parser, Instructor, etc.).
    """

    def __init__(self, object_store: str):
        self.object_store = object_store
        self.object_dict: dict[str, BaseModel] = self.load_object_cache()

    def load_object_cache(self) -> dict:
        """
        Load the object cache from the object store.
        If it doesn't exist, create an empty dictionary.
        """
        try:
            with open(self.object_store, "rb") as f:
                return cloudpickle.load(f)
        except FileNotFoundError:
            return {}
        except EOFError:
            print(
                f"EOF Error loading object cache. The file {self.object_store} may be corrupted."
            )
            self.clear_object_cache()
            return {}

    def save_object_cache(self):
        """
        Save the object cache to the object store.
        """
        with open(self.object_store, "wb") as f:
            cloudpickle.dump(self.object_dict, f)

    def clear_object_cache(self):
        """
        Clear the object cache.
        """
        self.object_dict = {}
        self.save_object_cache()
        print("Object cache cleared.")

    def get_object_hash(self, obj: BaseModel) -> str:
        """
        Generate a hash for the object.
        This is used to check if the object already exists in the cache.
        """
        obj_dict = obj.model_dump()
        obj_str = json.dumps(obj_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(obj_str).hexdigest()

    def add_object(self, obj: BaseModel) -> str:
        """
        Assign a UUID to the object and add it to the object cache.
        """
        if not isinstance(obj, BaseModel):
            raise ValueError("Object must be a pydantic model.")
        id = self.get_object_hash(obj)
        self.object_dict[id] = obj
        self.save_object_cache()
        return id

    def get_object(self, obj_id: str) -> BaseModel | None:
        """
        Get the object from the object cache.
        If it doesn't exist, return None.
        """
        return self.object_dict[obj_id]


class ChainCache:
    """
    Class to handle the caching of requests.
    If the same rendered prompt string + model name have been seen before, we return the cached response.
    """

    def __init__(
        self,
        db_name: str = ".cache.db",
        object_cache_name: str = ".object_cache.cloudpickle",
    ):
        self.db_name = db_name
        self.object_cache = ObjectCache(object_cache_name)
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
        if isinstance(cached_request.llm_output, BaseModel):
            id = self.object_cache.add_object(cached_request.llm_output)
            llm_output = "UUID:" + str(id)
        else:
            llm_output = cached_request.llm_output
        self.cursor.execute(
            "INSERT INTO cached_requests (user_input, llm_output, model) VALUES (?, ?, ?)",
            (
                cached_request.user_input,
                llm_output,
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
        # Unserialize any pydantic objects
        modified_data = []
        for row in data:
            if row[1].startswith("UUID:"):
                id = row[1][5:]
                obj = self.object_cache.get_object(id)
                if obj:
                    new_row = (row[0], obj, row[2])
                    modified_data.append(new_row)
                else:
                    print(
                        f"Object with ID {id} not found in object cache. Suggest rebuilding the cache."
                    )
            else:
                modified_data.append(row)
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
