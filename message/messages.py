from Chain.message.message import Message


class Messages(list):
    def __init__(self, iterable=None):
        iterable = iterable or []
        # Validate everything before initializing
        for item in iterable:
            self._validate(item)
        super().__init__(iterable)

    def append(self, item):
        self._validate(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._validate(item)
        super().extend(iterable)

    def insert(self, index, item):
        self._validate(item)
        super().insert(index, item)

    def __setitem__(self, index, value):
        # Handle slice assignment separately
        if isinstance(index, slice):
            for item in value:
                self._validate(item)
        else:
            self._validate(value)
        super().__setitem__(index, value)

    @staticmethod
    def _validate(item):
        if not isinstance(item, Message):
            raise TypeError(f"Only Message instances are allowed, got: {type(item)}")

    def model_dump(self) -> dict:
        return {"messages": [msg.model_dump() for msg in self]}

    def model_dump_json(self) -> str:
        return "[" + ", ".join(msg.model_dump_json() for msg in self) + "]"
