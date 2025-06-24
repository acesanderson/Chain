from Chain.message.message import Message


class Messages:
    def __init__(self, items: list[Message]):
        self._messages = items

    def __getitem__(self, index):
        return self._messages[index]

    def __len__(self):
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def append(self, message: Message):
        self._messages.append(message)

    def model_dump(self) -> dict:
        return {}

    def model_dump_json(self) -> str:
        return ""
