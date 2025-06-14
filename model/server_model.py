from Chain.model.clients.server_client import ServerClientSync
from Chain.model.model import Model


class ServerModel(Model):
    """
    Model for interacting with a ChainServer.
    Primary use case: ollama models using my desktop with RTX 5090.
    """

    def __init__(self, model: str = "llama3.3", url: str = "") -> None:
        """
        Initialize the ModelClient with the ChainClient.
        :param url: The URL of the ChainServer.
        """
        self._client = ServerClientSync(url=url)
        self.models = self._client.models
        self.model = model


if __name__ == "__main__":
    model = ServerModel("gpt")
    print(model.models)
