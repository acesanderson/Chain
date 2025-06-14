"""
This is our client for ChainServer.
Great for ollama calls on a powerful remote computer.
"""

from Chain.api.server.ChainRequest import ChainRequest
from Chain.response.response import Response

# from Chain.tests.test_ChainServer import example_requests
from Chain.model.clients.client import Client
from Chain.parser.parser import Parser
from typing import Optional
import requests, subprocess


class ServerClient(Client):
    """
    A client for sending requests to the Chain server.
    Currently defined entirely by url endpoint.
    """

    def __init__(self, url: str = ""):
        if not url:
            self.url = self._get_url()
        else:
            self.url = url
        self.models = {}
        self.client = self._initialize_client()

    def _initialize_client(self):
        """
        Initialization is a handshake with the server, where we also get our list of ollama_models.
        """
        response = requests.get(self.url)
        if response.status_code == 200:
            self.models = response.json()
            print(f"Available models: {self.models}")
        else:
            print(f"Failed to initialize client. Status code: {response.status_code}")
            raise Exception("Failed to initialize ChainServer client.")
        return self

    def _get_api_key(self):
        """
        Same as above
        """
        return ""

    def _get_url(self) -> str:
        hostnames = {
            "remote": ["Botvinnik", "bianders-mn7180.linkedin.biz", "Caruana"],
            "local": ["AlphaBlue"],
        }
        # get hostname using subprocess
        hostname = subprocess.check_output(["hostname"]).decode("utf-8").strip()
        if hostname in hostnames["local"]:
            url = "http://localhost:8000/"
        else:
            url = "http://10.0.0.87:8000/"
        return url

    def tokenize(self, model: str, text: str) -> None:  # type: ignore
        raise NotImplementedError("Tokenization not implemented for ChainServer yet.")


class ServerClientSync(ServerClient):

    def query(self, model: str, input: "str | list", parser: Parser | None = None, raw=False, temperature: Optional[float] = None) -> "BaseModel | str":  # type: ignore
        """
        Our query function here needs to: create a ChainRequest object, submit it to self._send_request, and return the response.
        """
        pydantic_model = parser.pydantic_model if parser else None
        chainrequest = ChainRequest(
            model=model,
            input=input,
            pydantic_model=pydantic_model,
            raw=raw,
            temperature=temperature,
        )
        response = self._send_request(chainrequest)
        if response:
            return response.content
        else:
            raise Exception("Failed to get a valid response from the server.")

    def _send_request(self, chainrequest: ChainRequest) -> Response | None:
        """
        Send a request to the Chain server and return the response.
        """
        request = chainrequest.model_dump()
        query_url = self.url + "query"
        http_response = requests.post(
            url=query_url, json=request, headers={"Content-Type": "application/json"}
        )
        if http_response.status_code == 201:
            response_data = http_response.json()
            pydantic_response = Response(**response_data)  # For Pydantic v1.x
            # Now you can access the data through your model
            return pydantic_response
        else:
            print(f"Error: {http_response.status_code}")
            print(f"Response: {http_response.text}")


# if __name__ == "__main__":
#     client = ServerClientSync()
#     for example_request in example_requests:
#         response = client._send_request(example_request)
#         print(response)
