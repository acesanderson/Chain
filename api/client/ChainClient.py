from Chain.api.server.ChainRequest import ChainRequest
from Chain.response.response import Response
# from Chain.api.server.test_ChainServer import example_requests
from Chain.model.clients.client import Client
import requests
import subprocess


def get_url() -> str:
    hostnames = {
        "remote": ["Botvinnik", "bianders-mn7180.linkedin.biz", "Caruana"],
        "local": ["AlphaBlue"],
    }
    # get hostname using subprocess
    hostname = subprocess.check_output(["hostname"]).decode("utf-8").strip()
    if hostname in hostnames["local"]:
        url = "http://localhost:8000/query"
    else:
        url = "http://10.0.0.87:8000/query"
    return url


class ChainClient(Client):
    """
    A client for sending requests to the Chain server.
    Currently defined entirely by url endpoint.
    """

    def __init__(self, url: str):
        self.url = url
        self.client = self._initialize_client()

    def _initialize_client(self) -> None:
        """
        This method is not needed for this client as it does not require any special initialization.
        """
        pass

    def send_request(self, chainrequest: ChainRequest) -> Response | None:
        """
        Send a request to the Chain server and return the response.
        """
        request = chainrequest.model_dump()
        http_response = requests.post(
            url=self.url, json=request, headers={"Content-Type": "application/json"}
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
#     client = ChainClient(url=get_url())
#     for example_request in example_requests:
#         response = client.send_request(example_request)
#         print(response)
