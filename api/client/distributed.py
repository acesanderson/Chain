from Chain import ChainClient
from Chain.api.server.test_ChainServer import example_requests
from itertools import cycle

urls = [
    "http://10.0.0.82:8000/query",
    "http://localhost:8000/query",
    "http://10.0.0.191:8000/query",
]
caruana_client = ChainClient(urls[0])
petrosian_client = ChainClient(urls[1])
botvinnik_client = ChainClient(urls[2])
clients = cycle([caruana_client, petrosian_client, botvinnik_client])


if __name__ == "__main__":
    examples_requests = example_requests.extend(example_requests)
    for example_request in example_requests:
        client = next(clients)
        print(f"Using client: {client.url}")
        response = client.send_request(example_request)
        print(f"Response: {response}")
        print(f"Duration: {response.duration}")
        print("-" * 40)
