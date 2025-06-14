from fastapi import FastAPI, status
from Chain.api.server.ChainRequest import ChainRequest, process_ChainRequest
from Chain.response.response import Response
from Chain.model.model import Model
import uvicorn, ollama, json

# Create a FastAPI instance
app = FastAPI()


# We want an initial handshake to ensure the server is running as well as for server to list their available models.
@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> dict:
    print("Root endpoint accessed.")
    models = Model.models()
    print("Available models:", json.dumps(models, indent=2))
    return models


# Our actual query
@app.post("/query/", status_code=status.HTTP_201_CREATED)
async def query(request: ChainRequest) -> Response:
    print("Request received:\n")
    print(request.model_dump_json)
    response = process_ChainRequest(request)
    return response


def main():
    uvicorn.run("Chain.api.server.run:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()

