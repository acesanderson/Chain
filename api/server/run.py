from fastapi import FastAPI, status
from Chain.api.server.ChainRequest import ChainRequest, process_ChainRequest
from Chain.response.response import Response
import uvicorn

# Create a FastAPI instance
app = FastAPI()

# Our actual query
@app.post("/query/", status_code=status.HTTP_201_CREATED)
async def query(request: ChainRequest) -> Response:
    print("Request received:\n")
    print(request.model_dump_json)
    response = process_ChainRequest(request)
    return response


def main():
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()

