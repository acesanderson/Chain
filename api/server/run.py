from fastapi import FastAPI, status
from ChainRequest import ChainRequest, process_ChainRequest, Response

# Create a FastAPI instance
app = FastAPI()


# Define a path operation decorator
@app.get("/")  # GET request to the root path "/"
async def read_root():
    # Return a simple JSON response
    return {"message": "Hello World"}


# Another simple endpoint
@app.get("/items/{item_id}")  # Path parameter 'item_id'
async def read_item(item_id: int):  # Type hint declares item_id must be an int
    return {"item_id": item_id, "description": f"This is item {item_id}"}


# Our actual query
@app.post("/query/", status_code=status.HTTP_201_CREATED)
async def query(request: ChainRequest) -> Response:
    print("Request received:\n")
    print(request.model_dump_json)
    response = process_ChainRequest(request)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
