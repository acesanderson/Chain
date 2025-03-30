# main.py
from fastapi import FastAPI

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
