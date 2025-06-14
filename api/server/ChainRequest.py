from Chain.model.model import Model
from Chain.response.response import Response
from Chain.message.message import Message
from Chain.parser.parser import Parser
from pydantic import BaseModel
from typing import Optional
import time


class ChainRequest(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    model: str
    input: str | list[Message]
    pydantic_model: Optional[BaseModel] = None
    raw: bool = False
    temperature: Optional[float] = None


def process_ChainRequest(chainrequest: ChainRequest) -> Response:
    """
    Takes a request and returns a response.
    """
    # Get our variables
    model = chainrequest.model # This will go in API request
    model_obj = Model(model) # We need this to get client
    input = chainrequest.input
    pydantic_model = chainrequest.pydantic_model
    raw = chainrequest.raw
    temperature = chainrequest.temperature
    # Reconstruct parser if pydantic_model is provided
    parser = Parser(pydantic_model) if pydantic_model else None  # type: ignore
    # Run the query at client level
    client = model_obj._client
    start_time = time.time()
    response = client.query(
        model=model,
        input=input,
        parser=parser,
        raw=raw,
        temperature=temperature,
    )
    end_time = time.time()
    response_obj = Response(
        content = response,
        status = "success",
        prompt = input,
        model = model,
        duration = end_time - start_time,
    )
    return response_obj
