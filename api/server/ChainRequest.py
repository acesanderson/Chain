from Chain.model.model import Model
from Chain.response.response import Response
from Chain.message.message import Message
from Chain.parser.parser import Parser
from pydantic import BaseModel
from typing import Optional


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
    response = client.query(
        model=model,
        input=input,
        parser=parser,
        raw=raw,
        temperature=temperature,
    )
    return response
