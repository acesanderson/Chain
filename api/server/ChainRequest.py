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
    parser: Optional[Parser]
    raw: bool = False
    input_variables: Optional[dict]
    temperature: Optional[float] = None


def process_ChainRequest(chainrequest: ChainRequest) -> Response:
    """
    Takes a request and returns a response.
    """
    # Get our variables
    model = Model(chainrequest.model)
    input = chainrequest.input
    parser = chainrequest.parser
    raw = chainrequest.raw
    input_variables = chainrequest.input_variables
    temperature = chainrequest.temperature
    # Run the query at client level
    model_obj = Model(model)
    client = model_obj._client
    response = client.query(
        model=model,
        input=input,
        parser=parser,
        raw=raw,
        temperature=temperature,
    )
    return response
