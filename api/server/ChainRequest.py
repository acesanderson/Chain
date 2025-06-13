from Chain.chain.chain import Chain
from Chain.model.model import Model
from Chain.prompt.prompt import Prompt
from Chain.response.response import Response
from Chain.message.message import Message
from pydantic import BaseModel
from typing import Optional


class ChainRequest(BaseModel):
    prompt: Optional[str]
    messages: Optional[list[Message]]
    model: str
    input_variables: Optional[dict]


def process_ChainRequest(chainrequest: ChainRequest) -> Response:
    """
    Takes a request and returns a response.
    """
    model = Model(chainrequest.model)
    messages = chainrequest.messages
    input_variables = chainrequest.input_variables
    if chainrequest.prompt:
        prompt = Prompt(chainrequest.prompt)
    else:
        prompt = None
    chain = Chain(model=model, prompt=prompt)
    response = chain.run(messages=messages, input_variables=input_variables)  # type: ignore
    return response
