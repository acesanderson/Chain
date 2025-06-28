from Chain.result.error import ChainError, ErrorInfo, ErrorDetail
from Chain.result.response import Response
from Chain.model.params.params import Params
from Chain.message.message import Message
from Chain.message.messages import Messages
from datetime import datetime

sample_message = Message(role="user", content="Hello, world!")

sample_messages = Messages([
    Message(role="user", content="Hello, world!"),
    Message(role="assistant", content="Hello! How can I assist you today?")
    Message(role="user", content="What is the weather like?"),
])

sample_error = ChainError(
    info=ErrorInfo(
        code="ERR001",
        message="An unexpected error occurred",
        category="RuntimeError",
        timestamp=datetime.now()
    ),
    detail=ErrorDetail(
        exception_type="ValueError",
        stack_trace="Traceback (most recent call last): ...",  # Example stack trace
        raw_response=None,  # Could be a response object or None
        request_params=None,  # Could be a dict of request parameters or None
        retry_count=0  # Number of retries attempted
    )
)

sample_response = Response(
    messages=Messages([
        Message(role="user", content="Hello, world!"),
        Message(role="assistant", content="Hello! How can I assist you today?")
    ]),
    params=Params(model="gpt-3.5-turbo"),
    duration=1.23
)


sample_params = Params(
    model="gpt-3.5-turbo",
    messages=[
        Message(role="user", content="Hello, world!"),
        Message(role="assistant", content="Hello! How can I assist you today?")
    ],
    temperature=0.7,
    stream=True,
    parser=None  # Assuming no parser for this example
)
