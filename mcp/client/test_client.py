"""
As I develop the server, this is literally just a script that sends a request to each endpoint.
"""


## Core Lifecycle Methods
def initialize():
    """
    1. **initialize** - First endpoint called to initialize a connection between client and server. The client sends this request with its protocol version and capabilities. The server responds with its own protocol version and capabilities.
    """
    pass


def initialized():
    """
    2. **initialized** - A notification sent from client to server as an acknowledgment after initialization is complete.
    """
    pass


## Tool-Related Endpoints
"""
3. **notifications/tools/list_changed** - A notification endpoint that servers use to inform clients when the available tools have changed.
"""


def list_tools():
    """
    1. **tools/list** - An endpoint that allows clients to discover and list all available tools provided by the server.
    """
    pass


def tools_call():
    """
    2. **tools/call** - Used to invoke a specific tool on the server, where the server performs the requested operation and returns results.
    """
    pass


## Resource-Related Endpoints
def list_resources():
    """
    1. **resources/list** - Similar to tools/list, this endpoint allows clients to discover available resources.
    """
    pass


def subscribe_resources():
    """
    2. **resources/get** - Used to retrieve a specific resource from the server.
    """
    pass


def unsubscribe_resources():
    """
    4. **resources/unsubscribe** - Allows clients to unsubscribe from resource updates.
    """
    pass


def notifications_resources_changes():
    """
    5. **notifications/resources/changed** - A notification endpoint that servers use to inform clients when resources have changed.
    """
    pass


## Prompt-Related Endpoints
def list_prompts():
    """
    1. **prompts/list** - Allows clients to discover available prompts.
    """
    pass


def get_prompt():
    """
    2. **prompts/get** - Used to retrieve a specific prompt from the server.
    """
    pass


## Sampling-Related Endpoints
def request_sampling():
    """
    1. **sampling/request** - Allows servers to request that the client perform LLM sampling.
    """
    pass


def return_sampling():
    """
    2. **sampling/result** - Used by clients to return the results of an LLM sampling operation.
    """
    pass


## General Endpoints
def cancel():
    """
    1. **cancel** - Used to cancel an ongoing operation.
    """
    pass


def progress():
    """
    2. **$/progress** - Used to report progress on long-running operations.
    """
    pass


def close():
    """
    3. **close** - Used to terminate the connection cleanly.
    """
    pass
