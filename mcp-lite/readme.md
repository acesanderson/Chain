Purpose of this project:
- lite implemenation of MCP that uses the base primitives but within the context of a python script. No client-server or dealing with protocol/transport layers.
- this should align with MCP spec in two very specific ways:
 - server definition logic (fastMCP style decorators used with a server object to compose the server)
 - host-side logic for exposing capabilities to llm, processing LLM tools/resource calls, and rendering results back to LLM.
