from Chain.api.client.ChainClient import ChainClient, ChainRequest, get_url
from Chain.model.model import Model


class ModelClient(Model):
    """
    Model for interacting with a ChainServer.
    Primary use case: ollama models using my desktop with RTX 5090.
    """

    _client = ChainClient()

    def __init__(
        self, url: str = get_url()
    ):  # get_url is me defaulting to my own server
        """
        Initialize the ModelClient with the ChainClient.
        """
        super().__init__()
        self.model = "chain"
        self._client_type = self._get_client_type(self.model)
        self._client = self.__class__._get_client(self._client_type)

    def query(
        self,
        input: str | list | Message | ImageMessage | AudioMessage,
        verbose: bool = True,
        pydantic_model: BaseModel | None = None,
        raw=False,
        cache=True,
        temperature: Optional[float] = None,  # None means just use the defaults
    ) -> BaseModel | str:
        if verbose:
            print(
                f"Model: {self.model}  Temperature: {temperature}  Query: "
                + self.pretty(str(input))
            )
        if Model._chain_cache and cache:
            cached_request = Model._chain_cache.cache_lookup(input, self.model)
            if cached_request:
                print("Cache hit!")
                if pydantic_model:
                    try:
                        cached_request_dict = json.loads(cached_request)
                        obj = pydantic_model(**cached_request_dict)  # type: ignore
                        if raw:
                            return (obj, cached_request)  # type: ignore
                        if not raw:
                            return obj
                    except Exception as e:
                        print(f"Failed to parse cached request: {e}")
                return cached_request
        if pydantic_model == None:
            llm_output = self._client.query(
                self.model, input, raw=False, temperature=temperature
            )
        else:
            obj, llm_output = self._client.query(
                self.model, input, pydantic_model, raw=True
            )
        if Model._chain_cache and cache:
            cached_request = CachedRequest(
                user_input=input, model=self.model, llm_output=llm_output
            )
            Model._chain_cache.insert_cached_request(cached_request)
        if pydantic_model and not raw:
            return obj  # type: ignore
        elif pydantic_model and raw:
            return obj, llm_output  # type: ignore
        else:
            return llm_output
