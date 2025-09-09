"""
WIP: this should leverage SiphonClient for sending requests to a Chain/SiphonServer.
"""

from Chain.model.model import Model
from Chain.model.models.modelstore import ModelStore
from Chain.progress.wrappers import progress_display
from Chain.progress.verbosity import Verbosity


class ModelClient(Model):
    """
    This class is used to send requests to a Chain/SiphonServer.
    """

    raise NotImplementedError(
        "ModelClient is an abstract class and should not be instantiated directly. "
        "Use a specific client implementation like OpenAIClient or AnthropicClient."
    )

    def __init__(self, model_name: str):
        """
        Initializes the ModelClient with the given model name.

        :param model_name: The name of the model to be used.
        """
        self.model = ModelStore._validate_model(model)

    @progress_display
    def query(
        self,
        # Standard parameters
        query_input: str | list | Message | None = None,
        response_model: type["BaseModel"] | None = None,
        cache=True,
        temperature: Optional[float] = None,
        stream: bool = False,
        output_type: OutputType = "text",
        max_tokens: Optional[int] = None,
        # For progress reporting decorator
        verbose: Verbosity = Verbosity.PROGRESS,
        index: int = 0,
        total: int = 0,
        # If we're hand-constructing Request params, we can pass them in directly
        request: Optional[Request] = None,
        # Options for debugging
        return_request: bool = False,
        return_error: bool = False,
    ) -> "ChainResult | Request | Stream | AnthropicStream":
        try:
            # Construct Request object if not provided (majority of cases)
            if not request:
                logger.info(
                    "Constructing Request object from query_input and other parameters."
                )
                import inspect

                frame = inspect.currentframe()
                args, _, _, values = inspect.getargvalues(frame)

                query_args = {k: values[k] for k in args if k != "self"}
                query_args["model"] = self.model
                cache = query_args.pop("cache", False)
                if query_input:
                    query_args.pop("query_input", None)
                    request = Request.from_query_input(
                        query_input=query_input, **query_args
                    )
                else:
                    request = Request(**query_args)

            assert isinstance(request, Request), (
                f"Request must be an instance of Request or None, got {type(request)}"
            )

            # For debug, return Request if requested
            if return_request:
                return request
            # For debug, return error if requested
            if return_error:
                from Chain.tests.fixtures import sample_error

                return sample_error

            # Check cache first
            logger.info("Checking cache for existing results.")
            if cache and self._chain_cache:
                cached_result = self._chain_cache.check_for_model(request)
                if isinstance(cached_result, ChainResult):
                    return (
                        cached_result  # This should be a Response (part of ChainResult)
                    )
                elif cached_result == None:
                    logger.info("No cached result found, proceeding with query.")
                    pass
                elif cached_result and not isinstance(cached_result, ChainResult):
                    logger.error(
                        f"Cache returned a non-ChainResult type: {type(cached_result)}. Ensure the cache is properly configured."
                    )
                    raise ValueError(
                        f"Cache returned a non-ChainResult type: {type(cached_result)}. Ensure the cache is properly configured."
                    )
            # Execute the query
            logger.info("Executing query with client.")
            start_time = time()
            result, usage = self._client.query(request)
            stop_time = time()
            logger.info(f"Query executed in {stop_time - start_time:.2f} seconds.")

            # Handle streaming responses
            from Chain.model.clients.openai_client import Stream
            from Chain.model.clients.anthropic_client import Stream as AnthropicStream

            if isinstance(result, Stream) or isinstance(result, AnthropicStream):
                if stream:
                    logger.info("Returning streaming response.")
                    return result  # Return stream directly
                else:
                    logger.error(
                        "Streaming responses are not supported in this method. "
                        "Set stream=True to receive streamed responses."
                    )
                    raise ValueError(
                        "Streaming responses are not supported in this method. "
                        "Set stream=True to receive streamed responses."
                    )

            # Construct Response object
            from Chain.result.response import Response
            from pydantic import BaseModel

            if isinstance(result, Response):
                logger.info("Returning existing Response object.")
                response = result
            elif isinstance(result, str) or isinstance(result, BaseModel):
                logger.info(
                    "Constructing Response object from result string or BaseModel."
                )
                # Construct relevant Message type per requested output_type
                match output_type:
                    case "text":  # result is a string
                        from Chain.message.textmessage import TextMessage

                        assistant_message = TextMessage(
                            role="assistant", content=result
                        )
                    case "image":  # result is a base64 string of an image
                        assert isinstance(result, str), (
                            "Image generation request should not return a BaseModel; we need base64 string."
                        )
                        from Chain.message.imagemessage import ImageMessage

                        assistant_message = ImageMessage.from_base64(
                            role="assistant", text_content="", image_content=result
                        )
                    case "audio":  # result is a base64 string of an audio
                        assert isinstance(result, str), (
                            "Audio generation (TTS) request should not return a BaseModel; we need base64 string."
                        )
                        from Chain.message.audiomessage import AudioMessage

                        assistant_message = AudioMessage.from_base64(
                            role="assistant",
                            audio_content=result,
                            text_content="",
                            format="mp3",
                        )

                response = Response(
                    message=assistant_message,
                    request=request,
                    duration=stop_time - start_time,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            else:
                logger.error(
                    f"Unexpected result type: {type(result)}. Expected Response or str."
                )
                raise TypeError(
                    f"Unexpected result type: {type(result)}. Expected Response or str."
                )

            # Update cache after successful query
            logger.info("Updating cache with the new response.")
            if cache and self._chain_cache:
                self._chain_cache.store_for_model(request, response)

            return response  # Return Response (part of ChainResult)

        except ValidationError as e:
            chainerror = ChainError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Validation error: {chainerror}")
            return chainerror
        except Exception as e:
            chainerror = ChainError.from_exception(
                e,
                code="query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Error during query: {chainerror}")
            return chainerror
