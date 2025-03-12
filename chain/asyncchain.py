from Chain.chain.chain import Chain, Prompt
from Chain.model.model import ModelAsync
from Chain.response.response import Response
from Chain.message.message import Message
from Chain.parser.parser import Parser
import asyncio
from typing import overload, Optional
from pydantic import BaseModel


class AsyncChain(Chain):

    def __init__(
        self,
        model: ModelAsync,
        prompt: Prompt | None = None,
        parser: Parser | None = None,
    ):
        if not isinstance(model, ModelAsync):
            raise TypeError("Model must be of type ModelAsync")
        if prompt and not isinstance(prompt, Prompt):
            raise TypeError("Prompt must be of type Prompt")
        if parser and not isinstance(parser, Parser):
            raise TypeError("Parser must be of type Parser")
        """Override to use ModelAsync"""
        self.prompt = prompt
        self.model = model
        self.parser = parser
        if self.prompt:
            self.input_schema = self.prompt.input_schema()  # this is a set
        else:
            self.input_schema = set()

    @overload
    def run(self, input_variables_list: list[dict]) -> list[Response]: ...

    @overload
    def run(self, *, prompt_strings: list[str]) -> list[Response]: ...

    def run(
        self,
        input_variables_list: list[dict] | None = None,
        prompt_strings: list[str] | None = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> list[Response]:

        async def _run_async():
            if prompt_strings:
                return await self._run_prompt_strings(prompt_strings, semaphore)
            if input_variables_list:
                return await self._run_input_variables(input_variables_list, semaphore)

        results = asyncio.run(_run_async())
        responses = self.convert_results_to_responses(results)

        return responses

    async def _run_input_variables(
        self,
        input_variables_list: list[dict],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Response:
        if not self.prompt:
            raise ValueError("No prompt assigned to AsyncChain object")
        if self.parser:
            pydantic_model = self.parser.pydantic_model
        else:
            pydantic_model = None

        async def process_with_semaphore(
            input_variables: dict,
            pydantic_model: BaseModel | None,
            semaphore: Optional[asyncio.Semaphore],
        ):
            if semaphore:
                # Use a semaphore to limit the number of concurrent requests
                async with semaphore:
                    return await self.model.query_async(
                        input=self.prompt.render(input_variables=input_variables),
                        pydantic_model=pydantic_model,
                    )
            else:
                return await self.model.query_async(
                    input=self.prompt.render(input_variables=input_variables),
                    pydantic_model=pydantic_model,
                )

        coroutines = [
            process_with_semaphore(input_variables, pydantic_model, semaphore)
            for input_variables in input_variables_list
        ]
        # Need to convert these to Response objects
        return await asyncio.gather(*coroutines, return_exceptions=True)

    async def _run_prompt_strings(
        self, prompt_strings: list[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> Response:
        if self.parser:
            pydantic_model = self.parser.pydantic_model
        else:
            pydantic_model = None

        async def process_with_semaphore(
            prompt_string: str,
            pydantic_model: BaseModel | None,
            semaphore: Optional[asyncio.Semaphore],
        ):
            if semaphore:
                # Use a semaphore to limit the number of concurrent requests
                async with semaphore:
                    return await self.model.query_async(
                        input=prompt_string, pydantic_model=pydantic_model
                    )
            else:
                return await self.model.query_async(
                    input=prompt_string, pydantic_model=pydantic_model
                )

        coroutines = [
            process_with_semaphore(prompt_string, pydantic_model, semaphore)
            for prompt_string in prompt_strings
        ]
        return await asyncio.gather(*coroutines, return_exceptions=True)

    def convert_results_to_responses(self, results: list[str]) -> list[Response]:
        # Convert results to Response objects
        responses = []
        for result in results:
            response = Response(
                content=result,
                status="success",
                prompt=None,  # This would be very hard to calculate; maybe later
                model=self.model.model,
                duration=None,  # This would be very hard to calculate; maybe later
                messages=[Message(role="assistant", content=result)],
            )
            responses.append(response)
        return responses
