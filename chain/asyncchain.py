from Chain.chain.chain import Chain, Prompt
from Chain.model.model import ModelAsync
from Chain.response.response import Response
from Chain.message.message import Message
from Chain.parser.parser import Parser
import asyncio
from typing import overload, Optional


class AsyncChain(Chain):

    def __init__(
        self,
        model: ModelAsync,
        prompt: Prompt | None = None,
        parser: Parser | None = None,
    ):
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
                results = await self._run_prompt_strings(prompt_strings, semaphore)
                return self.convert_results_to_responses(results)
            if input_variables_list:
                results = await self._run_input_variables(
                    input_variables_list, semaphore
                )
                return self.convert_results_to_responses(results)
            return []

        responses = asyncio.run(_run_async())
        return responses

    async def _run_input_variables(
        self,
        input_variables_list: list[dict],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> list:
        if not self.prompt:
            raise ValueError("No prompt assigned to AsyncChain object")

        async def process_with_semaphore(
            input_variables: dict, semaphore: Optional[asyncio.Semaphore]
        ):
            if semaphore:
                # Use a semaphore to limit the number of concurrent requests
                async with semaphore:
                    return await self.model.query_async(
                        input=self.prompt.render(input_variables=input_variables)
                    )
            else:
                return await self.model.query_async(
                    input=self.prompt.render(input_variables=input_variables)
                )

        coroutines = [
            process_with_semaphore(input_variables, semaphore)
            for input_variables in input_variables_list
        ]
        # Need to convert these to Response objects
        return await asyncio.gather(*coroutines, return_exceptions=True)

    async def _run_prompt_strings(
        self, prompt_strings: list[str], semaphore: Optional[asyncio.Semaphore] = None
    ) -> list:

        async def process_with_semaphore(
            prompt_string: str, semaphore: Optional[asyncio.Semaphore]
        ):
            if semaphore:
                # Use a semaphore to limit the number of concurrent requests
                async with semaphore:
                    return await self.model.query_async(input=prompt_string)
            else:
                return await self.model.query_async(input=prompt_string)

        coroutines = [
            process_with_semaphore(prompt_string, semaphore)
            for prompt_string in prompt_strings
        ]
        # Need to convert these to Response objects
        return await asyncio.gather(*coroutines, return_exceptions=False)

    def convert_results_to_responses(self, results: list) -> list[Response]:
        # Convert results to Response objects
        responses = []
        for result in results:
            if isinstance(result, Exception):  # Important! Handle exceptions
                response = Response(
                    content=str(result),  # Or some other error message
                    status="error",
                    prompt=None,
                    model=self.model.model,
                    duration=None,
                    messages=[Message(role="assistant", content=str(result))],
                )
            else:
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
