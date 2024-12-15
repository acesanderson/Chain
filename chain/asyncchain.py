from Chain.chain.chain import Chain
from Chain.response.response import Response
import asyncio
from typing import overload


class AsyncChain(Chain):
    @overload
    def run_async(self, input_variables_list: list[dict]) -> list[Response]: ...

    @overload
    def run_async(self, *, prompt_strings: list[str]) -> list[Response]: ...

    def run_async(
        self,
        input_variables_list: list[dict] | None = None,
        prompt_strings: list[str] | None = None,
    ) -> list[Response]:
        async def _run_async():
            if prompt_strings:
                return await self._run_prompt_strings(prompt_strings)
            if input_variables_list:
                return await self._run_input_variables(input_variables_list)

        results = asyncio.run(_run_async())
        return results

    async def _run_input_variables(self, input_variables_list: list[dict]) -> Response:
        coroutines = [
            self.model.query_async(prompt, input_variables)
            for input_variables in input_variables_list
        ]
        # Need to convert these to Response objects
        return await asyncio.gather(*coroutines)

    async def _run_prompt_strings(self, prompt_strings: list[str]) -> Response:
        coroutines = [
            self.model.query_async(prompt_string) for prompt_string in prompt_strings
        ]
        # Need to convert these to Response objects
        return await asyncio.gather(*coroutines)

    def batch(self):
        pass
