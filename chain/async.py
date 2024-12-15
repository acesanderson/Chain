"""
Vanilla implementation of Chain to understand what features I should add to package.
"""

from Chain import Prompt, Model, Chain

# Base use case: "name ten {{things}}"; a list of things

list_of_things = [
    "frogs",
    "buildings",
    "ukrainian poets",
    "careers in dentistry",
    "poetic themes",
    "Apple products",
    "dead people",
    "shocking films",
    "non-American cars",
]

prompt = Prompt("Name five {{things}}.")
model = Model("gpt-4o-mini")
chain = Chain(model, prompt)

# Serial example; works as expected
# for things in list_of_things:
#     response = chain.run(input_variables={"things": things})
#     print(response)

# Async: I need an async version of Chain.run().
# Chain.run has a lot of overheads, so let's start with the Model class, with query_async.
# But before that, we need to implement the async version of the OpenAIClient class.
# But before that, we should understand how Instructor library handles async openai.

# sync with async instructor
import openai
import instructor
import asyncio
from pprint import pprint

client = instructor.from_openai(openai.AsyncOpenAI())


async def query(input: str):
    print(f"Querying {input}.")
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": input},
        ],
        response_model=None,
    )


# async def main():
#     response = await query("Name five frogs.")
#     print(response.choices[0].message.content)


# asyncio.run(main())

# async with async instructor

things = ["Name ten " + thing + "." for thing in list_of_things]


async def main(things):
    coroutines = [query(thing) for thing in things]
    responses = await asyncio.gather(*coroutines)
    pprint(responses)


if __name__ == "__main__":
    asyncio.run(main(things))

# Takeaways: need to batch queries to avoid rate limiting.

# what do I want?
# pass:
# - a list of prompt strings (these are turned to Prompt objects)
# - a list of Prompt objects
# - a list of input_variables (these are passed to the Prompt objects)
"""
Async_Chain inherits from Chain and adds the async_run method.
This will be in a separate script which imports Chain, Model, Prompt, Response.
Model.query_async is called instead of Model.query (and these are all implemented down to the client classes).
"""
import asyncio
from typing import overload
from Chain import Chain, Prompt, Response
import instructor
import openai


class Async_Chain(Chain):

    @overload
    async def run_async(self, input_variables: list[dict]) -> list[Response]: ...

    @overload
    async def run_async(self, *, prompt_strings: list[str]) -> list[Response]: ...

    async def run_async(
        self,
        input_variables_list: list[dict] | None = None,
        prompt_strings: list[str] | None = None,
    ) -> list[Response]:
        # This routes the call to the appropriate method
        if prompt_strings:
            return await self._run_prompt_strings(prompt_strings)
        if input_variables_list:
            return await self._run_input_variables(input_variables_list)

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


# The above code is a rough sketch of what I want to implement, now we return to the OpenAIClient class.
# First, sketch out the type of call I want to make -- this will go in the associated pytest script.
