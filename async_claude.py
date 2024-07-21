
# # all our packages
# from jinja2 import Environment, meta, StrictUndefined   # we use jinja2 for prompt templates
# from openai import OpenAI                               # GPT
# import google.generativeai as genai                     # Google's models
# from openai import AsyncOpenAI                          # for async; not implemented yet
# from anthropic import Anthropic                         # claude
# from groq import Groq                                   # groq
# import ollama                                           # local llms
# import re                                               # for regex
# import os                                               # for environment variables
# import dotenv                                           # for loading environment variables
# import itertools                                        # for flattening list of models
# import json                                             # for our jsonparser
# import time                                             # for timing our query calls (saved in Response object)
# import textwrap                                         # to allow for indenting of multiline strings for code readability
# from pydantic import BaseModel, conlist
# from typing import List, Optional, Type, Union          # for type hints
# import instructor                                       # for parsing objects from LLMs
# import asyncio										    # for async
# from ollama import Client               				# for local llms

# # set up our environment: dynamically setting the .env location considered best practice for complex projects.
# dir_path = os.path.dirname(os.path.realpath(__file__))
# # Construct the path to the .env file
# env_path = os.path.join(dir_path, '.env')
# # Load the environment variables
# dotenv.load_dotenv(dotenv_path=env_path)
# api_keys = {}
# api_keys['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
# api_keys['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
# api_keys['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# api_keys['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
# client_openai = instructor.from_openai(OpenAI(api_key = api_keys['OPENAI_API_KEY']))
# client_anthropic = instructor.from_anthropic(Anthropic(api_key = api_keys['ANTHROPIC_API_KEY']))
# genai.configure(api_key=api_keys["GOOGLE_API_KEY"])
# async_client_openai = instructor.from_openai(AsyncOpenAI(api_key = api_keys["OPENAI_API_KEY"]))

# from anthropic import AsyncAnthropic
# async_client_anthropic = instructor.from_anthropic(AsyncAnthropic(api_key = api_keys['ANTHROPIC_API_KEY']))

# from Chain import Model, Prompt, Chain

# # Pydantic objects for validation

# class Message(BaseModel):
# 	"""
# 	Industry standard, more or less, for messaging with LLMs.
# 	System roles can have some weirdness (like with Anthropic), but role/content is standard.
# 	"""
# 	role: str
# 	content: str

# class Messages(BaseModel):
# 	"""
# 	Industry standard, more or less, for messaging with LLMs.
# 	System roles can have some weirdness (like with Anthropic), but role/content is standard.
# 	"""
# 	messages: List[Message]

# def is_messages_object(input):
# 	if not input:
# 		return False
# 	try:
# 		Messages(messages=input)
# 		return True
# 	except Exception as e:
# 		print(e)
# 		return False

# async def query_anthropic_async(prompt: Union[str, list], verbose: bool=True, model: str = "claude-3-5-sonnet-20240620", pydantic_model: Optional[Type[BaseModel]] = None, request_num: int = None) -> Union[BaseModel, str]:
#     """
#     Handles all asynchronous requests from Anthropic's models using Instructor.
#     Possibilities:
#     - pydantic object not provided, input is string -> return string
#     - pydantic object provided, input is string -> return pydantic object
#     """
#     if isinstance(prompt, str):
#         prompt = [{"role": "user", "content": prompt}]
#     elif is_messages_object(prompt):
#         prompt = prompt
#     else:
#         raise ValueError(f"Prompt not recognized as a valid prompt type: {type(prompt)}: {prompt}")
    
#     print(f"Sending request #{request_num} to {model}...") if verbose else None

#     try:
#         if pydantic_model:
#             response = await async_client_anthropic.chat.completions.create(
#                 model=model,
#                 max_tokens=2000,
#                 messages=prompt,
#                 response_model=pydantic_model
#             )
#             return response
#         else:
#             # Use a simple string response model when no Pydantic model is provided
#             class StringResponse(BaseModel):
#                 content: str

#             response = await async_client_anthropic.chat.completions.create(
#                 model=model,
#                 max_tokens=2000,
#                 messages=prompt,
#                 response_model=StringResponse
#             )
#             return response.content
    
#     except Exception as e:
#         print(f"Error in request #{request_num}: {str(e)}")
#         raise

#     finally:
#         print(f"Received response #{request_num} from {model}") if verbose else None


# # async def query_anthropic_async( prompt: Union[str, list], verbose: bool=True, model: str = "claude-3-5-sonnet-20240620", pydantic_model: Optional[Type[BaseModel]] = None, request_num: int = None) -> Union[BaseModel, str]:
# #     """
# #     Handles all asynchronous requests from OpenAI's models.
# #     Possibilities:
# #     - pydantic object not provided, input is string -> return string
# #     - pydantic object provided, input is string -> return pydantic object
# #     """
# #     if isinstance(prompt, str):
# #         prompt = [{"role": "user", "content": prompt}]
# #     elif is_messages_object(prompt):
# #         prompt = prompt
# #     else:
# #         raise ValueError(f"Prompt not recognized as a valid prompt type: {type(prompt)}: {prompt}")
    
# #     print(f"Sending request #{request_num} to {model}...") if verbose else None
# #     # call our client
# #     if pydantic_model:
# #         response = await async_client_anthropic.chat.completions.create(
# #             model = model,
# #             max_tokens = 2000,
# #             response_model = pydantic_model,
# #             messages = prompt
# #         )
# #     else:
# #         response = await async_client_anthropic.chat.completions.create(
# #             model = model,
# #             max_tokens = 2000,
# #             messages = prompt
# #         )
    
# #     print(f"Received response #{request_num} from {model}") if verbose else None
    
# #     if pydantic_model:
# #         return response  # Directly returning the response for pydantic processing
# #     else:
# #         if hasattr(response, 'choices'):
# #             return response.content[0].text
# #         else:
# #             print("Response structure:", response)  # Debugging line to inspect the structure
# #             raise AttributeError("Response object does not have 'content'")

# async def run_multiple_extracts(prompts: list[str], pydantic_model: Optional[Type[BaseModel]] = None, verbose: bool = True):
#     tasks = []
#     for i, p in enumerate(prompts, start = 1):
#         print(f"Preparing task #{i} of {len(prompts)}...") if verbose else None
#         tasks.append(query_anthropic_async(p, pydantic_model = pydantic_model, request_num = i, verbose = verbose))
#     print(f"Running {len(prompts)} tasks concurrently...") if verbose else None
#     start_time = time.time()
#     results = await asyncio.gather(*tasks)  # Run them concurrently
#     end_time = time.time()
#     print(f"All {len(prompts)} tasks completed in {end_time - start_time:.2f} seconds.") if verbose else None
#     return results

# def run_async(prompts: list[str], pydantic_model: Optional[Type[BaseModel]] = None, verbose: bool = True, throttle = 20) -> list[Union[BaseModel, str]]:
#     """
#     Example of how to run multiple extracts asynchronously.
#     """
#     if throttle > 0:
#         if len(prompts) > throttle:
#             raise ValueError("You've requested more than 50 prompts; throwing an error to spare you bank account.")
#     results = asyncio.run(run_multiple_extracts(prompts, pydantic_model = pydantic_model, verbose = verbose))
#     return results


from Chain import Model

if __name__ == "__main__":
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?"
    ]
    model = Model('claude')
    results = model.run_async(prompts = prompts, model = "claude")
    for i, r in enumerate(results, start = 1):
        print(f"Response #{i}: {r}")
    print("All responses received.")
