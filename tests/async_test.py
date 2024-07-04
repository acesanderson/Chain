# import openai
# import instructor
# from pydantic import BaseModel
# import asyncio
# from typing import List, Union, Optional, Type

# async_client_openai = instructor.from_openai(openai.AsyncOpenAI())

# count = 1

# class User(BaseModel):
#     name: str
#     age: int

# async def extract():
#     global count
#     print(f"kicking off extract #{count}")
#     count += 1
#     return await async_client.chat.completions.create(
#         model="gpt-3.5-turbo-0125",
#         messages=[
#             {"role": "user", "content": "Create a user"},
#         ],
#         response_model=User,
#     )

# async def run_multiple_extracts():
#     tasks = [extract() for _ in range(10)]  # Create a list of 10 extract() tasks
#     results = await asyncio.gather(*tasks)  # Run them concurrently
#     return results

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


# async def query_openai_async(self, input: Union[str, list], verbose: bool=True, model: str = "gpt-3.5-turbo-0125", pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:
#     """
#     Handles all asynchronous requests from OpenAI's models.
#     Possibilities:
#     - pydantic object not provided, input is string -> return string
#     - pydantic object provided, input is string -> return pydantic object
#     """
#     if isinstance(input, str):
#         input = [{"role": "user", "content": input}]
#     elif is_messages_object(input):
#         input = input
#     else:
#         raise ValueError(f"Input not recognized as a valid input type: {type(input)}: {input}")
#     # call our client
#     response = async_client_openai.chat.completions.create(
#         model = model,
#         response_model = pydantic_model,
#         messages = input
#     )
#     if pydantic_model:
#         return await response
#     else:
#         return await response.choices[0].message.content


from Chain import Prompt, Model
from pydantic import BaseModel
from typing import List

class Objects(BaseModel):
	objects: List[str]

prompts = ['birds', 'mammals', 'presidents', 'planets', 'countries', 'cities']
prompt_template = Prompt("Name ten {{objects}}.")
prompts = [prompt_template.render(input_variables = {'objects': prompt}) for prompt in prompts]
model = Model('gpt3')
results = model.run_async(prompts, pydantic_model = Objects)

