import openai
import instructor
from pydantic import BaseModel
import asyncio

async_client = instructor.from_openai(openai.AsyncOpenAI())

count = 1

class User(BaseModel):
    name: str
    age: int

async def extract():
    global count
    print(f"kicking off extract #{count}")
    count += 1
    return await async_client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=User,
    )

async def run_multiple_extracts():
    tasks = [extract() for _ in range(10)]  # Create a list of 10 extract() tasks
    results = await asyncio.gather(*tasks)  # Run them concurrently
    return results
