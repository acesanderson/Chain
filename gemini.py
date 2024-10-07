


# # def query_google(self, input: Union[str, list], verbose: bool=True, model: str = 'mistral:latest', pydantic_model: Optional[Type[BaseModel]] = None) -> Union[BaseModel, str]:

# import instructor
# import google.generativeai as genai
# from pydantic import BaseModel
# import os
# import dotenv                                           # for loading environment variables
# dotenv.load_dotenv()

# class User(BaseModel):
#     name: str
#     age: int

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # alternative API key configuration

# client = instructor.from_gemini(
#     client=genai.GenerativeModel(
#         model_name="models/gemini-1.5-flash-latest",  # model defaults to "gemini-pro"
#     ),
#     mode=instructor.Mode.GEMINI_JSON,
# )

# resp = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Extract Jason is 25 years old.",
#         }
#     ],
#     response_model=User,
# )


# # model = "gemini-1.5-pro-latest"
# input = "name ten mammals"
# gemini_model = genai.GenerativeModel(model_name = "gemini-1.5-pro-latest")
# response = gemini_model.generate_content(input)
# resp = response.candidates[0].content.parts[0].text
# print(resp)

from pydantic import BaseModel
from Chain import Model

class User(BaseModel):
    name: str
    age: int

print("String mode")
m=Model('gemini-1.5-pro-latest')
response = m.query("Name ten mammals.")
print(response)

print("Pydantic mode")
response = m.query("Name ten mammals.", pydantic_model=User)

print(response)