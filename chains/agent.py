"""
Following this tutorial: https://gathnex.medium.com/how-to-create-your-own-llm-agent-from-scratch-a-step-by-step-guide-14b763e5b3b8

This works; using this to inform my Agent.py class for Chain framework.
"""


"""
<script async src="https://cse.google.com/cse.js?cx=e05e3a4278c8f477c">
</script>
<div class="gcse-search"></div>
"""

from openai import OpenAI
from googleapiclient.discovery import build
from py_expression_eval import Parser
import re, time, os
import os
import dotenv
dotenv.load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")

"""
## Tools

Search: Conduct a search engine query using Google’s official custom search engine. You can send 100 requests per day in the free tier.

Calculator: I use py_expression_eval as a calculator (good balance between being able to run complex math expressions, without many of
the risks of trying to pull in a full Python REPL/eval).
"""

# Google search engine
def search(search_term):
    search_result = ""
    service = build("customsearch", "v1", developerKey=GOOGLE_CSE_KEY)
    res = service.cse().list(q=search_term, cx=GOOGLE_CSE_ID, num = 10).execute()
    for result in res['items']:
        search_result = search_result + result['snippet']
    return search_result

# Calculator
parser = Parser()
def calculator(str):
    return parser.parse(str).evaluate({})

System_prompt = """
Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

Search: Search: useful for when you need to answer questions about current events. You should ask targeted questions.
Calculator: Useful for when you need to answer questions about math. Use python code, eg: 2 + 2
Response To Human: When you need to respond to the human you are talking to.

You will receive a message from the human, then you should start a loop and do one of two things

Option 1: You use a tool to answer the question.
For this, you should use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: "the input to the action, to be sent to the tool"

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: "your response to the human, summarizing what you did and what you learned"

Begin!
"""

"""
So what the above loop can do ?

Telling the model that it will be run in the loop. In that loop, the LLM has two options: it can either ‘use a tool,’ giving that tool an input, 
or it can respond to the human. We give the model a list of the tools and a description of when/how to use each one. The thought-action pattern
creates a ‘chain of thought,’ telling the model to think about what it’s doing.

We used the GPT-4 model for this demo, but you can also use open source models like Llama, Mistral, Zephyr, etc.
"""

def Stream_agent(prompt):
    messages = [
        { "role": "system", "content": System_prompt },
        { "role": "user", "content": prompt },
    ]
    def extract_action_and_input(text):
          action_pattern = r"Action: (.+?)\n"
          input_pattern = r"Action Input: \"(.+?)\""
          action = re.findall(action_pattern, text)
          action_input = re.findall(input_pattern, text)
          return action, action_input
    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            top_p=1,)
        response_text = response.choices[0].message.content
        print(response_text)
        # To prevent the Rate Limit error for free-tier users, we need to decrease the number of requests/minute.
        time.sleep(10)
        action, action_input = extract_action_and_input(response_text)
        if action[-1] == "Search":
            tool = search
        elif action[-1] == "Calculator":
            tool = calculator
        elif action[-1] == "Response To Human":
            print(f"Response: {action_input[-1]}")
            break
        observation = tool(action_input[-1])
        print("Observation: ", observation)
        messages.extend([
            { "role": "system", "content": response_text },
            { "role": "user", "content": f"Observation: {observation}" },
            ])
