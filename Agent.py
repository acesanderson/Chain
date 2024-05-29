"""
This is my Chain framework class for Agents, Retrievers, Chatbots, and Tools.
Agent: takes a model and either a tool or a retriever.
Chatbot: takes model retriever, tool, and agent.
"""

from Chain import Model, Prompt, Parser, Chain      # Chain framework
from googleapiclient.discovery import build         # Google search engine
from py_expression_eval import Parser
import re, time
import os
import dotenv
dotenv.load_dotenv()
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")          # for Google search engine
GOOGLE_CSE_KEY = os.getenv("GOOGLE_CSE_KEY")        # for Google search engine
parser = Parser()

class Agent():
    """
    Agent class for the Chain framework.
    """

    def __init__(self, model, retriever=None, tools=None, system_prompt=None):
        self.model = model

        self.system_prompt = system_prompt
        self.tools = tools

class Tool():
    """
    Tool class for the Chain framework.
    """
    
    def __init__(self, tool_function, tool_instruction):
        self.tool_function = tool_function
        self.tool_instruction = tool_instruction
    
    def run(self, input):
        return self.tool_function(input)

class Search(Tool):
    """
    Search class for the Chain framework.
    """

    def search(search_term):
        search_result = ""
        service = build("customsearch", "v1", developerKey=GOOGLE_CSE_KEY)
        res = service.cse().list(q=search_term, cx=GOOGLE_CSE_ID, num = 10).execute()
        for result in res['items']:
            search_result = search_result + result['snippet']
        return search_result
    
    def __init__(self):    
        self.tool_function = self.search
        self.tool_instruction = "Search: Search: useful for when you need to answer questions about current events. You should ask targeted questions."
        super().__init__(self.tool_function, self.tool_instruction)

class Calculator(Tool):
    """
    Calculator class for the Chain framework.
    """

    def __init__(self):
        super().__init__('calculator', Agent.calculator, 'Calculator: Useful for when you need to answer questions about math. Use python code, eg: 2 + 2')
    
    def calculator(str):
        return parser.parse(str).evaluate({})

