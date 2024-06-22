"""
This is me running my own framework, called Chain.
A link is an object that takes the following:
- a prompt (a jinja2 template)
- a model (a string)
- an output (a dictionary)

Next up:
- incorporate Instructor for object parsing -- replaces most of Parser class
x - define input_schema (created backwards from jinja template (using find_variables on the original string))
x - allow user to edit input_schema
x - define output_schema (default is just "result", but user can define this)
x - add batch function
x - do more validation
 x - should throw an error if input is not a dictionary with the right schema
x - edit link.run so that it could take a single string if needed (just turn it into dict in the method)
x - eidt link.__init__ so that you can just enter a string to initialize as well
    i.e. instead of topic_chain = Chain(Prompt(topic_prompt)), can you just have Chain(topic_prompt)
    this would enable fast iteration
x - add default parsers to Parser class
x - add gemini models
x - add groq
x - handle messages
- add regex parser (takes a pattern)
- allow temperature setting for Model
- Base class is serial, there will be a parallel extension that leverages async
- a way to chain these together with pipes
- add an 'empty' model that just returns the input (converting dicts to strings), for tracing purposes
 - similarly, adding a "tracing" flag that logs all inputs and outputs throughout the process
- consider other format types like [langchain's](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/output_parsers/format_instructions.py)
"""

# all our packages
from jinja2 import Environment, meta, StrictUndefined   # we use jinja2 for prompt templates
from openai import OpenAI                               # GPT
import google.generativeai as client_gemini             # Google's models
from openai import AsyncOpenAI                          # for async; not implemented yet
from anthropic import Anthropic                         # claude
from groq import Groq                                   # groq
import ollama                                           # local llms
import re                                               # for regex
import os                                               # for environment variables
import dotenv                                           # for loading environment variables
import itertools                                        # for flattening list of models
import json                                             # for our jsonparser
import ast                                              # for our list parser ("eval" is too dangerous)
import time                                             # for timing our query calls (saved in Response object)
import textwrap                                         # to allow for indenting of multiline strings for code readability

# set up our environment: dynamically setting the .env location considered best practice for complex projects.
dir_path = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the .env file
env_path = os.path.join(dir_path, '.env')
# Load the environment variables
dotenv.load_dotenv(dotenv_path=env_path)
api_keys = {}
api_keys['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
api_keys['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
api_keys['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
api_keys['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
# dotenv.load_dotenv()
client_openai = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
client_anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
client_openai_async = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_gemini.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
env = Environment(undefined=StrictUndefined)            # # set jinja2 to throw errors if a variable is undefined

class Chain():
    """
    How we chain things together.
    Instantiate with:
    - a prompt (a string that is ready for jinja2 formatting),
    - a model (a name of a model (full list of accessible models in Model.models))
    - a parser (a function that takes a string and returns a string)
    Defaults to mistral for model, and empty parser.
    """
    # Put API keys for convenience across my system.
    api_keys = api_keys
    # Canonical source of models; or if there are new cloud models (fex. Gemini)
    models = {
        "ollama": [m['name'] for m in ollama.list()['models']],
        "openai": ["gpt-4o","gpt-4-turbo","gpt-3.5-turbo-0125"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"],
        "google": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro"],
        "groq": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        "testing": ["polonius"]
    }
    # Silly examples for testing; if you declare a Chain() without inputs, these are the defaults.
    examples = {
        'batch_example': [{'input': 'John Henry'}, {'input': 'Paul Bunyan'}, {'input': 'Babe the Blue Ox'}, {'input': 'Brer Rabbit'}],
        'run_example': {'input': 'John Henry'},
        'model_example': 'mistral:latest',
        'parser_example': lambda x: x,
        'prompt_example': 'sing a song about {{input}}. Keep it under 200 characters.',
        'system_prompt_example': "You're a helpful assistant.",
    }
    
    def update_models():
        """
        If you need to update the ollama model list on the fly, use this function.
        """
        models = [m['name'] for m in ollama.list()['models']]
        Chain.models['ollama'] = models
    
    def standard_repr(object):
        """
        Standard for all of my classes; changes how the object is represented when invoked in interpreter.
        Called from all classes related to Chain project (Model, Prompt, Chat, etc.).
        """
        attributes = ', '.join([f'{k}={repr(v)[:50]}' for k, v in object.__dict__.items()])
        return f"{object.__class__.__name__}({attributes})"
        # Example output: Chain(prompt=Prompt(string='tell me about {{topic}}', format_in, model=Model(model='mistral'), parser=Parser(parser=<function Chain.<lambda> at 0x7f7c5a
    
    def find_variables(self, template):    
        """
        This function takes a jinja2 template and returns a set of variables; used for setting input_schema of Chain object.
        """
        throwaway_env = Environment()
        parsed_content = throwaway_env.parse(template)
        variables = meta.find_undeclared_variables(parsed_content)
        return variables
    
    def __init__(self, prompt=None, model=None, parser=None):
        if prompt is None:              # if inputs are empty, use the defaults from Model.examples
            prompt = Prompt(Chain.examples['prompt_example'])
        elif isinstance(prompt, str):
            prompt = Prompt(prompt)     # if prompt is a string, turn it into a Prompt object <-- for fast iteration
        if model is None:
            model = Model(Chain.examples['model_example'])
        if parser is None:
            parser = Parser(Chain.examples['parser_example'])
        self.prompt = prompt
        self.model = model
        self.parser = parser
        # Now a little magic within and between the objects
        ## Set up input_schema and output_schema
        self.input_schema = self.find_variables(self.prompt.string)  # this is a set
        self.output_schema = {'result'}                         # in the future, we'll allow users to define this, for chaining purposes
        ## Now add our format instructions from our parser to the prompt
        self.prompt.format_instructions = self.parser.format_instructions
    
    def __repr__(self):
        return Chain.standard_repr(self)
    
    def create_messages(system_prompt = examples['system_prompt_example'], input = None) -> list[dict]:
        """
        Takes a system prompt object (Prompt()) or a string, an optional input object, and returns a list of messages.
        """
        if isinstance(system_prompt, str):
            system_prompt = Prompt(system_prompt)
        if input:
            messages = [{'role': 'system', 'content': system_prompt.render(input=input)}]
        else:
            messages = [{'role': 'system', 'content': system_prompt.string}]
        return messages
    
    def run(self, input=None, parsed=True, verbose=True, messages = None):
        """
        Input should be a dict with named variables that match the prompt.
        Chains are parsed by default, but you can turn this off if you want to see the raw output for debugging purposes.
        """
        if messages:
            return self.run_messages(input=input, messages=messages, parsed=parsed, verbose=verbose)
        if input is None:
            input = Chain.examples['run_example']
        # allow users to just put in one string if the prompt is simple <-- for fast iteration
        if isinstance(input, str) and len(self.input_schema) == 1:
            input = {list(self.input_schema)[0]: input}
        prompt = self.prompt.render(input=input)
        time_start = time.time()
        result = self.model.query(prompt, verbose=verbose)
        time_end = time.time()
        duration = time_end - time_start
        if parsed:
            result = self.parser.parse(result)
        # Return a response object
        response = Response(content=result, status="success", prompt=prompt, model=self.model.model, duration=duration, variables = input)
        return response
    
    def run_messages(self, input=None, messages = [], parsed=True, verbose=True):
        """
        Special version of Chain.run that takes a messages object.
        Converts input + prompt into a message object, appends to messages list, and runs to Model.chat.
        Input should be a dict with named variables that match the prompt.
        Chains are parsed by default, but you can turn this off if you want to see the raw output for debugging purposes.
        """
        # allow users to just put in one string if the prompt is simple <-- for fast iteration
        if isinstance(input, str) and len(self.input_schema) == 1:
            input = {list(self.input_schema)[0]: input}
        # Construct user message
        ## if input is None, just use the prompt, don't render.
        if input is None:
            prompt = self.prompt.string
        else:
            prompt = self.prompt.render(input=input)
        user_message = {'role': 'user', 'content': prompt}
        messages.append(user_message)
        # Run our query
        time_start = time.time()
        result = self.model.chat(messages, verbose=verbose)
        time_end = time.time()
        duration = time_end - time_start
        if parsed:              # This will be a source of many bugs I suspect!
            result = self.parser.parse(result)
        # Return a response object
        assistant_message = {'role': 'assistant', 'content': result}
        messages.append(assistant_message)
        response = Response(content=result, status="success", prompt=prompt, model=self.model.model, duration=duration, messages = messages, variables=input)
        return response
    
    def batch(self, input_list=[]):
        """
        Input list is a list of dictionaries.
        """
        if input_list == []:
            input_list = Chain.examples['batch_example']
        batch_output = []
        for input in input_list:
            print(f"Running batch with input: {input}")
            batch_output.append(self.run(input))
        return batch_output

class Prompt():
    """
    Generates a prompt.
    Takes a jinja2 ready string (note: not an actual Template object; that's created by the class).
    """
    
    def __init__(self, template = Chain.examples['prompt_example']):
        self.string = template
        self.format_instructions = "" # This starts as empty; gets changed if the Chain object has a parser with format_instruction
        self.template = env.from_string(template)
    
    def __repr__(self):
        return Chain.standard_repr(self)
    
    def render(self, input):
        """
        takes a dictionary of variables
        """
        rendered = self.template.render(**input)    # this takes all named variables from the dictionary we pass to this.
        rendered += self.format_instructions
        return rendered

class Model():
    """
    Our basic model class.
    Instantiate with a model name; you can find full list at Model.models.
    This routes to either OpenAI, Anthropic, Google, or Ollama models, in future will have groq.
    There's also an async method which we haven't connected yet (see gpt_async below).
    """
    
    def is_message(self, obj):
        """
        This check is a particular input is a Message type (list of dicts).
        Primarily for routing from query to chat.
        Return True if it is a list of dicts, False otherwise.
        """
        if not isinstance(obj, list):
            return False
        return all(isinstance(x, dict) for x in obj)
    
    def __init__(self, model=Chain.examples['model_example']):
        """
        Given that gpt and claude model names are very verbose, let users just ask for claude or gpt.
        """
        # set up a default query for testing purposes
        self.example_query = Prompt(Chain.examples['prompt_example']).render(Chain.examples['run_example'])
        # initialize model
        if model == 'claude':
            self.model = 'claude-3-5-sonnet-20240620'                               # newest claude model as of 6/21/2024
        elif model == 'gpt':
            self.model = 'gpt-4o'                                                   # defaulting to the cheap strong model they just announced
        elif model == 'gemini':
            self.model = 'gemini-pro'                                               # defaulting to the pro (1.0 )model
        elif model == 'groq':
            self.model = 'mixtral-8x7b-32768'                                       # defaulting to the mixtral model
        elif model in list(itertools.chain.from_iterable(Chain.models.values())):   # any other model we support (flattened the list)
            self.model = model
        else:
            raise ValueError(f"Model not found: {model}")
    
    def __repr__(self):
        return Chain.standard_repr(self)
    
    def query(self, user_input = None, verbose=True):
        """
        Sorts model to either cloud-based (gpt, claude), ollama, or returns an error.
        """
        if user_input is None:
            user_input = self.example_query
        if verbose:
            print(f"{self.model}: {self.pretty(user_input)}")
        if self.is_message(user_input):              # if this is a message, we use chat function instead of query.
            response = self.chat(user_input)
        elif self.model in Chain.models['openai']:
            response = self.query_gpt(user_input)
        elif self.model in Chain.models['anthropic']:
            response = self.query_claude(user_input)
        elif self.model in Chain.models['ollama']:
            response = self.query_ollama(user_input)
        elif self.model in Chain.models['google']:
            response = self.query_gemini(user_input)
        elif self.model in Chain.models['groq']:
            response = self.query_groq(user_input)
        elif self.model == 'polonius':
            response = self.query_polonius(user_input)
        else:
            response = f"Model not found: {self.model}"
        return response
    
    def pretty(self, user_input):
        """
        Truncate input to 150 characters for pretty logging.
        """
        pretty = re.sub(r'\n|\t', '', user_input).strip()
        return pretty[:150]
    
    def query_ollama(self, user_input):
        """
        Queries local models.
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                'role': 'user',
                'content': user_input,
                },
            ]
        )
        return response['message']['content']
    
    def query_gpt(self, user_input):
        """
        Queries OpenAI models.
        There's a parallel function for async (gpt_async)
        """
        response = client_openai.chat.completions.create(
            model = self.model,
            messages = [{"role": "user", "content": user_input}]
        )
        return response.choices[0].message.content
    
    async def query_gpt_async(self, user_input):
        """
        Async version of gpt call; wrap the function call in asyncio.run()
        """
        response = await client_openai_async.chat.completions.create(
            model = self.model,
            messages = [{"role": "user", "content": user_input}]
        )
        return response.choices[0].message.content
    
    def query_claude(self, user_input):
        """
        Queries anthropic models.
        """
        response = client_anthropic.messages.create(
            max_tokens=1024,
            model = self.model,
            messages = [{"role": "user", "content": user_input}]
        )
        return response.content[0].text
    
    def query_gemini(self, user_input):
        """
        Queries Google's models.
        """
        gemini_model = client_gemini.GenerativeModel(self.model)
        response = gemini_model.generate_content(user_input)
        return response.candidates[0].content.parts[0].text
    
    def query_groq(self, user_input):
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            model = self.model,
            )
        return chat_completion.choices[0].message.content
    
    def query_polonius(self, user_input):
        """
        Fake model for testing purposes.
        """
        _ = user_input
        response = textwrap.dedent("""\
            My liege, and madam, to expostulate /
            What majesty should be, what duty is, / 
            Why day is day, night night, and time is time, / 
            Were nothing but to waste night, day and time. / 
            herefore, since brevity is the soul of wit, / And tediousness the limbs and outward flourishes, /
            I will be brief: your noble son is mad: /
            Mad call I it; for, to define true madness, /
            What is't but to be nothing else but mad? / But let that go.
            """).strip()
        return response
    
    def chat(self, messages, verbose = True):
        """
        Handle messages (these are lists of dicts).
        Sorts model to either cloud-based (gpt, claude), ollama, or returns an error.
        """
        if self.model in Chain.models['openai']:
            return self.chat_gpt(messages, verbose)
        elif self.model in Chain.models['anthropic']:
            return self.chat_claude(messages, verbose)
        elif self.model in Chain.models['ollama']:
            return self.chat_ollama(messages, verbose)
        elif self.model in Chain.models['google']:
            return self.chat_gemini(messages, verbose)
        elif self.model in Chain.models['groq']:
            return self.chat_groq(messages, verbose)
        elif self.model == 'polonius':
            return self.query_polonius(messages, verbose)
        else:
            return f"Model not found: {self.model}"
    
    def chat_gpt(self, messages, verbose = True):
        """
        Chat with OpenAI models.
        """
        if verbose:
            print(f"{self.model}: {self.pretty(messages[-1]['content'])}")
        response = client_openai.chat.completions.create(
            model = self.model,
            messages = messages
        )
        return response.choices[0].message.content
    
    def chat_ollama(self, messages, verbose = True):
        """
        Queries local models.
        """
        print('messages ------')
        print(messages)
        if verbose:
            print(f"{self.model}: {self.pretty(messages[-1]['content'])}")
        response = ollama.chat(
            model=self.model,
            messages=messages
        )
        return response['message']['content']
    
    def chat_claude(self, messages, verbose = True):
        """
        Queries anthropic models.
        Claude doesn't accept system messages, so we have to remove it and pass it as a special system parameter.
        """
        if verbose:
            print(f"{self.model}: {self.pretty(messages[-1]['content'])}")
        # Capture and remove the system prompt so we can use Claude's API format properly.
        if messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            messages = messages[1:]
        # Passing the system prompt in our response (this is unique to Anthropic)
        response = client_anthropic.messages.create(
            max_tokens=1024,
            model = self.model,
            system = system_prompt,
            messages = messages
        )
        return response.content[0].text
    
    def chat_gemini(self, messages, verbose = True):
        """
        Queries Google's models.
        """
        if verbose:
            print(f"{self.model}: {self.pretty(messages[-1]['content'])}")
        # Gemini uses a different schema than Claude or OpenAI, annoyingly.
        for message in messages:
        # Change 'assistant' to 'model' if the role is 'assistant'
            if message['role'] == 'assistant':
                message['role'] = 'model'
            # Replace 'content' key with 'parts' and put the value inside a list
            if 'content' in message:
                message['parts'] = [message.pop('content')]
        gemini_model = client_gemini.GenerativeModel(self.model)
        response = gemini_model.generate_content(messages)
        return response.candidates[0].content.parts[0].text
    
    def chat_groq(self, messages, verbose = True):
        if verbose:
            print(f"{self.model}: {self.pretty(messages[-1]['content'])}")
        chat_completion = client_groq.chat.completions.create(
            messages=messages,
            model = self.model,
            )
        return chat_completion.choices[0].message.content
    
    def chat_polonius(self, messages, verbose = True):
        """
        Fake model for testing purposes
        """
        if verbose:
            print(f"{self.model}: {self.pretty(messages[-1]['content'])}")
        _ = messages
        response = textwrap.dedent("""\
            My liege, and madam, to expostulate /
            What majesty should be, what duty is, / 
            Why day is day, night night, and time is time, / 
            Were nothing but to waste night, day and time. / 
            herefore, since brevity is the soul of wit, / And tediousness the limbs and outward flourishes, /
            I will be brief: your noble son is mad: /
            Mad call I it; for, to define true madness, /
            What is't but to be nothing else but mad? / But let that go.
            """).strip()
        return response

class Parser():
    """
    Our basic parser class.
    Takes a function and applies it to output.
    At its most basic, it just validates the output.
    For more sophisticated uses (like json), it will also convert the output.
    It also appends format instructions to the prompt.
    """
    def string_parser(output):
        """
        Validates whether output is a string.
        """
        print(output)
        if isinstance(output, str):
            return input
        else:
            raise TypeError("Expected a string, but got a different type")
    
    def json_parser(output):
        """
        Converts string to json object.
        """
        try:
            return json.loads(output.strip())
        except:
            raise ValueError("Could not convert to json:\n" + output)
    
    def list_parser(output):
        """
        Converts string to list, assuming that the string is well-formed Python list syntax.
        This is VERY finicky; tried my best with the format_instructions.
        """
        try:
            return ast.literal_eval(output.strip())
        except:
            raise ValueError("Could not convert to list:\n" + output)
    
    parsers = {
        "str": {
            "function": string_parser,
            "format_instructions": ""
        },
        "json": {
            "function": json_parser,
            "format_instructions": textwrap.dedent("""
                Return your answer as a well-formed json object. Only return the json object; nothing else.
                Do not include any formatting like "```json" or "```" around your answer. I will be parsing this with json.loads().
                """).strip()},
        "list": {
            "function": list_parser,
            "format_instructions": textwrap.dedent("""
                Return your answer as a sort of Python list, though with a back slash and a double quote (\") around each item,
                like this: [\"item1\", \"item2\", \"item3\"]. It is important to escape the double quotes so that we can parse it properly.
                Only return the list; nothing else. Do not include any formatting like "```json" or "```" around your answer.
                I will be using python ast.literal_eval() to parse this.
                """).strip()},
        "curriculum_parser": {
            "function": json_parser,
            "format_instructions": textwrap.dedent("""
                Return your answer as a json object with this structure (this is just an example):
                {
                    "topic": "ux design",
                    "curriculum": [
                        {
                            "module_number" : 1,
                            "topic" : "topic",
                            "rationale" : "rationale",
                            "description" : "description",
                        }
                        {
                            "module_number" : 2,
                            "topic" : "topic",
                            "rationale" : "rationale",
                            "description" : "description",
                        }
                        {
                            "module_number" : 3,
                            "topic" : "topic",
                            "rationale" : "rationale",
                            "description" : "description",
                        }
                    ]
                }

                Return your answer as a well-formed json object. Only return the json object; nothing else.
                Do not include any formatting like "```json" or "```" around your answer. I will be parsing this with json.loads().
                """).strip()}
        }
    
    def __init__(self, parser = "str", format_instructions = ""):
        """
        User can set parser to set of pre-defined parsers (in Parser.parsers) or a custom parser (a function that takes a string).
        Default parser is 'str'. Will throw an error if it doesn't get string (that would be a problem!)
        'str' has empty format_instructions; defaults are added if you use a parser from Parser.parsers.
        User can also define their own format instructions.
        When instantiating a Chain object, the format_instructions from the parser are added to the prompt, and included after Prompt.render.
        """
        if parser in Parser.parsers:
            self.parser = Parser.parsers[parser]['function']
            self.format_instructions = Parser.parsers[parser]['format_instructions']
        elif callable(parser):
            self.parser = parser
            self.format_instructions = format_instructions
        else: 
            raise ValueError(f"Parser is not recognized: {parser}")
    
    def __repr__(self):
        return Chain.standard_repr(self)
    
    def parse(self, output):
        return self.parser(output)

class Response():
    """
    Simple class for responses.
    A string isn't enough for debugging purposes; you want to be able to see the prompt, for example.
    Should read content as string when invoked as such.
    TO DO: have chains pass a log from response to response (containing history of conversation).
    """
    
    def __init__(self, content = "", status = "N/A", prompt = "", model = "", duration = 0.0, messages = [], variables = {}):
        self.content = content
        self.status = status
        self.prompt = prompt
        self.model = model
        self.duration = duration
        self.messages = messages
        self.variables = variables
    
    def __repr__(self):
        return Chain.standard_repr(self)
    
    def __len__(self):
        """
        We want to be able to check the length of the content.
        """
        return len(self.content)
    
    def __str__(self):
        """
        We want to pass as string when possible.
        Allow json objects (dict) to be pretty printed.
        """
        if isinstance(self.content, dict):
            return json.dumps(self.content, indent=4)
        elif isinstance(self.content, list):
            return str(self.content)
        else:
            return self.content
    
    def __add__(self, other):
        """
        We this to be able to concatenate with other strings.
        """
        if isinstance(other, str):
            return str(self) + other
        return NotImplemented

class Chat():
    """
    My first implementation of a chatbot.
    """
    def __init__(self, model=Chain.examples['model_example'], system_prompt=Chain.examples["system_prompt_example"]):
        self.model = Model(model)
        self.system_prompt = system_prompt
    
    def __repr__(self):
        return Chain.standard_repr(self)
    
    def chat(self):
        """
        Chat with the model.
        """
        messages = [{'role': 'system', 'content': self.system_prompt}]
        print("Let's chat! Type '/exit' to leave.")
        while True:
            # handle annoying Claude exception (they don't accept system prompts)
            if self.model.model in Chain.models['anthropic']:
                if messages[0]['role'] == 'system':
                    messages[0]['role'] = 'user'
                    messages.append({'role': 'assistant', 'content': 'OK, I will follow that as my system message for this conversation.'})
            # grab user input
            user_input = input("You: ")
            new_system_prompt, new_model = "", ""       # reset these each time
            if user_input[:12] == "/set system ":       # because match/case doesn't do wildcards or regex.
                new_system_prompt = user_input[12:]
                user_input = "/set system"
            if user_input[:11] == "/set model ":        # because match/case doesn't do wildcards or regex.
                new_model = user_input[11:]
                user_input = "/set model"
            match user_input:
                case "/exit":
                    break
                case "/clear":
                    messages = [{'role': 'system', 'content': self.system_prompt}]
                    continue
                case "/show system":
                    print('============================\n' + 
                        self.system_prompt +
                        '\n============================\n')
                    continue
                case "/show model":
                    print(self.model.model)
                    continue
                case "/show models":
                    print(Chain.models)
                    continue
                case "/show messages":
                    print('============================\n' + 
                        '\n\n'.join([str(m) for m in messages]) +
                        '\n============================\n')
                    continue
                case "/set system":
                    if not new_system_prompt:
                        print("You need to enter a system prompt.")
                    else:
                        self.system_prompt = new_system_prompt
                        messages = [{'role': 'system', 'content': self.system_prompt}]
                    continue
                case "/set model":
                    if not new_model:
                        print("You need to enter a model.")
                    else:
                        try:
                            self.model = Model(new_model)
                            print(f"Model set to {new_model}. It may take a while to load after your next message.")
                        except ValueError:
                            print(f"Model not found: {new_model}")
                    continue
                case "/help":
                    print(textwrap.dedent("""\
                        ============================
                        Commands:
                            /exit: leave the chat
                            /clear: clear the chat history
                            /show system: show the system prompt
                            /show model: show the current model
                            /show models: show all available models
                            /show messages: show the chat history
                            /set system: set the system prompt
                            /set model: set the model
                        ============================
                    """).strip())
                    continue
                case _:
                    pass
            messages.append({"role": "user", "content": user_input})
            response = self.model.query(messages)
            messages.append({"role": "assistant", "content": response})
            print(f"Model: {response}")

