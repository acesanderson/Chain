import re
from ast import literal_eval
from Chain.message.message import Message
from Chain.model.model import Model
from Chain.react.Tool import Tool
from jinja2 import Template
from pathlib import Path
from typing import Callable

# Load system prompt from jinja file
dir_path = Path(__file__).parent
system_prompt_path = dir_path / "system_prompt.jinja"
with open(system_prompt_path, "r") as file:
    system_prompt_string = Template(file.read().strip())

preferred_model = "gpt"


# ReACT syntax functions
class ReACT:
    """
    A ReACT is a variant of Chain that defines a simple input-output ReACT workflow.
    Parameters:
    - input: str (this goes into system prompt, and tells the LLM what the input would be)
    - output: str (same as above, but for output)
    - tools: list[Callable] (a list of your functions; they should take parameters and have descriptive + short docstrings)
    - model: a Model object (currently only Model('gpt') support streaming)
    """

    def __init__(self, input: str, output: str, tools: list[Callable], model: Model):
        self.input = input
        self.output = output
        self.tools = tools
        self.model = model
        # Generate system prompt
        self.tool_objects = [Tool(tool) for tool in tools]
        self.system_prompt = self.render_system_prompt(input, output, self.tool_objects)
        # Initialize Message objects
        self.messages = [Message(role="system", content=self.system_prompt)]

    def render_system_prompt(self, input: str, output: str, tool_objects: list[Tool]):
        system_prompt = system_prompt_string.render(
            input=input, output=output, tool_objects=tool_objects
        )
        return system_prompt

    def process_stream(self, stream) -> tuple[str, dict, str]:
        buffer = ""
        for chunk in stream:
            buffer += str(chunk.choices[0].delta.content)
            if "</args>" in buffer:
                stream.close()
                break
        # Process either the args or the finish tool
        buffer = re.sub(r"</args>.*", "</args>", buffer, flags=re.DOTALL)
        # Stop token gets rendered as None, so we need to remove the last 4 characters
        if buffer.endswith("None"):
            buffer = buffer[:-4]
        # Grab two bits of data: <tool></tool> and <args></args>
        try:
            tool = re.search(
                r"<tool>(.*?)</tool>", buffer, re.DOTALL
            ).group(  # type:ignore
                1
            )
            args = re.search(
                r"<args>(.*?)</args>", buffer, re.DOTALL
            ).group(  # type:ignore
                1
            )
            args = literal_eval(args)
            return tool, args, buffer
        except AttributeError:
            return "", {}, buffer

    def return_observation(self, observation: str) -> Message:
        observation_string = f"<observation>{observation}</observation>"
        user_message = Message(role="user", content=observation_string)
        return user_message

    def query(self, prompt: str) -> str | None:
        self.messages.append(Message(role="user", content=prompt))
        # Our main loop
        while True:
            # Query OpenAI with the messages so far
            stream = self.model.stream(self.messages, verbose=False)
            tool, args, buffer = self.process_stream(stream)
            # Determine if we have final answer
            if tool == "finish":
                final_answer = args["final_answer"]
                print(final_answer)
                self.messages.append(Message(role="assistant", content=final_answer))
                break
            self.messages.append(Message(role="assistant", content=buffer))
            # Generate observation on command
            observation = ""
            for tool_object in self.tool_objects:
                if tool_object.name == tool:
                    observation = str(tool_object(**args))
            if observation:
                user_message = self.return_observation(observation)
                self.messages.append(user_message)
            else:
                print("No observation found")


# Our tools
def convert_temperature(celsius: float) -> float:
    """Convert celsius to fahrenheit"""
    return (celsius * 9 / 5) + 32


def calculate_wind_chill(temp_fahrenheit: float, wind_speed_mph: float) -> float:
    """
    Calculate wind chill using the NWS formula
    Valid for temperatures <= 50°F and wind speeds >= 3 mph
    """
    if temp_fahrenheit > 50 or wind_speed_mph < 3:
        return temp_fahrenheit

    wind_chill = (
        35.74
        + (0.6215 * temp_fahrenheit)
        - (35.75 * wind_speed_mph**0.16)
        + (0.4275 * temp_fahrenheit * wind_speed_mph**0.16)
    )
    return round(wind_chill, 1)


def get_clothing_recommendation(felt_temp: float) -> str:
    """Get clothing recommendations based on the felt temperature"""
    if felt_temp < 0:
        return "Heavy winter coat, layers, gloves, winter hat, and insulated boots required"
    elif felt_temp < 32:
        return "Winter coat, hat, gloves, and warm layers recommended"
    elif felt_temp < 45:
        return "Light winter coat or heavy jacket recommended"
    elif felt_temp < 60:
        return "Light jacket or sweater recommended"
    else:
        return "Light clothing suitable"


if __name__ == "__main__":
    # Our specific scenario
    input = "Facts about the current weather."
    output = "Clothing recommendations."
    prompt = "What should I wear for -5°C weather with 10 mph winds?"
    tools = [convert_temperature, calculate_wind_chill, get_clothing_recommendation]
    model = Model(preferred_model)
    react = ReACT(input, output, tools, model)
    react.query(prompt)
