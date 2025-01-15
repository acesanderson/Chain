from openai import OpenAI, Stream
import os
import re
from ast import literal_eval
from Chain.message.message import Message

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ReACT syntax functions


def query_openai_streaming(messages: list[Message]) -> Stream:
    messages_as_dicts: list[dict] = [m.model_dump() for m in messages]
    stream = client.chat.completions.create(  # type:ignore
        messages=messages_as_dicts,  # type:ignore
        model="gpt-4o",
        stream=True,
    )
    return stream


def process_stream(stream) -> tuple[str, dict, str]:
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
        tool = re.search(r"<tool>(.*?)</tool>", buffer, re.DOTALL).group(  # type:ignore
            1
        )
        args = re.search(r"<args>(.*?)</args>", buffer, re.DOTALL).group(  # type:ignore
            1
        )
        args = literal_eval(args)
        return tool, args, buffer
    except AttributeError:
        return "", {}, buffer


def return_observation(observation: str) -> Message:
    observation_string = f"<observation>{observation}</observation>"
    user_message = Message(role="user", content=observation_string)
    return user_message


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


# Updated system prompt
system_prompt_string = """
You are a weather planning assistant. Given a temperature in celsius and wind speed, help determine appropriate clothing recommendations.
You will interleave Thought (<thought>), Tool Name (<tool>), and Tool Args (<args>), and receive a resulting Observation (<observation>).
Thought can reason about the current situation, and Tool Name can be the following types:

(1) convert_temperature, whose description is <desc>Converts celsius to fahrenheit</desc>. It takes arguments {"celsius": float} in JSON format.

(2) calculate_wind_chill, whose description is <desc>Calculates wind chill given temperature in fahrenheit and wind speed in mph</desc>. It takes arguments {"temp_fahrenheit": float, "wind_speed_mph": float} in JSON format.

(3) get_clothing_recommendation, whose description is <desc>Provides clothing recommendations based on felt temperature in fahrenheit</desc>. It takes arguments {"felt_temp": float} in JSON format.

(4) finish, whose description is <desc>Signals that the final recommendation is available and marks the task as complete.</desc>. It takes arguments {"final_answer": str} in JSON format.

A typical interaction might look like:
<thought>First, I need to convert the temperature to fahrenheit</thought>
<tool>convert_temperature</tool>
<args>{"celsius": -5.0}</args>
<observation>23.0</observation>
<thought>Now I can calculate the wind chill using this temperature and the given wind speed</thought>
<tool>calculate_wind_chill</tool>
<args>{"temp_fahrenheit": 23.0, "wind_speed_mph": 10.0}</args>
<observation>9.8</observation>
<thought>With the felt temperature, I can get appropriate clothing recommendations</thought>
<tool>get_clothing_recommendation</tool>
<args>{"felt_temp": 9.8}</args>
<observation>Heavy winter coat, layers, gloves, winter hat, and insulated boots required</observation>
<thought>I now have all the information needed to provide a complete recommendation</thought>
<tool>finish</tool>
<args>{}</args>
""".strip()

tools = [convert_temperature, calculate_wind_chill, get_clothing_recommendation]

if __name__ == "__main__":
    # Initialize Message objects
    messages = [Message(role="system", content=system_prompt_string)]
    messages.append(
        Message(
            role="user",
            content="What should I wear for -5°C weather with 10 mph winds?",
        )
    )
    while True:
        # Query OpenAI with the messages so far
        stream = query_openai_streaming(messages)
        tool, args, buffer = process_stream(stream)
        if tool == "finish":
            final_answer = args["final_answer"]
            print(final_answer)
            break
        messages.append(Message(role="assistant", content=buffer))
        observation = ""
        for t in tools:
            if t.__name__ == tool:
                observation = str(t(**args))
        if observation:
            user_message = return_observation(observation)
            messages.append(user_message)
        else:
            print("No observation found")
