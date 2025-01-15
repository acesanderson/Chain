from openai import OpenAI, Stream
import os
import re
from ast import literal_eval
from Chain.message.message import Message

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


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
    # Removing everything after the last </args> tag
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


def convert_temperature(celsius: float) -> float:
    return (celsius * 9 / 5) + 32


system_prompt_string = """
You will be given a temperature in celsius and your goal is to finish with the converted temperature in fahrenheit.

To do this, you will interleave Thought (<thought>), Tool Name (<tool>), and Tool Args (<args>), and receive a resulting Observation (<observation>).

Thought can reason about the current situation, and Tool Name can be the following types:

(1) convert_temperature, whose description is <desc>Converts celsius to fahrenheit</desc>. It takes arguments {"celsius": float} in JSON format.

(2) finish, whose description is <desc>Signals that the final outputs, i.e. `area_at_temp`, are now available and marks the task as complete.</desc>. It takes arguments {} in JSON format.

A typical interaction might look like:

<thought>I need to convert the temperature first</thought>
<tool>convert_temperature</tool>
<args>{"celsius": 25.0}</args>
<observation>77.0</observation>
""".strip()

tools = [convert_temperature]

if __name__ == "__main__":
    messages = [Message(role="system", content=system_prompt_string)]
    messages.append(
        Message(role="user", content="Convert 25 degrees celsius to fahrenheit.")
    )
    stream = query_openai_streaming(messages)

    tool, args, buffer = process_stream(stream)
    messages.append(Message(role="assistant", content=buffer))
    print(args)
    observation = ""
    for t in tools:
        if t.__name__ == tool:
            observation = str(t(**args))
            print(observation)
    if observation:
        user_message = return_observation(observation)
        messages.append(user_message)
        stream = query_openai_streaming(messages)
        tool, args, buffer = process_stream(stream)
        messages.append(Message(role="assistant", content=buffer))
    print(messages)
