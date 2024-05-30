from Chain import Model
from Course_Descriptions import query_db
import re

system_prompt = """  
You work as the learning and development administrator for a large enterprise company (over 1,000 employees),
and you have been asked by leadership to help people develop learning path of video courses that will address
the most important skills for your company.

Answer the following questions and obey the following commands as best you can.  

You have access to the following tool:  

Search_Courses: you can use this tool to search for courses in the database. Use natural language, and aim for rich description
(e.g. "machine learning for a business audience who doesn't know how to code"). Your query should be more nuanced than "machine learning courses";
be sure to ask the human about ideal audience, prerequisites, and other relevant details that will help with your query. The response to your query
will be a list of course titles + their descriptions.

You will receive a message from the human, then you should start a loop and do one of two things:
 
Option 1: You use a tool to answer the question.  
For this, you should use the following format:  
Thought: you should always think about what to do  
Action: the action to take, should be one of [Search_Courses]
Action Input: "the input to the action, to be sent to the tool"  
 
After this, the human will respond with an observation, and you will continue.  
 
Option 2: You respond to the human.  
For this, you should use the following format:  
Action: Response To Human  
Action Input: "your response to the human, summarizing what you did and what you learned"  

Begin!
"""


"""
[{'role': 'system', 'content': '  \nYou work as the learning and development administrator for a large enterprise company (over 1,000 employees),\nand you have been asked by leadership to help people develop learning path of video courses that will address\nthe most important skills for your company.\n\nAnswer the following questions and obey the following commands as best you can.  \n\nYou have access to the following tool:  \n\nSearch_Courses: you can use this tool to search for courses in the database. Use natural language, and aim for rich description\n(e.g. "machine learning for a business audience who doesn\'t know how to code"). Your query should be more nuanced than "machine learning courses";\nbe sure to ask the human about ideal audience, prerequisites, and other relevant details that will help with your query. The response to your query\nwill be a list of course titles + their descriptions.\n\nYou will receive a message from the human, then you should start a loop and do one of two things:\n \nOption 1: You use a tool to answer the question.  \nFor this, you should use the following format:  \nThought: you should always think about what to do  \nAction: the action to take, should be one of [Search_Courses]\nAction Input: "the input to the action, to be sent to the tool"  \n \nAfter this, the human will respond with an observation, and you will continue.  \n \nOption 2: You respond to the human.  \nFor this, you should use the following format:  \nAction: Response To Human  \nAction Input: "your response to the human, summarizing what you did and what you learned"  \n\nBegin!\n'}, {'role': 'user', 'content': 'I want to learn to be a python developer'}, {'role': 'assistant', 'content': ' Thought: The human has asked me to help them learn Python development. I will use the Search_Courses tool to find courses that fit their request.\n\nAction: Search_Courses\nAction Input: "Python development for beginners"'}]
"""


# messages = [{'role': 'user', 'content': 'name ten mammals'}, {'role': 'assistant', 'content': 'Sure, here are ten examples of mammals:\n\n1. African Elephant\n2. Blue Whale \n3. Human \n4. Bengal Tiger \n5. Koala \n6. Gray Wolf \n7. Brown Bat \n8. Gorilla \n9. Kangaroo \n10. Polar Bear \n\nMammals are distinguished by certain characteristics such as having hair or fur, being warm-blooded, and most giving live birth (with the exception of monotremes like the platypus and echidna which lay eggs).'}, {'role': 'user', 'content': 'tell me more about #10'}]

def extract_action_and_input(text):
    action_pattern = r"Action: (.*?)\n"
    input_pattern = r"Action Input: \"(.+?)\""
    action = re.findall(action_pattern, text)
    action_input = re.findall(input_pattern, text)
    return action, action_input

def search(query):
    return query_db(query, n_results=5)

model = Model('gpt-3.5-turbo-0125')
# model = Model('gpt')
messages = [{'role': 'system', 'content': system_prompt}]
print("Let's chat! Type 'exit' to leave.")
while True:
    # 1: User Input
    user_input = input("You: ")
    match user_input:
        case "exit":
            break
        case "/show system":
            print(system_prompt)
            continue
        case "/show model":
            print(model.model)
            continue
        case "/help":
            print("Type 'exit' to leave the chat. Type '/show system' to see the system prompt. Type '/show model' to see the model.")
            continue
        case _:
            pass
    # 2: Machine evaluates human input
    messages.append({"role": "user", "content": user_input})
    while True:
        response = model.query(messages)
        print(f"Model: {response}")
        action, action_input = extract_action_and_input(response)
        match action[-1]:
            case "Search_Courses":          # 4: Machine queries database.
                tool = search
            case _:
                messages.append({"role": "assistant", "content": action_input[-1]})
                print(f"Response: {action_input[-1]}")
                break
        observation = tool(action_input[-1])
        messages.append({"role":"user","content": f"Here are some courses you can recommend to the user:\n{observation}"})
        response = model.query(messages)
        print(f"Model: {response}")
        messages.append({"role": "assistant", "content": response})

