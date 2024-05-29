from Chain import Model
from Course_Descriptions import query_db
import re

System_prompt = """  
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

# messages = [{'role': 'user', 'content': 'name ten mammals'}, {'role': 'assistant', 'content': 'Sure, here are ten examples of mammals:\n\n1. African Elephant\n2. Blue Whale \n3. Human \n4. Bengal Tiger \n5. Koala \n6. Gray Wolf \n7. Brown Bat \n8. Gorilla \n9. Kangaroo \n10. Polar Bear \n\nMammals are distinguished by certain characteristics such as having hair or fur, being warm-blooded, and most giving live birth (with the exception of monotremes like the platypus and echidna which lay eggs).'}, {'role': 'user', 'content': 'tell me more about #10'}]

def extract_action_and_input(text):
    action_pattern = r"Action: (.+?) +?\n"
    input_pattern = r"Action Input: \"(.+?)\""
    action = re.findall(action_pattern, text)
    action_input = re.findall(input_pattern, text)
    return action, action_input

def search(query):
    return query_db(query, n_results=5)

model = Model('gpt-3.5-turbo-0125')
messages = [{'role': 'system', 'content': System_prompt}]
print("Let's chat! Type 'exit' to leave.")
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    messages.append({"role": "user", "content": user_input})
    response = model.query(messages)
    action, action_input = extract_action_and_input(response)
    if action[-1] == "Search_Courses":
        tool = search
    elif action[-1] == "Response To Human":
        print(f"Response: {action_input[-1]}")
        break
    observation = tool(action_input[-1])
    messages.extend([
        {"role":"assistant","content":response},
        {"role":"user","content": f"Observation:{observation}"},])
    print(f"Model: {response}")

