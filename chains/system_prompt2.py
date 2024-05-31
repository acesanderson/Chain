"""
The nested while-loops of the first iteration of this script are too confusing and hard to bug.
I will refactor the script to make it more readable and easier to debug.
Embracing state-driven programming.
"""
from Chain import Model
from Course_Descriptions import query_db
import re
import logging

system_prompt = """  
You work as the learning and development administrator for a large enterprise company (over 1,000 employees),
and you have been asked by leadership to help people develop learning path of video courses that will address
the most important skills for your company.

Answer the following questions and obey the following commands as best you can.  

You have access to the following tool:

Search_Courses: you can use this tool to search for courses in the database. Use natural language, and aim for rich description
(e.g. "machine learning for a business audience who doesn't know how to code").

You will receive a message from the human, then you should do one of two things:

Option 1: You use the Search_Courses tool to answer the question.
For this, you should use the following format:
Thought: you should always think about what to do
Action: Search_Courses
Action Input: "the query for the courses database"

Option 2: You respond to the human.
For this, you should use the following format:
Action: Respond_To_Human
Action Input: "your response to the human, either asking them for more information or providing them with the information they need."

Your answer should ALWAYS have the Action: and Action Input: lines.

Begin!
"""

example_chat = [{'role': 'system', 'content': '  \nYou work as the learning and development administrator for a large enterprise company (over 1,000 employees),\nand you have been asked by leadership to help people develop learning path of video courses that will address\nthe most important skills for your company.\n\nAnswer the following questions and obey the following commands as best you can.  \n\nYou have access to the following tool:  \n\nSearch_Courses: you can use this tool to search for courses in the database. Use natural language, and aim for rich description\n(e.g. "machine learning for a business audience who doesn\'t know how to code"). Your query should be more nuanced than "machine learning courses";\nbe sure to ask the human about ideal audience, prerequisites, and other relevant details that will help with your query. The response to your query\nwill be a list of course titles + their descriptions.\n\nYou will receive a message from the human, then you should start a loop and do one of two things:\n \nOption 1: You use a tool to answer the question.  \nFor this, you should use the following format:  \nThought: you should always think about what to do  \nAction: the action to take, should be one of [Search_Courses]\nAction Input: "the input to the action, to be sent to the tool"  \n \nAfter this, the human will respond with an observation, and you will continue.  \n \nOption 2: You respond to the human.  \nFor this, you should use the following format:  \nAction: Response To Human  \nAction Input: "your response to the human, summarizing what you did and what you learned"  \n\nBegin!\n'}, {'role': 'user', 'content': 'I want to learn to be a python developer'}, {'role': 'assistant', 'content': ' Thought: The human has asked me to help them learn Python development. I will use the Search_Courses tool to find courses that fit their request.\n\nAction: Search_Courses\nAction Input: "Python development for beginners"'}]
messages = []
# model = Model('gpt-3.5-turbo-0125')
model = Model('gpt-4o')

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def human_input():
    """
    Square one -- we return here after machine does all its work.
    User has some command options.
    """
    logging.debug("State 1: Human_Input")
    while True:
        # 1: User Input
        user_input = input("You: ")
        match user_input:
            case "exit":
                return "Exiting..."
            case "/show system":
                print('============================\n' + 
                    system_prompt +
                    '\n============================\n')
                continue
            case "/show model":
                print(model.model)
                continue
            case "/show messages":
                print('============================\n' + 
                    '\n\n'.join([str(m) for m in messages]) +
                    '\n============================\n')
                continue
            case "/help":
                print("""
    Type 'exit' to leave the chat.
    Type '/show system' to see the system prompt.
    Type '/show model' to see the model.
    Type '/show messages' to see the conversation.
    """)
                continue
            case _:
                break
    messages.append({"role": "user", "content": user_input})
    return machine_evaluates_human_input

def machine_evaluates_human_input():
    """
    Machine takes human input and decides to either
    1. Respond to the human
    2. Query the database
    """
    logging.debug("State 2: Machine_Evaluates_Human_Input")

    def extract_action_and_input(text):
        """
        Extracts the action and input variables from the text.
        This is regex-heavy.
        """
        logging.debug("Running extract_action_and_input...")
        try:
            action_pattern = r"Action: (.*?)\n"
            input_pattern = r"Action Input: \"(.+?)\""
            action = re.findall(action_pattern, text)[0].replace('[','').replace(']', '').strip()
            action_input = re.findall(input_pattern, text)[0].strip()
            return action, action_input
        except:
            # throw an error here
            error = "Error: Could not extract action and input from text."
            logging.debug(text)
            return error
    
    # Machine evaluates human input
    response = model.query(messages)
    logging.debug(f"Model: {response}")
    
    try:
        action, action_input = extract_action_and_input(response)
    except:
        error = "Error: Could not extract action and input from response."
        print(response)
        return(error)
    
    match action:
        case "Search_Courses":
            messages.append({"role": "assistant", "content": action_input})
            return machine_queries_database
        case "Respond_To_Human":
            messages.append({"role": "assistant", "content": action_input})
            logging.debug(f"Response: {action_input}")
            return human_input
        case _:
            error = "Error: Unrecognized action."
            return error

def machine_queries_database():
    """
    Machine queries the database for the requested information.
    """
    logging.debug("State 3: Machine_Queries_Database")

    def search(query):
        """
        Wrapper for our query_db function. This is a vectordb of LiL course descriptions.
        """
        return query_db(query, n_results=10)
    
    try:
        action_input = messages[-1]['content']
    except:
        logging.debug(messages[-1])
        error = "Error: Could not extract action input from messages."
        return error
    
    courses = search(action_input)
    courses = '\n'.join([f"{i+1}. {courses[i]}" for i in range(len(courses))])
    messages.append(courses)
    return machine_receives_database_query

def machine_receives_database_query():
    """
    Here the machine receives the database query and responds to the human.
    """
    logging.debug("State 4: Machine_Receives_Database_Query")

    # Pop the courses string from messages.
    if isinstance(messages[-1], str):
        courses = messages[-1]
        _ = messages.pop()
    else:
        return "Error: Expected courses string in messages, but none found."
    
    prompt = f"""
    Look at the following courses.
    ===========               
    {courses}
    ===========
    Do any of these answer the user's most recent query?
    
    If so, mention them in your response.
    If not, do not mention them.
    
    Assume that the user cannot see these courses in this prompt, and that you must summarize them when presenting the courses to the user.
    """
    messages.append({"role": "user", "content": prompt})
    response = model.query(messages[1:])            # messages [1:] to exclude the system prompt, important!
    print(f"Model: {response}")
    messages.append({"role": "assistant", "content": response})
    return human_input

if __name__ == "__main__":
    current_state = human_input  # Start with the initial state
    messages = [{'role': 'system', 'content': system_prompt}]
    print("Let's chat! Type 'exit' to leave.")
    while True:
        # If the current state is not callable, it's an error. Print and break.
        if not callable(current_state):
            print(current_state)
            break
        current_state = current_state()
    for m in messages:
        print(m)                          # for debugging -- what was the conversation?

