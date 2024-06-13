"""
Simple command line interface for asking questions to the chatbot.
Useful for me since I don't know Linux very well.
"""

from Chain import Chain, Model, Prompt
import sys
import platform
import subprocess
import os
import textwrap

system_instructions = """
You are a helpful IT admin who is patiently trying to assist a new coder.
They use Python and Linux. They are experienced with Python programming but don't know much about how to do the following:
- package scripts into proper applications
- use git and GitHub
- set up a development environment
- use things like Docker, virtual environments, or networking tools
- write shell scripts
- use a terminal effectively
- the linux filesystem or basic linux commands (beyond ls, mkdir, cd, mv, etc.)
Your answers should be very short and to the point.
Only provide a solution to the user's problem.
Do not introduce yourself or provide emotional support.
If a code snippet is all that the user needs, just provide the code snippet.

Here are details about the user's hardware, OS, and software:
""".strip()

def get_system_info():
    # Operating System and Version
    os_info = platform.system() + " " + platform.release()
    # Python Version
    python_version = platform.python_version()
    # System Hardware
    cpu_model = platform.processor()
    try:
        memory_size = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True).stdout.strip().split(': ')[1]
    except:
        memory_size = 'Unknown'
    # GPU Information
    try:
        gpu_info = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True).stdout.strip()
    except:
        gpu_info = 'Unknown'
    # Network Information
    try:
        local_ip = subprocess.run(['hostname', '-I'], capture_output=True, text=True).stdout.strip()
    except:
        local_ip = 'Unknown'
    # Shell and Terminal
    shell = os.environ.get('SHELL')
    terminal = os.environ.get('TERM_PROGRAM', 'Unknown')
    # Read .zshrc and .zprofile files
    zshrc_content = read_file_content(os.path.expanduser('~/.zshrc'))
    zprofile_content = read_file_content(os.path.expanduser('~/.zprofile'))
    return textwrap.dedent(f"""
OS: {os_info}
Python: {python_version}
CPU: {cpu_model}
Memory: {memory_size}
GPU: {gpu_info}
Local IP: {local_ip}
Shell: {shell}
Terminal: {terminal}

.zshrc Content:
{zshrc_content}

.zprofile Content:
{zprofile_content}
""").strip()

def read_file_content(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error reading file: {e}"

def query(prompt, system_info):
    model = Model('gpt')
    full_prompt = f"{system_instructions}\n============================\n{system_info}\n============================\n\nUser Query: {prompt}"
    prompt = Prompt(full_prompt)
    chain = Chain(prompt, model)
    response = chain.run()
    return response

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_prompt = " ".join(sys.argv[1:])
        system_info = get_system_info()
        print(query(input_prompt, system_info))
    else:
        print("Expecting a prompt.")



# def query(prompt):
#     """
    
#     """
#     model = Model('gpt')
#     prompt = Prompt(prompt)
#     chain = Chain(prompt, model)
#     response = chain.run()
#     return response

# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         input = " ".join(sys.argv[1:])
#         print(query(input))
#     else:
#         print("Expecting a prompt.")

