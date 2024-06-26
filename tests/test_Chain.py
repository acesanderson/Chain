# """
# Command line utility which takes a python file as input and generates a pytest test file.

# Substeps:
# - read the file and load into memory
# - form a prompt with the 

# Modules:
# - instructor: gets a structured response from LLM
# """

# from Chain import Chain
# import instructor

# example_pytest_file = """


# """

# system_prompt = """
# You are a new SDET (Software Development Engineer in Test) who has joined my team.
# You are skilled both in software development and software testing, and are really good at writing test cases.
# You are an expert in Python programming, in particular the Pytest framework.

# You will be pair programming with a developer who is writing python scripts, and your job will be to write
# test cases for the scripts with maximum code coverage.

# For every script you are given, you need to generate the corresponding pytest file with a test function for each.

# Here is an example pytest file that you can use as a model for future pytest files:
# =================================================================
# {{ example_pytest_file }}
# =================================================================
# """.strip()

# pytest_prompt = """
# The developer you are working with has written a python script that needs to be tested.

# Here is the script:

# =================================================================
# {{ script }}
# =================================================================




# """