from Chain import Model, Verbosity

print("-------------------------")
print("PLAINTEXT")
# These should show enhanced output
model = Model("gpt-4o")
print("""response = model.query("What is 2+2?", verbose=Verbosity.SUMMARY)""")
response = model.query("What is 2+2?", verbose=Verbosity.SUMMARY)  # Enhanced display
print(response.content)

print("""response = model.query("What is 2+2?", verbose="vv")""")
response = model.query("What is 2+2?", verbose="vv")             # Same thing
print(response.content)

# This should still work as before  
print("""response = model.query("What is 2+2?", verbose=True)""")
response = model.query("What is 2+2?", verbose=True)             # Normal progress
print(response.content)

print("""response = model.query("What is 2+2?", verbose=False)""")
response = model.query("What is 2+2?", verbose=False)   
print(response.content)



print("-------------------------")
print("RICHTEXT")
# These should show enhanced output; instantiate Console first.
from rich.console import Console
console = Console()
Model._console = console
# Tests
model = Model("gpt-4o")
print("""response = model.query("What is 2+2?", verbose=Verbosity.SUMMARY)""")
response = model.query("What is 2+2?", verbose=Verbosity.SUMMARY)  # Enhanced display
print(response.content)

print("""response = model.query("What is 2+2?", verbose="vv")""")
response = model.query("What is 2+2?", verbose="vv")             # Same thing
print(response.content)

# This should still work as before  
print("""response = model.query("What is 2+2?", verbose=True)""")
response = model.query("What is 2+2?", verbose=True)             # Normal progress
print(response.content)

print("""response = model.query("What is 2+2?", verbose=False)""")
response = model.query("What is 2+2?", verbose=False)   
print(response.content)



