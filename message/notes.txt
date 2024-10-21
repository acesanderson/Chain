"""
This is fake code for me to design our message_store.
We want to add persistance to our usual chain workflow.
"""
from Chain import Model, Prompt, Chain, Response, Message, MessageStore, Parser

# our prompts
prompt_string = """You are a blah blah blah."""


# our chain
prompt = Prompt(prompt_string)
model = Model("gpt-3.5-turbo")
chain = Chain(prompt, model))

response = chain.run()

# Ephemeral use case
messagestore = MessageStore()
messagestore.add(response)
messagestore.responses # Get all response objects, which may or may not be the entirety of the store (we won't sweat this functionality for now, if I end up introspecting to Response objects often, we extend the functionality.)
messagestore.messages # Get a list of all messages
messagestore # alone, this returns messagestore.messages
messagestore[0] # Get a message by index
messagestore.responses[0] # Get a response by index
messagestore.history # pretty print all messages with an index (first 50 or so chars of long strings)
# Persistent use case
messagestore = MessageStore(file="messages.json")
messagestore.load() # Load messages and response objects from file
messagestore.save() # Save messages and response objects to file
messagestore.clear() # Clear all messages and response objects
messagestore.last() # Get the last message
messagestore.get() # Ge