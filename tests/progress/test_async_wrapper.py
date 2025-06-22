from Chain import AsyncChain, ModelAsync
from rich.console import Console

# Set up rich console for progress display
AsyncChain._console = Console()

# Test with simple prompts
model = ModelAsync("gpt-4o-mini")
chain = AsyncChain(model=model)

prompt_strings = [
    "What is 2+2?",
    "Name five animals",
    "Explain gravity briefly"
]

# This should now show progress for each async operation
responses = chain.run(prompt_strings=prompt_strings, verbose=True)
