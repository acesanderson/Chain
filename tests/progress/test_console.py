from rich.console import Console
from Chain import Chain, Model

def test_console_resolution():
   # Create test consoles
   console1 = Console()
   console2 = Console()
   
   # Reset state
   Chain._console = None
   
   # Test 1: No console anywhere
   model = Model("gpt-4o")
   print(f"Test 1 - No console: {model.console is None}")
   
   # Test 2: Chain class console
   Chain._console = console1
   print(f"Test 2 - Chain console: {model.console is console1}")
   
   # Test 3: Model instance override
   model.console = console2
   print(f"Test 3 - Model override: {model.console is console2}")
   
   # Test 4: New model inherits Chain console
   model2 = Model("gpt-4o")
   print(f"Test 4 - New model inherits: {model2.console is console1}")

if __name__ == "__main__":
   test_console_resolution()
