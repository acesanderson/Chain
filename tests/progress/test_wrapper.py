from Chain import Model

# Test 1: Simple string query with progress
print("=== Test 1: String query with progress ===")
model = Model("gpt-4o-mini")
response = model.query("What is 2+2?", verbose=True)
print(f"Response: {response[:50]}...")
print()

# Test 2: Same query without progress  
print("=== Test 2: String query without progress ===")
response = model.query("What is 2+2?", verbose=False)
print(f"Response: {response[:50]}...")
print()

# Test 3: Long query that gets truncated
print("=== Test 3: Long query (truncation test) ===")
long_query = "This is a very long query that should definitely be truncated because it exceeds the 30 character limit we set for display purposes"
response = model.query(long_query, verbose=True)
print(f"Response: {response[:50]}...")
print()

# Test 4: Message list query
print("=== Test 4: Message list query ===")
from Chain import Message
messages = [
   Message(role="system", content="You are a helpful assistant"),
   Message(role="user", content="What is the capital of France?")
]
response = model.query(messages, verbose=True)
print(f"Response: {response[:50]}...")
print()

# Test 5: Test error handling (optional - might fail)
print("=== Test 5: Error handling (optional) ===")
try:
   response = model.query("", verbose=True)  # Empty query might cause error
except Exception as e:
   print(f"Caught expected error: {e}")
