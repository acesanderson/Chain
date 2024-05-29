# Chain Framework Documentation

The Chain framework is a Python library for creating and running prompt-based workflows using various language models. It provides a simple and flexible way to define prompts, select models, and parse outputs. This was created out of the frustration of learning Langchain, and finding it to be too verbose, too invested in the non-Pythonic pipe operator, and really hard to debug. This takes a similar concept of a Chain but is more minimal and flexible.

## Why is this helpful?
- use any model you want with a standardized `query()` function (get your text immediately!).
- quickly compose simple chains; build larger chains with some helpful abstractions
- continue to use procedural, function-call based scripting to run your chains.
- use parsers to constrain LLM output to structured data formats

## Hello World
```python
chain = Chain("write a haiku about {{input}}")
output = chain.run("the beauty of nature")
print(output)
```

In this minimal example, a `Chain` object is created by passing a string prompt directly to the constructor. The `run` method is then called with a single string input.

## Batch Processing
```python
chain = Chain("generate a short story about {{input}}")
inputs = {'input':["a haunted house", "a magical forest", "a time-traveling adventure"]}
outputs = chain.batch(inputs)
print(outputs)
```

In this example, a `Chain` object is created with a prompt template. The `batch` method is called with a list of inputs, and it returns a list of corresponding outputs. Each input is processed independently using the specified prompt, model, and parser.

## How a Chain works
A `Chain` object is constructed by combining a `Prompt`, a `Model`, and a `Parser`. Here's a general overview of how these components work together:

1. The `Prompt` object is created with a template string that defines the structure of the prompt. The template can include placeholders (e.g., `{{input}}`) for dynamic content. (this uses Jinja2 syntax)

2. The `Model` object is initialized with the name of the desired language model (e.g., "gpt-4o", "claude-3-haiku-20240307"). The available models are defined in `Model.models`. (these include OpenAI, Anthropic, Google, and local models accessed through the Python `ollama` module) Default is Mistral.

3. The `Parser` object is created with a parser type (e.g., "str", "json", "list") or a custom parser function. The parser determines how the model's output will be processed and returned.

4. The `Chain` object is constructed by passing the `Prompt`, `Model`, and `Parser` objects to its constructor. As in the Hello World example above, if your needs are simple (i.e. one input variable in the template, default model is fine), you can just pass an unlabeled string when initializing `Chain`, and when invoking `Chain.run()`.

5. When `Chain.run` is called with an input dictionary, the following steps occur:
   - The `Prompt` object renders the template string with the provided input, replacing the placeholders with the actual values.
   - The rendered prompt is passed to the `Model` object, which queries the specified language model and returns the generated output.
   - The generated output is then passed to the `Parser` object, which processes the output according to the specified parser type or custom parser function.
   - Finally, the parsed output is returned by `Chain.run` as a `Response`.

This modular design allows for flexibility in creating custom chains with different prompts, models, and parsers, enabling a wide range of applications and use cases.

## Response

## Prompt

## Model

## Parsers

The `Parser` class includes built-in parsers for strings, JSON, and lists, as well as support for custom parser functions. The default is 'str', which simply validates that output is a string (it should be for all queries run with `Model.query()`.)

### JSON Parser Example
```python
prompt = """
Please provide the following information about {{input}} in a JSON format:
- Name
- Description
- Category
"""
chain = Chain(Prompt(prompt), Model("gpt-4o"), Parser("json"))
output = chain.run({"input": "laptop"})
print(output)
```

In this example, a `Chain` object is created with a prompt template that requests information in JSON format. The `Parser` is set to "json", indicating that the output should be parsed as JSON. The `run` method is called with an input dictionary, and the resulting JSON output is printed.

### List Parser Example
```python
prompt = """
Generate a list of 5 creative {{input}} ideas. Format your response as a Python list.
"""

chain = Chain(Prompt(prompt), Model("claude-3-haiku-20240307"), Parser('list'))
output = chain.run({"input": "birthday party themes"})
print(output)
```

In this example, a `Chain` object is created with a prompt template that requests a list of ideas. The `Parser` is set to "list", indicating that the output should be parsed as a Python list. The `run` method is called with an input dictionary, and the resulting list output is printed.

## Additional notes
- Example chains are available in `chains/` directory.
- This framework will be expanded to handle retrieval (documents, splitters, vector dbs), testing (debugging, logging, tracing).