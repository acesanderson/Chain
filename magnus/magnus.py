from Chain import Chain, Model, Prompt
import argparse

def magnus(prompt:str, model = 'llama3.1:70b-instruct-q2_K') -> str:
    """
    Queries Magnus for command line.
    """
    model = Model(model)
    model.custom_ollama_client('magnus')
    prompt = Prompt(prompt)
    chain = Chain(prompt, model)
    response = chain.run()
    return response.content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Magnus from command line')
    parser.add_argument('prompt', type=str, help='Prompt to query Magnus')
    # Add an argument --quick/-q to use the quick model
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--quick', '-q', action='store_true', help='Use the quick model')
    args = parser.parse_args()
    if args.model:
        model = args.model
    elif args.quick:
        model = 'llama3.1:latest'
    else:
        model = 'llama3.1:70b-instruct-q2_K'
    print(magnus(args.prompt, model))

