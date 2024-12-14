"""
This script updates the list of Ollama models in the Chain module.
Use this when switching environments, ssh tunnels, or when new models are added.
"""

from Chain.model.model import Model
from rich import console

console = console.Console(width=80)


def main():
    console.print("[green]Updating Ollama Models...[/green]")
    m = Model("llama3.1:latest")
    m._client.update_ollama_models()
    console.print(
        f"[green]Model list updated: [/green][yellow]{Model.models['ollama']}[/yellow]"
    )


if __name__ == "__main__":
    main()
