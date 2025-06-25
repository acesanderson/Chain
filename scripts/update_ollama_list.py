"""
This script updates the list of Ollama models in the Chain module.
Use this when switching environments, ssh tunnels, or when new models are added.
Need to figure out where to automatically implement this in my Chain package to avoid manual updates but also preserve lazy loading.
"""

from Chain.model.model import Model, ModelStore
from rich import console

console = console.Console(width=80)


def main():
    console.print("[green]Updating Ollama Models...[/green]")
    m = Model("llama3.1:latest")
    m._client.update_ollama_models()
    console.print(
        f"[green]Model list updated: [/green][yellow]{ModelStore.models()['ollama']}[/yellow]"
    )


if __name__ == "__main__":
    main()
