import argparse
from Chain.model.models.modelstore import ModelStore

def main():
    parser = argparse.ArgumentParser(description="CLI for managing models.")
    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Name of the model to retrieve details for."
    )
    args = parser.parse_args()

    if args.model:
        modelspec = ModelStore.get_model(args.model)
        modelspec.card
    else:
        ModelStore.display()

if __name__ == "__main__":
    main()


