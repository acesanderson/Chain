"""
Note:
- models are directories within ~/.cache/huggingface/hub
- naming convention is: models--BAAI--bge-reranker-large (for example)
- model info can be accessed with:
```python
import huggingface_hub
model_info = huggingface_hub.model_info("BAAI/bge-reranker-large")

```

If using a lot of models, consider changing $HF_HUB_CACHE to a different directory, perhaps on a separate drive.

Consider trying liberal use of load_in_8bit=True and device_map="auto" for large models, or even load_in_4bit=True if you have the right hardware.
"""

from Chain.model.clients.client import Client
from Chain.model.clients.load_env import load_env
from Chain.message.message import Message
import huggingface_hub
import re, json, torch
from pydantic import BaseModel
from pathlib import Path
from transformers import pipeline, Pipeline
from typing import Union, Optional, Any, Callable


hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
dir_path = Path(__file__).resolve().parent
# This should be appended to prompt, and input_variable should be json_schema
# json_schema = json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)
hf_system_prompt = dir_path / "hf_instructor_prompt.jinja2"


def get_model_info(model_id: str):
    """
    Get model info from huggingface_hub
    :param model_id: str
    :return: dict
    """
    # Check if the model is already downloaded
    model_info = huggingface_hub.model_info(model_id)
    return model_info


def is_text_generation_model(model_id: str):
    """
    Check if the model is a text generation model
    :param model_id: str
    :return: bool
    """
    # Check if the model is already downloaded
    model_info = huggingface_hub.model_info(model_id)
    if "text-generation-inference" in model_info.tags:  # type: ignore
        return True
    return False


def get_installed_models() -> list[str]:
    """
    Check cache dir for installed models.
    """
    model_dirs = hf_cache_dir.glob("models--*")
    model_ids = []
    for model_dir in model_dirs:
        model_id = model_dir.name
        model_id = re.sub(r"^models--", "", model_id)
        model_id = re.sub(r"--", "/", model_id)
        model_ids.append(model_id)
    return model_ids


def delete_model(model_id: str):
    """
    Deletes model dir from cache dir
    """
    model_dir = hf_cache_dir / f"models--{model_id.replace('/', '--')}"
    if model_dir.exists():
        print(f"Deleting model dir: {model_dir}")
        model_dir.rmdir()
    else:
        print(f"Model dir {model_dir} does not exist.")


def try_model(model_id: str):
    """
    Download the model and run it through some benchmarks.
    """
    pass


class HuggingFaceClient(Client):
    """
    This is a base class; we have two subclasses: HuggingFaceClientSync and HuggingFaceClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._pipeline_cache: dict[str, Pipeline] = {}
        self.api_token = (
            self._get_api_key()
        )  # This is a dummy for ABC adherence; look at _get_api_key.
        self.default_device = self._determine_device()
        self._client = self._initialize_client()

    def _get_api_key(self) -> str:
        """
        This function would just load the api key, but instead we log in to HF so that models who need it can use it, others can ignore.
        """
        api_key = load_env("HUGGINGFACEHUB_API_TOKEN")
        try:
            huggingface_hub.login(token=api_key, add_to_git_credential=False)
        except Exception as e:
            print(f"Error logging in to Hugging Face: {e}")
            raise RuntimeError(f"Error logging in to Hugging Face: {e}")
        return api_key

    def _determine_device(
        self, requested_device: Optional[Union[str, int]] = None
    ) -> Union[str, int]:
        """
        Determine the device to use for the model.
        :param requested_device: str or int
        :return: str or int
        """
        if requested_device is None:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return requested_device

    def _initialize_client(self) -> Callable:
        """
        Since we are using pipeline as our client, and all args are passed to it, this function literally just returns the pipeline class.
        This adheres to ABC structure.
        """
        return pipeline

    def _get_pipeline(self, model: str, task: str = "text-generation") -> Pipeline:
        """
        Get the pipeline for the model.
        :param model: str
        :param task: str
        :return: Pipeline
        """
        cache_key = f"{task}:{model}"
        if cache_key not in self._pipeline_cache:
            print(f"Creating and caching pipeline for model: {model}, task: {task}")
            try:
                # Use self._client which holds the pipeline function
                self._pipeline_cache[cache_key] = self._client(  # type: ignore
                    task=task,
                    model=model,
                    device=self.default_device,
                )
            except Exception as e:
                print(f"Error creating pipeline: {model}: {e}")
                raise RuntimeError(
                    f"Error creating Hugging Face pipeline: {model}: {e}"
                )
        else:
            print(f"Using cached pipeline for model: {model}, task: {task}")

        return self._pipeline_cache[cache_key]

    def query(
        self,
        model: str,
        input: Union[str, list],
        pydantic_model: Optional[BaseModel] = None,
        raw: bool = False,
        max_new_tokens: int = 256,  # Added common generation parameter
        **kwargs,  # Allow passing other pipeline kwargs
    ) -> Union[str, list[dict[str, Any]]]:  # Return type adjusted
        """
        Queries a Hugging Face model using the text-generation pipeline.

        Args:
            model: The Hugging Face model identifier (e.g., "gpt2", "mistralai/Mistral-7B-v0.1").
            input: A string prompt or a list of dictionaries (OpenAI chat format).
            pydantic_model: NOT SUPPORTED by the standard text-generation pipeline.
                            If provided, a NotImplementedError will be raised.
                            Special implementation needed (leverage the prompt template referenced in constants)
            raw: If True, returns the raw list of dictionaries from the pipeline.
                 If False (default), returns only the generated text string.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Additional keyword arguments passed directly to the pipeline's
                      __call__ method (e.g., temperature, top_k, top_p, num_return_sequences).

        Returns:
            - If raw=False: The generated text as a string.
            - If raw=True: The raw output from the pipeline (a list of dictionaries).

        Raises:
            NotImplementedError: If pydantic_model is provided.
            RuntimeError: If the pipeline fails to initialize.
            TypeError: If the input type is invalid.
        """
        if pydantic_model:
            raise NotImplementedError("Pydantic model structured output is a TBD")
        # Get (or create and cache) the pipeline for the requested model
        generator = self._get_pipeline(model=model, task="text-generation")
        # Format the input into a single string prompt
        if isinstance(input, list):
            if isinstance(input[-1], Message):
                input = str(Message.content)
        # Prepare generation arguments
        generation_kwags = {
            "max_new_tokens": max_new_tokens,
            **kwargs,  # Include any additional kwargs passed by the user
        }
        print(f"Querying model '{model}' with input: '{input[:100]}...'")
        # Run inference
        try:
            response = generator(input, **generation_kwags)
        except Exception as e:
            print(f"Error during inference: {e}")
            raise RuntimeError(f"Error during inference for model '{model}': {e}")

        # Process and return the response
        if raw:
            print("Returning the raw pipeline output.")
            return response  # type: ignore
        else:
            # Standard text-generation pipeline returns a list of dicts,
            # usually with one item containing 'generated_text'.
            if (
                response
                and isinstance(response, list)
                and isinstance(response[0], dict)
                and "generated_text" in response[0]
            ):
                # Extract only the newly generated part (heuristic: remove the prompt)
                # This might not be perfect for all models/pipelines
                full_text = response[0]["generated_text"]
                # A simple way to remove the prompt if it's exactly at the beginning
                if full_text.startswith(input):
                    generated_part = full_text[len(input) :].strip()
                    print(f"Generated text (prompt removed):\n{generated_part}")
                    return generated_part
                else:
                    # Fallback if prompt isn't exactly at the start (might happen with complex formatting)
                    print(
                        f"Generated text (full output, prompt may be included):\n{full_text}"
                    )
                    return full_text  # Return full text if prompt removal is tricky

            else:
                print(
                    f"Warning: Unexpected pipeline output format: {response}. Returning empty string."
                )
                return ""  # Or raise an error


if __name__ == "__main__":
    hf_client = HuggingFaceClient()
    # model = "gpt2"
    model = "axolotl-quants/Llama-4-Scout-17B-16E-Linearized-bnb-nf4-bf16"
    prompt = "Name ten mammals"
    response = hf_client.query(model=model, input=prompt)
    print(f"Response: {response}")
