# Set global logging settings
import logging

def set_log_level(level):
    """Set logging level for entire Chain package"""
    logging.getLogger('Chain').setLevel(level)
    
    # Also set for common third-party libraries
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

# Convenience functions
def disable_logging():
    set_log_level(logging.CRITICAL)

def enable_debug_logging():
    set_log_level(logging.DEBUG)

# Imports
from Chain.chain.chain import Chain
from Chain.chain.asyncchain import AsyncChain
from Chain.prompt.prompt import Prompt
from Chain.model.model import Model
from Chain.model.model_async import ModelAsync
from Chain.model.models.models import ModelStore
from Chain.model.params.params import Params, ClientParamsTypes, OllamaParams, OpenAIParams, AnthropicParams, PerplexityParams, GoogleParams
from Chain.result.response import Response
from Chain.result.error import ChainError
from Chain.result.result import ChainResult
from Chain.parser.parser import Parser
from Chain.message.message import Message, create_system_message
from Chain.message.messages import Messages
from Chain.message.imagemessage import ImageMessage
from Chain.message.messagestore import MessageStore
from Chain.cache.cache import ChainCache
from Chain.chat.chat import Chat
from Chain.api.server.ChainRequest import ChainRequest
from Chain.model.clients.server_client import ServerClient
from Chain.cli.cli import CLI, ChainCLI, arg
from Chain.decorator.decorator import llm


__all__ = [
    "Chain",
    "AsyncChain",
    "Prompt",
    "Model",
    "ModelAsync",
    "Parser",
    "Response",
    "ChainError",
    "ChainResult",
    "Message",
    "ImageMessage",
    "MessageStore",
    "ModelStore",
    "Params",
    "ClientParamsTypes",
    "OllamaParams",
    "OpenAIParams",
    "AnthropicParams",
    "PerplexityParams",
    "GoogleParams",
    "create_system_message",
    "Messages",
    "ChainCache",
    "Chat",
    "ChainRequest",
    "ServerClient",
    "CLI",
    "ChainCLI",
    "arg",
    "llm",
    "disable_logging",
    "enable_debug_logging",
]
