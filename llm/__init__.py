import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm.base import (
        BaseChatClient
    )
    from llm.openai import (
        OpenAIChatClient
    )
    from llm.azure_openai import (
        AzureChatClient
    )
    from llm.ollama import (
        OllamaChatClient
    )
    from llm.bedrock import (
        BedrockChatClient
    )


# __all__ = [
#     "BaseChatClient",
#     "OpenAIChatClient"
#     "AzureChatClient",
#     "OllamaChatClient",
#     "BedrockChatClient"
# ]

_module_lookup = {
    "BaseChatClient" : "llm.base",
    "OpenAIChatClient": "llm.openai",
    "AzureChatClient": "llm.azure_openai",
    "OllamaChatClient": "llm.ollama",
    "BedrockChatClient": "llm.bedrock"
}

def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())