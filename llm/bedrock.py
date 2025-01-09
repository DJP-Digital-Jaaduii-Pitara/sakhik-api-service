import os
from typing import Any

from langchain_aws import ChatBedrock
from llm.base import BaseChatClient


class BedrockChatClient(BaseChatClient):

    """
    This class provides a bedrock chat interface.
    """
    def get_client(self, model=os.getenv("BEDROCK_MODEL_ID"), **kwargs: Any) -> ChatBedrock:
        """
        This method creates and returns a ChatBedrock instance.

        Args:
            model (str, optional): The Bedrock model to use. Defaults to the value of the environment variable "BEDROCK_MODEL_ID".
            **kwargs: Additional arguments to be passed to the ChatBedrock constructor.

        Returns:
            An instance of the ChatBedrock class.
        """
        return ChatBedrock(
            # credentials_profile_name=credentials_profile_name,
            # provider="anthropic",
            model_id=model,
        )  # , **kwargs)