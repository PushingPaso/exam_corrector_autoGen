import os

from autogen_core.models import ModelInfo
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str = None, output_format = None):
    if model_name is None:
        model_name = "gpt-4o"
        # base_url="https://api.groq.com/openai/v1",


    model_client = OpenAIChatCompletionClient(
        model=model_name,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "openai"
        }
    )
    return model_client


class AIOracle:
    """Base class for AI-powered operations using Groq."""

    def __init__(self, model_name: str = None,output_format = None):
        self.__llm = get_llm(model_name,output_format)

    @property
    def llm(self):
        return self.__llm

    @property
    def model_name(self):
        return "gpt-4o"

    @property
    def model_provider(self):
        return "opanAI"