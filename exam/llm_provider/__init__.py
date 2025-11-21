import os

from autogen_core.models import ModelInfo
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str = None, output_format = None):

    if model_name is None:
        #model_name = "llama-3.3-70b-versatile"
        model_name = "gpt-5.1-2025-11-13"

        # Configura il client per usare l'endpoint di Groq
        # Groq è compatibile con l'SDK OpenAI, basta cambiare base_url e api_key
    model_client = OpenAIChatCompletionClient(
        model=model_name,
        #api_key=os.getenv("GROQ_API_KEY"),  # Assicurati di avere questa var d'ambiente
        #base_url="https://api.groq.com/openai/v1",
        api_type= "openai",
        api_base= "https://api.openai.com/v1",
    # Endpoint specifico per Groq

        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "groq"
            "str"
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
        return "gpt-5.1-2025-11-13"

    @property
    def model_provider(self):
        return "opanAI"