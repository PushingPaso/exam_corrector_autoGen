import getpass
import os

from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()




def get_llm(model_name: str = None) -> dict:

    if model_name is None:
        model_name = "gemini-2.0-flash"
    # Prende l'API key dalla variabile d'ambiente
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    return model_client


class AIOracle:
    """Base class for AI-powered operations using Groq."""

    def __init__(self, model_name: str = None):
        self.__llm = get_llm(model_name)

    @property
    def llm(self):
        return self.__llm

    @property
    def model_name(self):
        return "gemini-2.0-flash"

    @property
    def model_provider(self):
        return "gemini"