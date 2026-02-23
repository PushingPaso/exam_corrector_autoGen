import os
import getpass
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

# Define the environment variable key for Groq
KEY_GROQ_API_KEY = "GROQ_API_KEY"
KEY_OPENAI_API_KEY = "OPENAI_API_KEY"

def ensure_openai_api_key():
    if not os.environ.get(KEY_OPENAI_API_KEY):
        os.environ[KEY_OPENAI_API_KEY] = getpass.getpass("Enter API key for OpenAI: ")
    return os.environ[KEY_OPENAI_API_KEY]

def ensure_groq_api_key():
    """
    Ensures the Groq API key is set in the environment variables.
    If not, prompts the user to enter it.
    """
    if not os.environ.get(KEY_GROQ_API_KEY):
        api_key = getpass.getpass("Enter your Groq API key: ")
        os.environ[KEY_GROQ_API_KEY] = api_key
    return os.environ[KEY_GROQ_API_KEY]


def get_llm(model_name: str = None, output_format=None):
    """
    Creates and returns an OpenAIChatCompletionClient configured for Groq.
    """
    if model_name is None:
        #model_name = "llama-3.3-70b-versatile"
        #model_name= "llama-3.1-8b-instant"
        model_name = "gpt-4o"

    api_key = ensure_openai_api_key()

    #base_url = "https://api.groq.com/openai/v1"

    model_client = OpenAIChatCompletionClient(
        model=model_name,
        #base_url=base_url,
        api_key=api_key,
        temprature= 0.0,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "llama",
            "structured_output": False
        }
    )
    return model_client


class AIOracle:
    """Base class for AI-powered operations using Groq."""

    def __init__(self, model_name: str = None, output_format=None):
        self.__llm = get_llm(model_name, output_format)
        self._model_name = model_name if model_name else "llama-3.3-70b-versatile"

    @property
    def llm(self):
        return self.__llm

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_provider(self):
        return "Groq"