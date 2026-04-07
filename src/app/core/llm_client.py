from openai import AzureOpenAI

from dotenv import load_dotenv
import os

load_dotenv()


def get_llm_client():
    llm_client = AzureOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    return llm_client