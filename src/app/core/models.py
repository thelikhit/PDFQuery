from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
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

def get_embedding_model():
    embedding_model = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    return embedding_model

