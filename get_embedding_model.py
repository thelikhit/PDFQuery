from langchain_openai import AzureOpenAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()


def get_embedding_model():
    embedding_model = AzureOpenAIEmbeddings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    return embedding_model