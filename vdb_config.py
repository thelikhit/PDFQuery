from dotenv import load_dotenv
import os

load_dotenv()

import chromadb

def vdb_config():
    chroma_client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )

    collection = chroma_client.get_or_create_collection(
        name="documents",
        embedding_function=None,
    )

    return chroma_client, collection