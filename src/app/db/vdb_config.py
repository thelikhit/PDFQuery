from dotenv import load_dotenv
import logging
import os

import chromadb

load_dotenv()

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = {
    "CHROMA_API_KEY": "ChromaDB API key",
    "CHROMA_TENANT": "ChromaDB tenant",
    "CHROMA_DATABASE": "ChromaDB database",
}

COLLECTION_NAME = "documents"


def vdb_config():
    missing = [var for var, label in _REQUIRED_ENV_VARS.items() if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    try:
        chroma_client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        logger.info("ChromaDB client created for tenant '%s'", os.getenv("CHROMA_TENANT"))
    except Exception as e:
        logger.error("Failed to create ChromaDB client: %s", e)
        raise

    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )
        logger.info("Collection '%s' ready", COLLECTION_NAME)
    except Exception as e:
        logger.error("Failed to get or create collection '%s': %s", COLLECTION_NAME, e)
        raise

    return chroma_client, collection