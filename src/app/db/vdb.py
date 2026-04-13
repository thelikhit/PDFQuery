# load document using langchain document loader
# https://reference.langchain.com/python/langchain-community/document_loaders

"""
load pdf document from user/path, perform chunking, and save chunks to vector database.
"""

from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema.document import Document
from src.app.core.models import get_embedding_model
import hashlib
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

# accepts bytes
def _load_documents_from_bytes(content: bytes) -> list[Document]:
    pdf = PdfReader(BytesIO(content))
    documents = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        documents.append(Document(
            page_content=text,
            metadata={"page": i}
        ))
    return documents

# chunk document
def _split_document(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=512,
        separators=["\n\n", "\n", ".", " "],
    )
    return text_splitter.split_documents(documents)

# add chunks to vector database
def _add_to_vector_database(chunks: list[Document]):
    # 1. Extract texts and metadata from LangChain Document chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # 2. Generate embeddings using Azure Embedding Model
    print(f"Embedding {len(texts)} chunks...")
    embeddings = get_embedding_model().embed_documents(texts)

    #3. generate ID for each chunk
    ids = [f"{hashlib.md5(chunk.page_content.encode()).hexdigest()}_{i}" for i, chunk in enumerate(chunks)]

    vdb_client, vdb_collection = vdb_config()

    # 4. Add to Chroma collection
    vdb_collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

# accepts content: bytes, passes to new loader
def pdf_to_vectordb(content: bytes):
    documents = _load_documents_from_bytes(content)
    chunks = _split_document(documents)
    _add_to_vector_database(chunks)