# load document using langchain document loader
# https://reference.langchain.com/python/langchain-community/document_loaders

"""
load pdf document from user/path, perform chunking, and save chunks to vector database.
"""

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema.document import Document

from get_embedding_model import get_embedding_model
import hashlib

from vdb_config import vdb_config

DATA_PATH = "pdfs"

# load document from user/from doc path
def _load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# chunk document
def _split_document(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
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
    ids = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]

    vdb_client, vdb_collection = vdb_config()

    # 4. Add to Chroma collection
    vdb_collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

def pdf_to_vectordb():
    # load pdf
    documents = _load_documents()

    # chunk pdf
    chunks = _split_document(documents)

    # add to vector
    _add_to_vector_database(chunks)
