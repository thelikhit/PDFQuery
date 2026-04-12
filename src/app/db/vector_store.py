# load document using langchain document loader
# https://reference.langchain.com/python/langchain-community/document_loaders

"""
load pdf document from user/path, perform chunking, and save chunks to vector database.
"""

from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema.document import Document
from src.app.core.embedding_model import get_embedding_model
import hashlib
from src.app.db.vdb_config import vdb_config


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