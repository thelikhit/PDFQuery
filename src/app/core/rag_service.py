from src.app.db.vdb import vdb_config
from src.app.core.models import get_llm_client, get_embedding_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def retrieve(query_text: str):

    # perform embedding on the query
    query_embeddings = get_embedding_model().embed_documents([query_text])

    # connection to vector database, here ChromaDB
    vdb_client, vdb_collection = vdb_config()

    # get relevant chunks based on embeddings
    query_results = vdb_collection.query(
        query_embeddings=query_embeddings,
        n_results=5,
    )

    # Quellen
    sources = {
        meta["source"]
        for group in query_results["metadatas"]
            for meta in group
                if "source" in meta
    }
    sources_list = list(sources)

    context_text = [str(doc) for sublist in query_results.get("documents", []) for doc in sublist]

    return context_text, sources_list

def generate(query_text: str, context_text: list, sources_list: list):

    prompt_template = """
    Answer the question based only upon the following context:
    {context}
    ----
    Answer the question based on the above context: {question}
    """

    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # get openai llm client
    llm_client = get_llm_client()

    # generate response from llm
    response = llm_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": str(prompt),
            }
        ],
        max_completion_tokens=16384,
        model=os.getenv("AZURE_OPENAI_LANGUAGE_MODEL_NAME")
    )

    return response.choices[0].message.content

def rag(query_text: str, return_context: bool = False):
    context_text, sources_list = retrieve(query_text)
    result = generate(query_text, context_text, sources_list)
    if return_context:
        return result, context_text
    return result



