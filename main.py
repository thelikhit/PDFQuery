from pdf_to_vectordb import pdf_to_vectordb
from rag import rag

def main():
    pdf_to_vectordb()
    query_text = input("Enter a question based on the document: ")
    response = rag(query_text)
    print(response)

main()