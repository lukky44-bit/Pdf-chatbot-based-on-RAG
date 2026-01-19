from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingest import load_files, split_documents
import shutil
import os

DB_path = "/Users/lakshand/Desktop/python/RAG2/data/Vector_store"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(chunks):
    embeddings = get_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_path,
    )

    # VERY IMPORTANT

    return vectordb


def load_vector_store():
    embeddings = get_embedding_model()
    vectordb = Chroma(persist_directory=DB_path, embedding_function=embeddings)
    return vectordb


def rebuild_vector_store(pdf_folder):
    print("Rebuilding vector database...")

    # 1. Delete old vector DB safely
    if os.path.exists(DB_path):
        shutil.rmtree(DB_path)
        print("Old vector store deleted")

    # 2. Rebuild DB
    docs = load_files(pdf_folder)
    chunks = split_documents(docs)
    db = create_vector_store(chunks)

    print("Vector DB rebuilt successfully!")
    return db


# if __name__ == "__main__":
#     docs = load_files("/Users/lakshand/Desktop/python/RAG2/data")
#     chunks = split_documents(docs)
#     db = create_vector_store(chunks)
#     print("Vector store created and saved successfully!")
