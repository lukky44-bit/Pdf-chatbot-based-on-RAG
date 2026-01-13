from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ingest import load_pdfs, split_documents

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
    return vectordb


def load_vector_store():
    embeddings = get_embedding_model()
    vectordb = Chroma(persist_directory=DB_path, embedding_function=embeddings)
    return vectordb


if __name__ == "__main__":
    doc = load_pdfs("/Users/lakshand/Desktop/python/RAG2/data")
    chunks = split_documents(doc)
    db = create_vector_store(chunks)
    print("Vector store created and saved successfully!")
