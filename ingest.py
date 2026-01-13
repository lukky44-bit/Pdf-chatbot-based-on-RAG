from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path


def load_pdfs(data_path):
    docs = []

    for pdf_file in Path(data_path).glob("*.pdf"):
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        document = loader.load()

        for doc in document:
            doc.metadata["source"] = pdf_file.name

        docs.extend(document)

    print(f"Total documents loaded: {len(docs)}")
    return docs


def split_documents(docs):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks
