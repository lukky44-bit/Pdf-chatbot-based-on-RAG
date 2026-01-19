from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from ocr import ocr_image, ocr_scanned_pdf


def load_files(data_path):
    docs = []

    for file in Path(data_path).iterdir():
        if file.suffix.lower() == ".pdf":
            print(f"Processing PDF: {file.name}")
            text = extract_pdf_text(file)

        elif file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            print(f"Processing Image: {file.name}")
            text = ocr_image(file)

        elif file.suffix.lower() == ".txt":
            print(f"Processing TXT: {file.name}")
            text = file.read_text()

        else:
            continue

        docs.append(Document(page_content=text, metadata={"source": file.name}))

    print(f"Total documents loaded: {len(docs)}")
    return docs


def extract_pdf_text(pdf_path):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    normal_text = ""
    for p in pages:
        normal_text += p.page_content or ""

    # Detect scanned PDF
    if len(normal_text.strip()) < 50:
        print(f"Scanned PDF detected → Using OCR: {pdf_path.name}")
        return ocr_scanned_pdf(pdf_path)

    print(f"Normal PDF detected: {pdf_path.name}")
    return normal_text


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks
