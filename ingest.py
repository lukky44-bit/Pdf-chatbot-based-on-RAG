from langchain_core.documents import Document
from pdf import extract_pdf_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from ocr import ocr_image
from audio import audio_to_text


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

        elif file.suffix.lower() in [".mp3", ".wav", ".m4a", ".mp4"]:
            print(f"transcribing audio file: {file.name}")
            text = audio_to_text(file)

        else:
            continue

        docs.append(Document(page_content=text, metadata={"source": file.name}))

    print(f"Total documents loaded: {len(docs)}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks
