from langchain_community.document_loaders import PyPDFLoader
from ocr import ocr_scanned_pdf


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
