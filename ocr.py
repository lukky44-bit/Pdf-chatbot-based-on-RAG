import pytesseract
from PIL import Image
from pdf2image import convert_from_path


def ocr_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    return text


def ocr_scanned_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    full_text = ""

    for page_number, page in enumerate(pages):
        text = pytesseract.image_to_string(page)
        full_text += f"\n\n--- Page {page_number + 1} ---\n\n"
        full_text += text

    return full_text


# if __name__ == "__main__":
#     file_path = "/Users/lakshand/Desktop/python/RAG2/data/ocr.png"

#     if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
#         text = ocr_image(file_path)
#     elif file_path.lower().endswith(".pdf"):
#         text = ocr_scanned_pdf(file_path)
#     else:
#         raise ValueError("Unsupported file type")

#     print(text)
