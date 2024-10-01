from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extract text from PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Split extracted text into chunks
def split_pdf_text(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_text(text)
