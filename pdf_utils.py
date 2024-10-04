
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Extract text from PDF file
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except FileNotFoundError:
        print(f"Error: File {pdf_path} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return ""

# Clean extracted text
def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except for necessary punctuation
    text = re.sub(r'[^\w\s.,@+-]', '', text)
    return text.strip()

# Split extracted text into chunks
def split_pdf_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    return text_splitter.split_text(text)


# if __name__ == "__main__":
#     # Example usage for testing purposes
#     pdf_path  = "resume_all_merged.pdf"
#     text = extract_text_from_pdf(pdf_path)
#     # print('-----------raw text-------------------------')
#     # print(text)
#     text = clean_text(text)
#     chunks = split_pdf_text(text)
#     print('--------------------chunks is ------------')
#     print(chunks[0])
