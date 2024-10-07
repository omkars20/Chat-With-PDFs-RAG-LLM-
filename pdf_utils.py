
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import re

# # Extract text from PDF file
# def extract_text_from_pdf(pdf_path):
#     try:
#         reader = PdfReader(pdf_path)
#         text = ""
#         for page_number, page in enumerate(reader.pages):
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text
#         return text
#     except FileNotFoundError:
#         print(f"Error: File {pdf_path} not found.")
#     except Exception as e:
#         print(f"An error occurred: {str(e)}")
#     return ""

# # Clean extracted text
# def clean_text(text):
#     # Remove non-ASCII characters
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     # Replace multiple spaces with a single space
#     text = re.sub(r'\s+', ' ', text)
#     # Remove special characters except for necessary punctuation
#     text = re.sub(r'[^\w\s.,@+-]', '', text)
#     return text.strip()

# # Split extracted text into chunks
# def split_pdf_text(text, chunk_size=500, chunk_overlap=100):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " "]
#     )
#     return text_splitter.split_text(text)



import logging
import os
import re
from typing import List, Optional

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Additional imports for advanced preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize NLTK componnet

stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load spaCy model for NER
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.errro("Spacy Model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        raise FileNotFoundError(f"File not found: {pdf_path}")

    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                logger.warning(f"No text found on page {page_number + 1}")
        logger.info(f"Extracted text from {pdf_path}")
        return text
    except Exception as e:
        logger.exception(f"Failed to extract text from PDF: {e}")
        raise


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing unwanted characters and extra spaces.

    Args:
        text (str): Raw text extracted from the PDF.

    Returns:
        str: Cleaned text.
    """
    try:
        # Retain words, whitespace, and common punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        logger.info("Text cleaned successfully.")
        return text.strip()
    except Exception as e:
        logger.exception(f"Failed to clean text: {e}")
        raise


def extract_entities(text: str) -> List[str]:
    """
    Extract named entities from the text.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of named entities.
    """
    try:
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        logger.info(f"Extracted {len(entities)} entities.")
        return entities
    except Exception as e:
        logger.exception(f"Failed to extract entities: {e}")
        raise


def preprocess_text_english(text: str) -> str:
    """
    Preprocess text for English documents, including entity extraction.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text with entities appended.
    """
    try:
        # Step 1: Clean the text
        text = clean_text(text)

        # Step 2: Entity Extraction
        entities = extract_entities(text)

        # Step 3: Tokenization
        tokens = word_tokenize(text)

        # Step 4: Remove stopwords
        tokens = [word for word in tokens if word.lower() not in stop_words_en]

        # Step 5: Lemmatization
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_text = ' '.join(lemmatized_tokens)

        # Step 6: Append entities to the preprocessed text
        if entities:
            entities_text = ' '.join(entities)
            preprocessed_text = lemmatized_text + ' ' + entities_text
        else:
            preprocessed_text = lemmatized_text

        logger.info("English text preprocessing completed with entity extraction.")
        return preprocessed_text
    except Exception as e:
        logger.exception(f"Failed to preprocess English text: {e}")
        raise




def preprocess_text_hinglish(text: str) -> str:
    """
    Preprocess text for Hinglish documents.

    Args:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    try:
        text = clean_text(text)
        # Minimal preprocessing for Hinglish
        logger.info("Hinglish text preprocessing completed.")
        return text
    except Exception as e:
        logger.exception(f"Failed to preprocess Hinglish text: {e}")
        raise



def split_pdf_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks suitable for vector embeddings.

    Args:
        text (str): Preprocessed text.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.exception(f"Failed to split text into chunks: {e}")
        raise
