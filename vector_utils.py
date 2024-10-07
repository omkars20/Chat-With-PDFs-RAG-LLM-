
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
import os
import logging

logger = logging.getLogger(__name__)

# Generate vector embeddings and save FAISS index
def create_vector_store_from_text(chunks, api_key, save_path):
    """
    Generate vector embeddings from text chunks and save FAISS index.

    Args:
        chunks (list): List of text chunks to generate embeddings for.
        api_key (str): OpenAI API key for generating embeddings.
        save_path (str): Directory path to save the FAISS index.

    Returns:
        FAISS: The FAISS vector store instance.
    """
    if not chunks:
        raise ValueError("The chunks list is empty. Please provide valid text chunks.")
    
    try:
        # Using the updated version of OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        vector_store = FAISS.from_texts(chunks, embeddings)
        vector_store.save_local(save_path)
        print(f"vetor store is saved to the {save_path}")
        logger.info(f"Vector store saved at {save_path}")
        return vector_store
    except ValueError as e:
        print(f"ValueError occurred: {e}")
        logger.error(f"ValueError occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.exception(f"An error occurred: {e}")
    return None
