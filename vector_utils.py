
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
import os

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
        return vector_store
    except ValueError as e:
        print(f"ValueError occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None
