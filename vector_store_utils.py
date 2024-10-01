from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS

# Generate vector embeddings and save FAISS index
def create_vector_store_from_text(chunks, api_key, save_path):
    # Using the updated version of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(save_path)
    return vector_store

