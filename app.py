

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from db_utils import store_document_info, get_all_documents
from pdf_utils import extract_text_from_pdf, split_pdf_text
from vector_utils import create_vector_store_from_text
from dotenv import load_dotenv

# Import statements corrected to handle deprecations:
from langchain_openai import OpenAIEmbeddings  # Corrected import for OpenAIEmbeddings
from langchain_community.llms import OpenAI  # Updated import for the LLM wrapper
from langchain_community.vectorstores import FAISS  # Corrected import for FAISS vector store
from langchain.chains import ConversationalRetrievalChain  # No change needed here
from langchain.prompts import PromptTemplate  # No change needed here
from langchain_openai import ChatOpenAI  # Corrected import for ChatOpenAI
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)

# OpenAI setup using LangChain wrapper
llm = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini",temperature=0.5)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        username = request.form.get('username')
        pdf_file = request.files['pdf']
        pdf_name = pdf_file.filename

        if not username or not pdf_file:
            return jsonify({"error": "Missing username or PDF file"}), 400

        # Save PDF temporarily
        pdf_path = os.path.join('uploads', pdf_name)
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        pdf_file.save(pdf_path)

        # Extract text and split into chunks
        pdf_text = extract_text_from_pdf(pdf_path)
        pdf_chunks = split_pdf_text(pdf_text)

        # Create vector embeddings and save vector store
        vector_store_path = f"models/{username}_{pdf_name}_vector_store"
        create_vector_store_from_text(pdf_chunks, OPENAI_API_KEY, vector_store_path)

        # Store document information in the database
        store_document_info(username, pdf_name, vector_store_path)

        return jsonify({"message": "PDF uploaded and processed successfully."})
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Get Documents Route
@app.route('/get_documents', methods=['GET'])
def get_documents():
    documents = get_all_documents()
    return jsonify({"documents": documents})



# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         # Get user inputs from the request
#         user_id = request.json.get('user_id')
#         pdf_name = request.json.get('pdf_name')
#         user_query = request.json.get('query', '')
#         conversation_history = request.json.get('history', [])

#         # Retrieve vector path from the database
#         documents = get_all_documents()
#         vector_store_path = None
#         for doc in documents:
#             if doc['username'] == user_id and doc['doc_name'] == pdf_name:
#                 vector_store_path = doc['vector_path']
#                 break

#         if vector_store_path is None:
#             return jsonify({"response": "Document not found."}), 404

#         # Load vector store and run retrieval
#         embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#         vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

#         # Create a retrieval chain using LangChain
#         retriever = vector_store.as_retriever()

#         # Retrieve relevant documents using retriever
#         docs = retriever.get_relevant_documents(user_query)

#         # Log the retrieved chunks for transparency
#         print("\n<--------Retrieved Relevant Chunks-------->")
#         # Limit the number of retrieved chunks to include in the context
#         max_chunks = 5  # Only use the top 2 retrieved chunks to provide a more focused context
#         retrieved_chunks = ""
#         for idx, doc in enumerate(docs[:max_chunks]):
#             chunk_text = doc.page_content[:500]
#             retrieved_chunks += f"\nChunk {idx + 1}: {chunk_text}"


#         # Format chat history explicitly for LLM to understand the context
#         formatted_chat_history = ""
#         for entry in conversation_history:
#             formatted_chat_history += f"Human: {entry['query']}\nAssistant: {entry['response']}\n"

#         # Add the current question to formatted history
#         formatted_chat_history += f"Human: {user_query}\n"

#         # Define the prompt for the LLM, including the retrieved context
#         prompt_template = """
#         You are an AI assistant that answers questions based on the provided context from the document. 
#         Here is the current conversation:
#         {chat_history}

#         Relevant information from the document:
#         {context}

#         Please provide a concise response to the user's question.
#         """

        
#         # Format the prompt for LLM input
#         formatted_prompt = prompt_template.format(
#             chat_history=formatted_chat_history,
#             context=retrieved_chunks
#         )

#         # Run the LLM using `invoke`
#         response = llm.invoke(formatted_prompt)

#         # Extract the content from the response
#         if hasattr(response, "content"):
#             answer = response.content.strip()
#         else:
#             answer = str(response).strip()

#         # Add the new question and answer to the conversation history
#         new_entry = {"query": user_query, "response": answer}
#         conversation_history.append(new_entry)

#         # Log retrieval details for better debugging
#         print("\n<--------Retrieval and LLM Interaction Details-------->")
#         print({
#             "question": user_query,
#             "chat_history": formatted_chat_history,
#             "retrieved_context": retrieved_chunks,
#             "answer": answer
#         })

#         # Serialize the relevant source documents to send as part of the response
#         serialized_documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

#         return jsonify({"response": answer, "sources": serialized_documents})

#     except Exception as e:
#         print(f"Error processing query: {e}")
#         return jsonify({"error": "Internal server error", "details": str(e)}), 500




@app.route('/query', methods=['POST'])
def query():
    try:
        # Get user inputs from the request
        user_id = request.json.get('user_id')
        pdf_name = request.json.get('pdf_name')
        user_query = request.json.get('query', '')
        conversation_history = request.json.get('history', [])

        # Retrieve vector path from the database
        documents = get_all_documents()
        vector_store_path = None
        for doc in documents:
            if doc['username'] == user_id and doc['doc_name'] == pdf_name:
                vector_store_path = doc['vector_path']
                break

        if vector_store_path is None:
            return jsonify({"response": "Document not found."}), 404

        # Load vector store and run retrieval
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

        # Create a retrieval chain using LangChain
        retriever = vector_store.as_retriever()

        # Retrieve relevant documents using retriever
        # docs = retriever.get_relevant_documents(user_query)
        docs = retriever.invoke(user_query)

        # Log the retrieved chunks for transparency
        print("\n<--------Retrieved Relevant Chunks-------->")
        max_chunks = 5  # Increase to include more relevant information
        retrieved_chunks = ""
        for idx, doc in enumerate(docs[:max_chunks]):
            chunk_text = doc.page_content[:800]
            retrieved_chunks += f"\nChunk {idx + 1}: {chunk_text}"

            print(retrieved_chunks)

        # Format chat history explicitly for LLM to understand the context
        formatted_chat_history = ""
        for entry in conversation_history:
            formatted_chat_history += f"Human: {entry['query']}\nAssistant: {entry['response']}\n"

        # Add the current question to formatted history
        formatted_chat_history += f"Human: {user_query}\n"

        # Define the prompt for the LLM, including the retrieved context
        prompt_template = """
        You are an AI assistant that provides answers strictly based on the provided context from the document.
        The following is the conversation so far, along with retrieved information that should help answer the user's question.

        Here is the conversation history:
        {chat_history}

        Relevant information retrieved from the document:
        {context}

        Based on the above context, answer the user's current question as specifically and accurately as possible.
        """

        # Format the prompt for LLM input
        formatted_prompt = prompt_template.format(
            chat_history=formatted_chat_history,
            context=retrieved_chunks
        )

        # Run the LLM using `invoke`
        response = llm.invoke(formatted_prompt)

        # Extract the content from the response
        if hasattr(response, "content"):
            answer = response.content.strip()
        else:
            answer = str(response).strip()

        # Add the new question and answer to the conversation history
        new_entry = {"query": user_query, "response": answer}
        conversation_history.append(new_entry)

        # Log retrieval details for better debugging
        print("\n<--------Retrieval and LLM Interaction Details-------->")
        print({
            "question": user_query,
            "chat_history": formatted_chat_history,
            "retrieved_context": retrieved_chunks,
            "answer": answer
        })

        # Serialize the relevant source documents to send as part of the response
        serialized_documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        return jsonify({"response": answer, "sources": serialized_documents})

    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


# Main entry point for the application
if __name__ == "__main__":
    app.run(debug=True)


