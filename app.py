
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from db_utils import store_document_info, get_all_documents
from pdf_utils import extract_text_from_pdf, split_pdf_text
from vector_store_utils import create_vector_store_from_text
from dotenv import load_dotenv
from langchain.llms import OpenAI  # Use LangChain's LLM wrapper
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)

# OpenAI setup using LangChain wrapper
llm = ChatOpenAI(api_key=OPENAI_API_KEY,model="gpt-4o-mini",temperature=0.7)

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
        docs = retriever.get_relevant_documents(user_query)

        # Log the retrieved chunks for transparency
        print("\n<--------Retrieved Relevant Chunks-------->")
        # Limit the number of retrieved chunks to include in the context
        max_chunks = 2  # Only use the top 2 retrieved chunks to provide a more focused context
        retrieved_chunks = ""
        for idx, doc in enumerate(docs[:max_chunks]):
            chunk_text = doc.page_content[:500]
            retrieved_chunks += f"\nChunk {idx + 1}: {chunk_text}"


        # Format chat history explicitly for LLM to understand the context
        formatted_chat_history = ""
        for entry in conversation_history:
            formatted_chat_history += f"Human: {entry['query']}\nAssistant: {entry['response']}\n"

        # Add the current question to formatted history
        formatted_chat_history += f"Human: {user_query}\n"

        # Define the prompt for the LLM, including the retrieved context
# Define a shorter prompt template
        prompt_template = """
        You are an AI assistant that helps answer questions based on provided PDF content and user context.
        Here is the conversation so far:
        {chat_history}

        Here are the relevant document excerpts that may help answer the question:
        {context}

        Please provide a brief and concise response to the user's latest question.
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

