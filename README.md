Chat-With-PDFs-RAG-LLM
An end-to-end application that allows users to chat with PDF documents using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) through LangChain. The system can analyze uploaded PDF documents, retrieve relevant sections, and provide answers to user queries in natural language.

Features
PDF Upload: Upload multiple PDF files.
RAG-based Querying: Uses Retrieval-Augmented Generation to find and answer questions based on PDF content.
Chat Interface: Interactive UI that allows users to ask questions and get responses based on the uploaded PDF content.
Context-aware Responses: The system keeps track of the conversation and provides context-aware answers.

Installation
Prerequisites
Python 3.8 or higher
Node.js for the frontend
pip for Python package management

Backend Setup
Clone the repository:  git clone https://github.com/omkars20/Chat-With-PDFs-RAG-LLM-.git
cd Chat-With-PDFs-RAG-LLM/backend


python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

Set environment variables:

Create a .env file in the backend directory with the following:


OPENAI_API_KEY=your_openai_api_key_here
MYSQL_HOST=your_mysql_host
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=your_mysql_database
Run the backend server:

python app.py

Frontend Setup
Navigate to the frontend directory:

cd ../frontend
install dependencies
npm install
run the development server: npm start


Usage
Upload a PDF using the frontend interface.
Ask questions related to the content of the uploaded PDF.
View responses that the AI model generates based on the relevant portions of the PDF document.
Example
Upload a PDF document, for example, "Project Documentation.pdf".
Ask: "What is the main objective of the project?"
The system will respond with the relevant portion from the PDF document.


Project Structure
Chat-With-PDFs-RAG-LLM/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── pdf_utils.py
│   ├── vector_store_utils.py
│   └── ...
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── ...
└── README.md

backend/: Python Flask backend that handles PDF processing, vector storage, and LLM-based responses.
frontend/: React-based frontend that allows users to interact with the system through a web interface.


Technologies Used
Flask: Backend framework for handling requests and running the API.
LangChain: Library used for creating the retrieval chain with LLM integration.
FAISS: Vector storage solution to support document embedding retrieval.
React: Frontend UI for user interaction.
OpenAI API: For generating natural language responses based on user queries.







