import pymysql
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# MySQL connection setup using environment variables
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Function to establish a MySQL connection using pymysql
def get_db_connection():
    try:
        db = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE
        )
        return db
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
        return None

# Function to store document information in the database
def store_document_info(username, doc_name, vector_path):
    db = get_db_connection()
    if db:
        cursor = db.cursor()
        insert_query = """
            INSERT INTO pdf_chatbot_document (username, doc_name, vector_path, created_at, updated_at)
            VALUES (%s, %s, %s, NOW(), NOW())
        """
        cursor.execute(insert_query, (username, doc_name, vector_path))
        db.commit()
        cursor.close()
        db.close()

# Function to get all documents for listing
def get_all_documents():
    db = get_db_connection()
    if db:
        cursor = db.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT id, username, doc_name, vector_path, created_at FROM pdf_chatbot_document")
        documents = cursor.fetchall()
        cursor.close()
        db.close()
        return documents
    return []

# Test the connection independently
if __name__ == "__main__":
    db = get_db_connection()
    if db:
        print("Database connection was successful.")
    else:
        print("Failed to connect to the database.")

