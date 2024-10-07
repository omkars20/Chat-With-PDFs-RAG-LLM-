import pymysql
from dotenv import load_dotenv
import os
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv()

# MySQL connection setup using environment variables
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

def get_db_connection():
    """
    Establish a MySQL connection using pymysql.

    Returns:
        Connection: pymysql database connection object.
    """
    try:
        db = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info("Database connection established.")
        return db
    except pymysql.MySQLError as e:
        logger.exception(f"Database connection failed: {e}")
        raise

def store_document_info(username, doc_name, vector_path):
    """
    Store document information in the database.

    Args:
        username (str): Username of the user.
        doc_name (str): Name of the document.
        vector_path (str): Path to the vector store.
    """
    try:
        db = get_db_connection()
        with db.cursor() as cursor:
            insert_query = """
                INSERT INTO pdf_chatbot_document (username, doc_name, vector_path, created_at, updated_at)
                VALUES (%s, %s, %s, NOW(), NOW())
            """
            cursor.execute(insert_query, (username, doc_name, vector_path))
            db.commit()
            logger.info(f"Document info stored for user {username}, document {doc_name}.")
    except Exception as e:
        logger.exception(f"Failed to store document info: {e}")
        raise
    finally:
        if db:
            db.close()
            logger.info("Database connection closed.")

def get_all_documents():
    """
    Retrieve all documents from the database.

    Returns:
        list: List of documents.
    """
    try:
        db = get_db_connection()
        with db.cursor() as cursor:
            cursor.execute("SELECT id, username, doc_name, vector_path, created_at FROM pdf_chatbot_document")
            documents = cursor.fetchall()
            logger.info(f"Retrieved {len(documents)} documents from the database.")
            return documents
    except Exception as e:
        logger.exception(f"Failed to retrieve documents: {e}")
        raise
    finally:
        if db:
            db.close()
            logger.info("Database connection closed.")

