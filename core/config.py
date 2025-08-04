import os
from dotenv import load_dotenv

load_dotenv()

# API keys 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")


# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Directory Paths
PDF_DIR="data/pdfs"
VECTOR_DB_DIR="data/faiss_db"
USER_HISTORY_PATH = os.path.join(BASE_DIR, "data", "user_history.json")
