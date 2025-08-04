import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.pdf_utils import load_and_split_pdfs
from core.rag import create_vector_store
from core.config import PDF_DIR

def build_vector_store():
    # Load and split PDFs into chunks
    pdf_docs = load_and_split_pdfs(PDF_DIR)
    print(f"Loaded {len(pdf_docs)} chunks from PDFs.")

    # Create and save FAISS vector store
    create_vector_store(pdf_docs)
    print("FAISS vector store created and saved at data/faiss_db/")
