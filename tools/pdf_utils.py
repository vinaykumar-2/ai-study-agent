import os
from typing import List
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def load_and_split_pdfs(pdf_dir: str) -> List[Document]:
    """Load all PDFs in a folder, extract and chunk them into Document objects."""
    documents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            path = os.path.join(pdf_dir, file_name)
            raw_text = extract_text_from_pdf(path)
            chunks = splitter.create_documents([raw_text])
            for chunk in chunks:
                chunk.metadata["source"] = file_name
            documents.extend(chunks)

    return documents
