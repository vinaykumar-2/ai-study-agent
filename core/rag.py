import os
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from core.config import GEMINI_API_KEY, VECTOR_DB_DIR

# Gemini Embedding
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

# Create and save FAISS vector store
def create_vector_store(docs: list[Document]):
    if not docs:
        raise ValueError("No text could be extracted from the uploaded PDFs. Please upload a text-based PDF instead of scanned images.")

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(VECTOR_DB_DIR)

# Load vector Store
def load_vector_store():
    if os.path.exists(VECTOR_DB_DIR):
        return FAISS.load_local(VECTOR_DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    else:
        return None
    
# Basic Retriever
def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,
            "k": 3
        }
    )

# Compression Retriever
def get_compressed_retriever(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY
    )
    compressor = LLMChainExtractor.from_llm(llm)
    base_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,
            "k": 3
        }
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    return retriever
