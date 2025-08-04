from langchain_core.tools import Tool
from core.rag import load_vector_store, get_compressed_retriever

# Load FAISS vector store
vectorstore = load_vector_store()
retriever = get_compressed_retriever(vectorstore) if vectorstore else None

# PDF Search Tool function
def search_pdfs(query: str) -> str:
    """Search relevant content from uploaded PDFs based on user query."""
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        return f"[ERROR] Could not search PDFs: {e}"

    if not docs:
        return "No relevant information found in the uploaded PDFs."
    
    return "\n\n".join([doc.page_content for doc in docs[:3]])

# LangChain Tool
pdf_search_tool = Tool(
    name="PDF Search Tool",
    func=search_pdfs,
    description="Useful for answering questions from the uploaded PDFs."
)
