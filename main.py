import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from tools.pdf_utils import load_and_split_pdfs
from core.agent import run_agent
from core.config import PDF_DIR,USER_HISTORY_PATH
from core.rag import create_vector_store,load_vector_store,get_retriever
from core.memory import load_history

# Load environment variable
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="📚 AI Study Agent", layout="wide")

# Load chat history from file
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

st.title("📚 AI Study Agent – Learn from PDFs & Web")

# Side-bar for PDFs
st.sidebar.header("📂 Upload PDFs")

# Clear Chat History Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    if os.path.exists(USER_HISTORY_PATH):
        os.remove(USER_HISTORY_PATH)
    st.rerun()

# PDF Upload
uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Reset PDF directory
    shutil.rmtree(PDF_DIR, ignore_errors=True)
    os.makedirs(PDF_DIR, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(PDF_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded")

    with st.spinner("Processing PDFs and updating vector store..."):
        docs = load_and_split_pdfs(PDF_DIR)
        create_vector_store(docs)

        # Reload retriever
        from core import rag
        rag.vectorstore = load_vector_store()
        rag.retriever = get_retriever(rag.vectorstore)

        # Check FAISS exists in cloud
        faiss_path = "data/faiss_db/index.faiss"
        if os.path.exists(faiss_path):
            st.sidebar.success("Vector store ready for PDF Q&A")
        else:
            st.sidebar.error("Vector store not created — PDF answers may not work.")

# Display Chat Messages
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Type your question here...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner(" Thinking..."):
        answer, source = run_agent(user_input)

    answer = answer or "I couldn't find an answer"
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # 📌 Show source in sidebar
    st.sidebar.info(f"📌 Source: {source}")