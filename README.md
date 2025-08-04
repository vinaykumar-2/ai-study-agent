```markdown
# 📚 AI Study Agent – Learn from PDFs & Web

An AI-powered study assistant that helps students learn faster by answering questions from **uploaded PDFs** and performing **live web searches** when needed.  
Built using **Streamlit, LangChain, Google Gemini API, Groq LLM, FAISS, and DuckDuckGo Search** for an interactive and intelligent study experience.

---

## 🚀 Features

- 📄 **Ask Questions from PDFs** – Upload study material and query it instantly.  
- 🌍 **Live Web Search** – Uses DuckDuckGo for up-to-date answers when the PDF can’t help.  
- 🧠 **Persistent Conversation Memory** – Stores chat history in JSON format for better context.  
- ⚡ **Fast & Accurate Responses** – Powered by Gemini 1.5 Flash & Groq LLM via LangChain.  
- 🔍 **Hybrid Knowledge Retrieval** – Combines PDF-based RAG and real-time search.  

---

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit  
- **LLM Integration:** LangChain + Gemini 1.5 Flash + Groq LLM  
- **Vector Search:** FAISS  
- **PDF Processing:** PyMuPDF  
- **Web Search:** DuckDuckGo Search API  
- **Memory:** Custom JSON-based storage (`user_history.json`)

---

## ⚙️ Installation

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/vinaykumar-2/ai-study-agent.git
cd ai-study-agent
```

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Set up environment variables**
Create a `.env` file in the root folder and add your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶️ Usage

Run the app:

```bash
streamlit run main.py
```

---

## 📌 Example Workflow

1. Upload a **PDF** (textbook, notes, research paper).
2. Ask a question — the AI finds the answer **from the PDF**.
3. If the answer is missing in the PDF, it automatically performs a **web search**.
4. Continue chatting — the AI remembers your previous questions using JSON-based memory.

---

## 🎯 [🚀 Live Demo – Try it Now](https://ai-study-agent-nkxjeqkxdmv3jpaftuzbdw.streamlit.app/)


## 💡 Future Improvements

* 📹 YouTube video recommendations for learning topics.
* 🎯 Personalized quiz generation.

