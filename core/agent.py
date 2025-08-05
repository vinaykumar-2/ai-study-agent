from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from core.config import GEMINI_API_KEY,GROQ_API_KEY
from core.memory import add_message,load_history
#from core.rag import load_vector_store, get_retriever
from core import rag
from tools.web_search import web_search_duckduckgo

load_dotenv()

# Gemini for PDF RAG
gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7
)

# Groq model for web summarization and general chat
groq_llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# Load retriever
#vectorstore = load_vector_store()
#retriever = get_retriever(vectorstore) if vectorstore else None

# Prompt Template For RAG
rag_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant. Use the context from PDFs to answer the question naturally.
If the answer is not in the PDFs, say "I couldn't find that in your uploaded PDFs."

Context:
{context}

Question: {question}
""")

# Node: PDF RAG
def pdf_rag_node(state):
    question = state["question"]

    vectorstore = rag.load_vector_store()
    retriever = rag.get_retriever(vectorstore) if vectorstore else None

    if not retriever:
        return {**state,"result": None, "source": "no_pdf"}

    docs = retriever.invoke(question)
    if not docs:
        return {**state,"result": None, "source": "no_pdf_match"}

    
    context = "\n\n".join(doc.page_content for doc in docs[:3])
    prompt = rag_prompt.format(context=context, question=question)
    answer = gemini_llm.invoke([HumanMessage(content=prompt)])
    text = getattr(answer, "content", str(answer))
    return {**state,"result": text , "source": "pdf"}

# Node: Web Search (Groq Summarization)
def web_search_node(state):
    question = state["question"]
    results = web_search_duckduckgo(question, max_results=4)

    if not results:
        return {**state, "result": "No relevant web results found.", "source": "web"}

    # Extract text snippets from DuckDuckGo results
    snippets = []
    for r in results:
        snippet = r.get("body") or r.get("snippet") or ""
        title = r.get("title", "")
        if snippet:
            snippets.append(f"{title}: {snippet}")

    if not snippets:
        return {**state, "result": "No relevant content found to summarize.", "source": "web"}

    combined_text = "\n\n".join(snippets)

    # Summarize for students using Groq
    summary_prompt = f"""
    You are an academic AI tutor for students.

    Summarize the following web search results into a clear, concise, and student-friendly answer.

    - Include the exact date of the event if available in the search results.
    - If the date is not mentioned, clearly say "Date not mentioned in the sources."
    - Focus only on factual updates relevant to the question.
    - Avoid unnecessary disclaimers like "I do not have access to real-time information."
    - Do not include hyperlinks.

    Query: "{question}
    Search Results:
    {combined_text}
    """

    answer = groq_llm.invoke([HumanMessage(content=summary_prompt)])
    text = getattr(answer, "content", str(answer))

    return {
        **state,
        "result": text.strip(),
        "source": "web"
    }

# Node: Groq with Memory
def groq_memory_node(state):
    history = load_history()
    question = state["question"]

    # Convert memory to readable conversation text
    history_text = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Prompt with memory context
    prompt = f"""
    You are a helpful AI tutor. 
    Here is the conversation so far:

    {history_text}

    Now answer the latest question:

    User: {question}
    Assistant:
    """

    answer = groq_llm.invoke([HumanMessage(content=prompt)])
    text = getattr(answer, "content", str(answer))
    return {**state, "result": text, "source": "groq"}
    


# Node: Conditional Router
def router(state):
    question_lower = state["question"].lower()

    # Academic-related keywords (always trigger web search)
    academic_keywords = [
        "exam", "admission", "syllabus", "result",
        "neet", "ssc", "jee", "cbse", "education",
        "science", "technology", "research", "study",
        "latest", "update", "current affairs", "exam date", "admit card"
    ]
    
    if any(k in question_lower for k in academic_keywords):
        return "web_search"
    
    # If no PDF uploaded or no relevant match -> use Groq
    if state["source"] in ["no_pdf", "no_pdf_match"]:
        return "groq"
    
    # If PDF answer is empty or "couldn't find" -> use Groq
    if state["source"] == "pdf" and (
        not state["result"] 
        or "i couldn't find" in state["result"].lower()
    ):
        return "groq"

    # If Gemini failed -> try web search
    if state["source"] == "groq" and (
        not state["result"] 
        or "i don't know" in state["result"].lower()
    ):
        return "web_search"

    # Otherwise, end the flow
    return "end"


# LangGraph Flow
graph = StateGraph(dict)
graph.add_node("pdf_rag", pdf_rag_node)
graph.add_node("groq", groq_memory_node)
graph.add_node("web_search", web_search_node)

graph.add_conditional_edges("pdf_rag", router)
graph.add_conditional_edges("groq", router)

graph.set_entry_point("pdf_rag")
graph.add_edge("web_search", END)
graph.add_edge("groq", END)

flow = graph.compile()

# Main Agent function
def run_agent(question: str):
    add_message("user",question)
    initial_state = {"question": question}
    result = flow.invoke(initial_state)

    # Save response to memory
    add_message("assistant",result.get("result" ,""))

    print(f"[DEBUG] Source: {result['source']} | Question: {question}")
    return result.get("result",""), result.get("source","unknown")