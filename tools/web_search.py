from duckduckgo_search import DDGS

def web_search_duckduckgo(query: str, max_results: int = 3):
    """Search the web using DuckDuckGo and return top results as list of dicts."""
    if not query.strip():
        return []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results,region="in-en"))
        return results or  []
    except Exception as e:
        print(f"[ERROR] DuckDuckGo search failed: {e}")
        return []