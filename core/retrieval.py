import os
from tavily import TavilyClient


def fetch_answer(question):
    api_key = os.environ.get("TAVILY_API_KEY")

    if api_key is None:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    else:
        pass

    client = TavilyClient(api_key=api_key)

    response = client.search(question,max_results=3,include_answer=True)

    return response.get("answer")


def research_answer(question):
    queries = [
        question,
        question + "explained simply",
        "what is " + question,
    ]

    best = ""
    for q in queries:
        ans = fetch_answer(q)
        if ans and len(ans) > len(best):
            best = ans
        
    return best if best else "I couldn't find the information you asked for"