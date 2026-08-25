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
