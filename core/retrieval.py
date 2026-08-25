import os
from tavily import TavilyClient


def fetch_passages(question, max_results=3):

    api_key = os.environ.get("TAVILY_API_KEY")

    if api_key is None:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    else:
        pass

    client = TavilyClient(api_key=api_key)

    response = client.search(question, max_results=max_results)

    results = response["results"]

    passages = []

    for r in results:
        passages.append(r["content"])

    return passages


def clean_passage(content):
    clean_content = content.replace("[...]", "")
    clean_content = clean_content[:1000]

    start = clean_content.find("File:")
    if start != -1:
        end = clean_content.find(".wav", start)
        if end != -1:
            clean_content = clean_content[:start] + clean_content[end + 4 :]

    clean_content = clean_content.replace("[...]", "")
    clean_content = clean_content[:1000]

    return clean_content