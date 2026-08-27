from queue import Queue
import threading

job_results = Queue()

def run_research(text):
    from core.retrieval import research_answer

    ans = research_answer(text)

    job_results.put(ans or "I couldn't find the information you asked for")

def start_research(text):
    t = threading.Thread(target=run_research,args = (text,),daemon = True)
    t.start()