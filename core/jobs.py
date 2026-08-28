from datetime import datetime
import time
from queue import Queue, Empty
import threading
from memory.reminder import due_reminders
from core.notify import show_toast

job_results = Queue()

def run_research(text):
    from core.retrieval import research_answer

    try:
        ans = research_answer(text)
        job_results.put(ans or "I couldn't find the information you asked for.")
    except Exception as e:
        job_results.put(f"Research failed: {e}")


def start_research(text):
    threading.Thread(target=run_research, args=(text,), daemon=True).start()


def _reminder_loop():
    while True:
        for msg in due_reminders(datetime.now()):
            show_toast("Reminder",msg)
            job_results.put(f"Reminder: {msg}")
        time.sleep(5)


def start_reminder_loop():
    threading.Thread(target=_reminder_loop, daemon=True).start()


def start_job_drain(print_and_speak):
    def drain():
        while True:
            try:
                msg = job_results.get(timeout=1)
            except Empty:
                continue
            print(msg)
            print_and_speak(msg)

    threading.Thread(target=drain, daemon=True).start()
