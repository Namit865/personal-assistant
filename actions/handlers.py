import dateparser
from dateparser.search import search_dates
import webbrowser
from urllib.parse import quote_plus
from datetime import datetime
from config import ROOT
import sys
from pathlib import Path
from actions.app_finder import (
    find_app_path,
    list_running_processes,
    find_matching_processes,
)
import os
import psutil
import subprocess
from datetime import datetime
import json
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.text_utils import tokenize
from core.retrieval import fetch_answer,research_answer
from actions.file_finder import find_files,open_file
from memory.profile import remember_question
from core.jobs import start_research
from memory.profile import remember_browser,last_browser
from memory.reminder import add_reminder

FILLER = {
    "open","launch","start","run","up","fire","the","my","a","an","please",
    "can","you","for","me","app","application","program","close","kill","quit",
    "exit","stop","shut","down","terminate","out","of","could","would","mind",
    "closing","force","window","need","i","to",
}

SEARCH_FILLER = FILLER | {
    "search","google","web","look","find","browse","search for",
    "google for","web for","for","research","find out",
}

FILE_FILLER = FILLER | {
    "file", "files", "document", "documents", "folder",
    "show", "find", "locate", "fetch", "get", "want",
    "where", "from", "desktop", "downloads", "documents",
    "please", "me", "my",
}


NOTE_FILLERS = FILLER | {
    "note","notes","remember","remind","write","down","take",
    "make","add","create","new","saying","about","that",
}

REMINDER_FILLER = NOTE_FILLERS | {
    "set", "timer", "alarm", "schedule", "later", "tomorrow",
    "today", "tonight", "morning", "afternoon", "evening",
}

def open_url(url,context):
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ["LOCALAPPDATA"])
        / "Google/Chrome/Application/chrome.exe",
    ]

    for path in candidates:
        if path.exists():
            subprocess.Popen([str(path),url])
            return True

    subprocess.Popen(["cmd","/c","start","chrome",url],shell=False)
    return True

def close_app(text, content):
    app_name = extract_app_name(text)
    processes = list_running_processes()

    matches = find_matching_processes(app_name, processes)

    if not matches:
        return f"No running app called '{app_name}'"
    else:
        for pid, name in matches:
            try:
                psutil.Process(pid).terminate()
                return f"closed {name} (pid {pid})"
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                return f"couldn't close the {name} (pid {pid}): {e}"

def knowledge_query(text, content):
    remember_question(text)
    start_research(text)
    return "Searching. I'll speak when I have an answer"

def open_app(text, context):
    app_name = extract_app_name(text)
    result = find_app_path(app_name, context["app_index"])
    if result:
        os.startfile(result)
        return f"opened {app_name}"
    else:
        return f"couldn't find an app called '{app_name}'"


def web_search(text, context):
    lower = text.lower()
    if "last" in lower and any(w in lower for w in ("search","browser","tab","page","youtube","google")):
        info = last_browser()
        if not info or not info.get("url"):
            return "No last search saved yet"
        open_url(info["url"],context)
        q = info.get("query") or "last search"
        return f"reopened {q}"
        
    clean_query = extract_search_query(text)
    words = clean_query.split()

    if "youtube" in words:
        query = " ".join(w for w in words if w not in ("youtube", "in", "on"))
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        open_url(url, context)
        remember_browser(url,query if "youtube" in words else clean_query)
        return f"searching for {query} on youtube"

    url = f"https://www.google.com/search?q={quote_plus(clean_query)}"
    open_url(url, context)
    remember_browser(url,clean_query)
    return f"opened search for {clean_query}"

def create_note(text, context):
    body = extract_note_body(text)
    notes = ROOT / "notes.txt"
    with open(notes, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {body}\n")
    return f"note saved to {body}"

def set_reminder(text, context):
    body,when = parse_reminder(text)

    if not when:
        return "I couldn't figure out when to remind you"
    
    add_reminder(body,when)
    return f"Reminder set for {when:%Y-%m-%d %H:%M}: {body}"

def system_status(text, context):
    lines = []
    text = text.lower()
    now = datetime.now()
    lines.append(f"Time: {now.strftime("%I:%M:%S %p")}")
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    battery = psutil.sensors_battery()

    result = subprocess.run(
        ["powercfg", "/getactivescheme"], capture_output=True, text=True
    )
    mode = result.stdout.split("(")[-1]
    mode = mode.split(")")[0]

    if "battery" in text or "charging" in text:
        if battery:
            lines.append(f"Battery Percent: {battery.percent} | Charging Status: {'Charging' if battery.power_plugged else 'On Battery'}")
        else:
            lines.append("No Battery Detected.")

    if "cpu" in text:
        lines.append(f"Cpu Usage: {cpu}%")

    if "ram" in text:
        lines.append(f"RAM: {(mem.used / 1024 ** 3):.2f}/{(mem.total / 1024 ** 3):.2f} GB ({mem.percent}%)")

    if "current mode" in text or "power plan" in text:
        lines.append(f"Power Mode: {mode}")

    if "disk" in text:
        lines.append(f"Disk usage: {(disk.used / 1024 ** 3):.2f}/{(disk.total / 1024 ** 3):.2f} GB ({disk.percent}% Used)")

    if "memory" in text:
        processes = []

        for proc in psutil.process_iter(["name", "memory_info"]):

            processes.append((proc.info["name"], proc.info["memory_info"].rss))

        sorted_processes = sorted(processes, key=lambda p: p[1], reverse=True)

        top_processes = sorted_processes[:5]

        for name, mem_bytes in top_processes:
            mem_bytes = mem_bytes / 1024**2
            lines.append(f"Top Processes: {name} - {mem_bytes:.1f} MB")

    if "system status" in text or "check status" in text:
        if battery:
            lines.append(f"Battery Percent: {battery.percent} | Charging Status: {'Charging' if battery.power_plugged else 'On Battery'}")
        else:
            lines.append("No Battery Detected.")

        lines.append(f"Cpu Usage: {cpu}%")

        lines.append(f"RAM: {(mem.used / 1024 ** 3):.2f}/{(mem.total / 1024 ** 3):.2f} GB ({mem.percent}%)")
        
        lines.append(f"Power Mode: {mode}")

        lines.append(f"Disk usage: {(disk.used / 1024 ** 3):.2f}/{(disk.total / 1024 ** 3):.2f} GB ({disk.percent}% Used)")

    return "\n".join(lines)


def exit_assistant(text, context):
    return "Shutting down. Goodbye!"


def unknown(text, context):
    return "I didn't understand that."

def extract_app_name(text):
    words = tokenize(text)
    kept = []

    for word in words:
        if word not in FILLER:
            kept.append(word)

    result = " ".join(kept)

    return result

def extract_search_query(text):
    words = tokenize(text)
    kept = [w for w in words if w not in SEARCH_FILLER]
    if not kept:
        return text
    
    return " ".join(kept)

def extract_note_body(text):
    words = tokenize(text)
    kept = [w for w in words if w not in NOTE_FILLERS]
    if not kept:
        return text
    
    return " ".join(kept)

def extract_file_name(text):
    words = tokenize(text)
    kept = [w for w in words if w not in FILE_FILLER]
    if not kept:
        return text
    return " ".join(kept)


def open_path(text, context):
    name = extract_file_name(text)
    hits = find_files(name)
    if not hits:
        return f"couldn't find a file like '{name}'"
    path = hits[0]
    open_file(path)
    return f"found {path.name}"


def read_notes(text, context):
    notes = ROOT / "notes.txt"
    if not notes.exists():
        return "No notes found"
    
    lines = notes.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return "You have no notes yet"

    shown = []
    for line in lines[-5:]:
        if " | " in line:
            ts, body = line.split(" | ", 1)
            if body.startswith("[reminder] "):
                body = "reminder: " + body[len("[reminder] "):]
            shown.append(f"{ts} | {body}")
        else:
            shown.append(line)

    return "Your notes: " + "; ".join(shown)

def parse_reminder(text):
    hits = search_dates(text,settings = {"PREFER_DATES_FROM": "future"})
    if not hits:
        return None,None
    
    when_phrase,when_dt = hits[-1]
    body = text.replace(when_phrase,"")
    words = tokenize(body)
    body = " ".join(w for w in words if w not in REMINDER_FILLER)

    return body or "reminder",when_dt