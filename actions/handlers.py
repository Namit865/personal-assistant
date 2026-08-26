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
from core.retrieval import fetch_answer

FILLER = {
    "open",
    "launch",
    "start",
    "run",
    "up",
    "fire",
    "the",
    "my",
    "a",
    "an",
    "please",
    "can",
    "you",
    "for",
    "me",
    "app",
    "application",
    "program",
    "close",
    "kill",
    "quit",
    "exit",
    "stop",
    "shut",
    "down",
    "terminate",
    "out",
    "of",
    "could",
    "would",
    "mind",
    "closing",
    "force",
    "window",
    "need",
    "i",
    "to",
}

SEARCH_FILLER = FILLER | {
    "search","google","web","look","find","browse","search for","google for","web for","for"
}

NOTE_FILLERS = FILLER | {
    "note",
    "notes",
    "remember",
    "remind",
    "write",
    "down",
    "take",
    "make",
    "add",
    "create",
    "new",
    "saying",
    "about",
    "that",
}


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
    ans = fetch_answer(text)
    if ans:
        return ans
    else:
        return "I couldn't find the information you asked for."

def open_app(text, context):
    app_name = extract_app_name(text)
    result = find_app_path(app_name, context["app_index"])
    if result:
        os.startfile(result)
        return f"opened {app_name}"
    else:
        return f"couldn't find an app called '{app_name}'"


def web_search(text, context):
    clean_query = extract_search_query(text)
    words = clean_query.split()

    if "youtube" in words:
        query = " ".join(w for w in words if w not in ("youtube","in","on"))
        webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")

        return f"searching for {query} on youtube"
    
    webbrowser.open(f"https://www.google.com/search?q={quote_plus(clean_query)}")

    return f"opened search for {clean_query}"

def create_note(text, context):
    body = extract_note_body(text)
    notes = ROOT / "notes.txt"
    with open(notes, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {body}\n")
    return f"note saved to {body}"


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