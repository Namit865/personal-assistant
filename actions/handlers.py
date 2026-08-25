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


def close_app(text, content):
    app_name = extract_app_name(text)
    processes = list_running_processes()

    matches = find_matching_processes(app_name, processes)

    if not matches:
        print(f"No running app called '{app_name}'")
    else:
        for pid, name in matches:
            try:
                psutil.Process(pid).terminate()
                print(f"closed {name} (pid {pid})")
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"couldn't close the {name} (pid {pid}): {e}")

def knowledge_query(text, content):
    ans = fetch_answer(text)
    if ans:
        print(ans)
    else:
        print("I couldn't find the information you asked for.")

def open_app(text, context):
    app_name = extract_app_name(text)
    result = find_app_path(app_name, context["app_index"])
    if result:
        os.startfile(result)
    else:
        print(f"couldn't find an app called '{app_name}'")


def web_search(text, context):
    print(f"[web search] {text}")
    webbrowser.open(f"https://www.google.com/search?q={quote_plus(text)}")


def create_note(text, context):
    notes = ROOT / "notes.txt"
    with open(notes, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {text}\n")
    print(f"note saved to {notes}")


def system_status(text, context):
    text = text.lower()
    print(f"[system_status] {text}")
    now = datetime.now()
    print("Time:", now.strftime("%I:%M:%S %p"))
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
            print(
                f"Battery Percent: {battery.percent} | Charging Status: {'Charging' if battery.power_plugged else 'On Battery'}"
            )
        else:
            print("No Battery Detected.")

    if "cpu" in text:
        print(f"Cpu Usage: {cpu}%")

    if "ram" in text:
        print(
            f"RAM: {(mem.used / 1024 ** 3):.2f}/{(mem.total / 1024 ** 3):.2f} GB ({mem.percent}%)"
        )

    if "current mode" in text or "power plan" in text:
        print("Power Mode:", mode)

    if "disk" in text:
        print(
            f"Disk usage: {(disk.used / 1024 ** 3):.2f}/{(disk.total / 1024 ** 3):.2f} GB ({disk.percent}% Used)"
        )

    if "memory" in text:
        processes = []

        for proc in psutil.process_iter(["name", "memory_info"]):

            processes.append((proc.info["name"], proc.info["memory_info"].rss))

        sorted_processes = sorted(processes, key=lambda p: p[1], reverse=True)

        top_processes = sorted_processes[:5]

        for name, mem_bytes in top_processes:
            mem_bytes = mem_bytes / 1024**2
            print(f"Top Processes: {name} - {mem_bytes:.1f} MB")

    if "system status" in text or "check status" in text:
        if battery:
            print(
                f"Battery Percent: {battery.percent} | Charging Status: {'Charging' if battery.power_plugged else 'On Battery'}"
            )
        else:
            print("No Battery Detected.")

        print(f"Cpu Usage: {cpu}%")

        print(
            f"RAM: {(mem.used / 1024 ** 3):.2f}/{(mem.total / 1024 ** 3):.2f} GB ({mem.percent}%)"
        )
        print("Power Mode:", mode)

        print(
            f"Disk usage: {(disk.used / 1024 ** 3):.2f}/{(disk.total / 1024 ** 3):.2f} GB ({disk.percent}% Used)"
        )


def exit_assistant(text, context):
    print(f"[exit] {text}")
    print("Shutting down. Goodbye!")


def unknown(text, context):
    print("I didn't understand that.")

def extract_app_name(text):
    words = tokenize(text)
    kept = []

    for word in words:
        if word not in FILLER:
            kept.append(word)

    result = " ".join(kept)

    return result
