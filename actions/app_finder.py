import os
from pathlib import Path
import psutil


def build_app_index():

    APPDATA_PATH = Path(os.environ["APPDATA"]).joinpath(
        "Microsoft/Windows/Start Menu/Programs"
    )
    PROGRAMDATA_PATH = Path(os.environ["PROGRAMDATA"]).joinpath(
        "Microsoft/Windows/Start Menu/Programs"
    )

    index = {}
    already_exists = 0

    for folder in [APPDATA_PATH, PROGRAMDATA_PATH]:
        if folder.exists():
            for file in folder.rglob("*.lnk"):
                if file.stem.lower() not in index:
                    index[file.stem.lower()] = str(file)
                else:
                    already_exists += 1
    return index


def find_app_path(app_name, index):
    if app_name in index:
        return index[app_name]
    else:
        index = build_app_index()

    matches = []
    prefix_matches = []

    for key in index:
        if app_name in key:
            if key.startswith(app_name):
                prefix_matches.append(key)
            else:
                matches.append(key)

    pool = prefix_matches if prefix_matches else matches

    if not pool:
        return None
    else:
        best_key = min(pool, key=len)
        return index[best_key]


def list_running_processes():
    processes = []
    for proc in psutil.process_iter(["pid", "name"]):
        processes.append((proc.info["pid"], proc.info["name"]))
    return processes


def find_matching_processes(app_name, processes):
    matches = []
    for pid, name in processes:
        clean_name = name.lower().removesuffix(".exe")

        if app_name in clean_name:
            matches.append((pid, name))

    return matches


processes = list_running_processes()
matches = find_matching_processes("claude", processes)
print(matches)
