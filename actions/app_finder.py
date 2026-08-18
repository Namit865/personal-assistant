import os
from pathlib import Path
import psutil
import subprocess
import json


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

    try:
        cmd = 'powershell -Command "Get-StartApps | ConvertTo-Json"'
        result = subprocess.run(
            cmd, capture_output=True, text=True, shell=True
        )

        if result.returncode == 0 and result.stdout.strip():
            raw_apps = json.loads(result.stdout)

            if isinstance(raw_apps, dict):
                raw_apps = [raw_apps]

            for app in raw_apps:
                name_key = app["Name"].lower()
                app_id = app["AppID"]

                if name_key not in index:
                    if "\\" not in app_id and ":" not in app_id:
                        index[name_key] = f"shell:AppsFolder\\{app_id}"
    except Exception as e:
        print("Error:", e)

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
