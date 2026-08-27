import json
from config import PROFILE_FILE

MAX = 5

def load_profile():
    if PROFILE_FILE.exists():
        return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    return {"recent_questions" : [],"last_browser":None}

def save_profile(profile):
    PROFILE_FILE.write_text(json.dumps(profile,indent=2),encoding="utf-8")

def remember_question(text):
    profile = load_profile()
    qs = profile.get("recent_questions",[])
    qs.append(text)

    profile["recent_questions"] = qs[-MAX:]
    save_profile(profile)

def remember_browser(url,query=""):
    profile = load_profile()
    profile["last_browser"] = {"url":url,"query":query}
    save_profile(profile)

def last_browser():
    return load_profile().get("last_browser")