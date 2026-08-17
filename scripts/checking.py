import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TIER1_OBJECTS = {
    "app",
    "program",
    "window",
    "script",
    "process",
    "thread",
    "task",
    "terminal",
    "shell",
    "dialog",
    "interface",
    "pipeline",
    "module",
    "tool",
    "daemon",
    "framework",
}

TIER2_KEEP = {
    "exit",
    "quit",
    "goodbye",
    "bye",
    "shutdown",
    "shut",
    "down",
    "off",
    "logoff",
    "log",
    "deactivate",
    "leave",
    "disconnect",
    "assistant",
    "session",
    "yourself",
    "interaction",
    "everything",
    "sleep",
}

to_delete = "exit application frame"

data = json.load(open("data/seed_examples.json"))

to_close_app = {
    "close",
    "terminate",
    "close this program",
    "kill process",
    "get out",
    "abort",
    "cancel execution",
    "close window",
    "stop running",
    "see you later",
    "end the current task",
    "close app completely",
    "kill this app",
    "stop the script",
    "wrap up and close",
    "stop the program safely",
    "close down the interface",
    "get rid of this window",
    "go away for now",
    "minimise and close",
    "power off the tool",
    "exit program execution",
    "stop operations",
    "close execution loop",
    "terminate runner instance",
    "kill active interface",
    "close interactive shell",
    "quit background thread",
    "exit runtime pipeline",
    "shut off system process",
    "could you close out now",
    "i need to close this",
    "kill the running assistant module",
    "close window console framework",
    "stop software execution pathway",
    "shut off device controller framework",
    "terminate standard operation cycle completely",
    "close client portal connection frame",
    "exit local host daemon execution",
    "abort process worker thread immediately",
    "kill current task thread allocation",
    "close system dialog box window",
    "quit terminal command processing framework",
    "shut down active interface panels",
    "end conversation thread container environment",
    "terminate local worker instance safely",
    "close primary system running framework",
    "exit device management driver terminal",
    "quit current automation script session",
    "terminate ongoing software thread loops",
    "close active operation screen instance",
    "quit the application",
}

fresh = json.load(open("data/seed_examples.json"))
print(Counter(ex["intent"] for ex in fresh))
