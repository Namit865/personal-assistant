from actions.handlers import (
    open_app,
    web_search,
    create_note,
    system_status,
    exit_assistant,
    unknown,
)

REGISTRY = {
    "open_app": open_app,
    "web_search": web_search,
    "create_note": create_note,
    "system_status": system_status,
    "exit": exit_assistant,
    "unknown": unknown,
}
