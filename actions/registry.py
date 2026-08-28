from actions.handlers import (
    open_app,
    web_search,
    create_note,
    system_status,
    exit_assistant,
    unknown,
    close_app,
    knowledge_query,
    open_path,
    read_notes,
)

REGISTRY = {
    "close_app": close_app,
    "open_app": open_app,
    "web_search": web_search,
    "create_note": create_note,
    "system_status": system_status,
    "exit": exit_assistant,
    "unknown": unknown,
    "knowledge_query" : knowledge_query,
    "open_path": open_path,
    "read_notes": read_notes,
}
