def show_toast(title: str, message: str):
    try:
        from winotify import Notification

        toast = Notification(
            app_id="Personal Assistant",
            title=title,
            msg=message,
            duration="short",
        )
        toast.show()
    except Exception as e:
        print(f"[toast failed] {type(e).__name__}: {e}")
