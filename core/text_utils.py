import re


def clean_text(text):
    text = text.lower()

    text = text.replace("wi-fi", "wifi")

    text = re.sub(r"(\w)-(?=\w)", r"\1", text)

    text = text.replace("'", "").replace("’", "").replace("`", "")

    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.split())


def tokenize(text):
    return clean_text(text).split()
