import json
import numpy as np
import speech_recognition as sr
from config import VOCAB_FILE, WEIGHTS_FILE, SEED_FILE, CORRECTIONS_FILE
from core.classifier import predict
from core.data_loader import load_examples, build_label_map
from actions.registry import REGISTRY
from memory.corrections import log_correction
from actions.app_finder import build_app_index

THRESHOLD = 0.6


def load_model():
    vocab = json.loads(VOCAB_FILE.read_text())
    data = np.load(WEIGHTS_FILE)
    w1, b1, w2, b2 = data["w1"], data["b1"], data["w2"], data["b2"]

    label_map = build_label_map(load_examples(SEED_FILE))

    app_index = build_app_index()

    context = {"app_index": app_index}

    return vocab, label_map, w1, b1, w2, b2, context

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak...")
        audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


def process(text, vocab, label_map, w1, b1, w2, b2, context):
    label,confidence = predict(text, vocab, label_map, w1, b1, w2, b2)
    if confidence < THRESHOLD:
        return None, "Uncertainity"

    result = REGISTRY[label](text, context)
    if result:
        return label, result
    
    return label,"I didn't understand that."


def main():
    vocab, label_map, w1, b1, w2, b2, context = load_model()
    print("Write a message...")

    last_text = None

    while True:
        text = input("> ").strip()

        if text.startswith("!fix "):
            label = text[5:].strip()
            if last_text is None:
                print("Nothing to correct")
                continue

            if label not in REGISTRY:
                print(f"unknown label. valid: {list(REGISTRY)}")
                continue

            log_correction(CORRECTIONS_FILE, last_text, label)
            print(f"logged: {last_text!r} -> {label}")
            continue

        if not text:
            continue
        
        if text == "v":
            heard = listen()
            if not heard:
                print("I didn't hear you.")
                continue
            text = heard
            print("heard:",text)

        last_text = text

        label,message = process(text,vocab, label_map, w1, b1, w2, b2, context)
        print(message)

        if label == "exit":
            break

        if text == "exit":
            break


if __name__ == "__main__":
    main()
