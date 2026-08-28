import json
import threading
import numpy as np
import speech_recognition as sr
import pyttsx3
from config import VOCAB_FILE, WEIGHTS_FILE, SEED_FILE, CORRECTIONS_FILE
from core.classifier import predict
from core.jobs import start_reminder_loop, start_job_drain
from core.data_loader import load_examples, build_label_map
from actions.registry import REGISTRY
from memory.corrections import log_correction
from actions.app_finder import build_app_index
import time

THRESHOLD = 0.6
_speak_lock = threading.Lock()

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
        recognizer.adjust_for_ambient_noise(source,duration=0.5)
        try:
            audio = recognizer.listen(source)
        except sr.WaitTimeoutError as e:
            return None

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None


def speak(message):
    with _speak_lock:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()


def safe_speak(message):
    speak(message)

def normalize(text):
    words = text.lower().split()
    if len(words) < 2:
        return text
    
    first = words[0]
    rest = words[1:]

    if first == "search" and rest[0] != "for":
        return "search for " + " ".join(rest)

    if first in ("web","google"):
        return "search for " + " ".join(rest)

    return text

def process_one(text, vocab, label_map, w1, b1, w2, b2, context):
    text = normalize(text)
    label,confidence = predict(text, vocab, label_map, w1, b1, w2, b2)
    if confidence < THRESHOLD:
        return None, "Uncertainity"

    result = REGISTRY[label](text, context)
    if result:
        return label, result
    
    return label,"I didn't understand that."

def process(text,vocab,label_map,w1,b1,w2,b2,context):
    if " and " in text.lower():
        left,right = text.lower().split(" and ",1)
        left_label,left_message = process_one(left.strip(),vocab,label_map,w1,b1,w2,b2,context)
        right_label,right_message = process_one(right.strip(),vocab,label_map,w1,b1,w2,b2,context)

        label = "exit" if left_label == "exit" or right_label == "exit" else left_label

        return label, left_message + "\n" + right_message

    return process_one(text,vocab,label_map,w1,b1,w2,b2,context)


def main():
    start_reminder_loop()
    start_job_drain(safe_speak)

    vocab, label_map, w1, b1, w2, b2, context = load_model()
    print("Write a message...")
    print("Press 'v' to enable voice mode")
    print("say keyboard to go back to keyboard mode")

    last_text = None
    voice_mode = False
        
    while True:
        if voice_mode:
            heard = listen()
            if not heard:
                print("I didn't hear you.")
                continue
            text = heard
            print("heard:",text)

            if text.lower() in ["stop", "keyboard", "stop listening", "keyboard mode", "back to keyboard mode"]:
                voice_mode = False
                print("Type again...")
                continue

        else:
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
                voice_mode = True
                continue

        last_text = text

        label, message = process(text, vocab, label_map, w1, b1, w2, b2, context)
        print(message)
        speak(message)

        if voice_mode:
            time.sleep(1)
        if label == "exit":
            break
        if text == "exit":
            break


if __name__ == "__main__":
    main()
