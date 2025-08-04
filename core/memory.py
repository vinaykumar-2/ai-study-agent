import os
import json
from core.config import USER_HISTORY_PATH


def load_history():
    """Load chat history from file."""
    if os.path.exists(USER_HISTORY_PATH):
        try:
            with open(USER_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("[WARNING] Corrupted history file. Starting fresh.")
        except Exception as e:
            print(f"[ERROR] Could not load history: {e}")
    return []

def save_history(history):
    """Save chat history to file."""
    os.makedirs("data", exist_ok=True)
    try:
        with open(USER_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save history: {e}")

def add_message(role, content):
    """Append a new message and save."""
    history = load_history()
    history.append({"role": role, "content": content})
    save_history(history)
