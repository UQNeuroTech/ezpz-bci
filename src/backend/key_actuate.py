import yaml
from pathlib import Path
from pynput.keyboard import Controller, Key
import time

def press_key(key: str, duration: float = 0.1):
    """
    Presses the specified key on the keyboard for a given duration.
    """
    new_key = None
    mapping = read_config()
    for k,v in mapping.items():
        if v == key:
            new_key = k
    if new_key is None:
        print("Error: key not found in config")
        return
    try:
        Controller().press(new_key)
        time.sleep(duration)  # Hold the key for the specified duration
    finally:
        Controller().release(new_key)

def read_config() -> dict:
    """
    Reads the YAML configuration file and returns its contents as a dictionary.
    """
    path = Path("./data/config.json")
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    press_key("test")
