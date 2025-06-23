from pynput.keyboard import Controller, Key
import time
import threading

class KeyActuator(threading.Thread):
    def __init__(self):
        super().__init__()
        self.keyboard = Controller()

    def press_key(self, key: str, duration: int = 1):
        """
        Presses the specified key on the keyboard for a given duration.
        """
        try:
            self.keyboard.press(key)
            time.sleep(duration)  # Hold the key for the specified duration
        finally:
            self.keyboard.release(key)


