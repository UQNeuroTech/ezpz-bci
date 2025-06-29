from pynput.keyboard import Key, Controller, GlobalHotKeys
from pynput import keyboard

keyboard = Controller()

def press(keyInput):
    keyboard.press(keyInput)

press("a")

# ekgjel