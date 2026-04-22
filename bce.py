import threading
import tkinter as tk

from Models.eeg_state import EEGState
from Parsers.eeg_parser import serial_worker
from app import EEGEyeMouseApp


def main():
    state = EEGState()

    worker = threading.Thread(target=serial_worker, args=(state,), daemon=True)
    worker.start()

    root = tk.Tk()
    EEGEyeMouseApp(root, state)
    root.mainloop()


if __name__ == "__main__":
    main()
