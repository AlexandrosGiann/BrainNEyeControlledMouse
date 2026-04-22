# BrainNEyeControlledMouse

A Python app that controls the mouse cursor using eye movements and clicks using eeg. This project is a work in progress.

---

## Project Structure
```
BrainNEyeControlledMouse/
├── README.md
├── config.py
├── main.py
├── app.py
├── requirements.txt
├── Models/
│   ├── __init__.py
│   └── eeg_state.py
├── Controllers/
│   ├── __init__.py
│   └── mediapipe_eye_contour_controller.py
└── Parsers/
   ├── __init__.py
   └── eeg_parser.py
```

---

## Installation
```bash
git clone https://github.com/AlexandrosGiann/BrainNEyeControlledMouse.git
pip install -r requirements.txt
```

---

## How to Run:

```bash
python main.py
```

---

## Tested on:

Windows 11 (python 3.11)

---

## Author:

Alexandros Giannakis
GitHub: https://github.com/AlexandrosGiann
