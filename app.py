import tkinter as tk
import time
import pyautogui
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Models.eeg_state import EEGState
from Controllers.mediapipe_eye_contour_controller import EyeContourMouseController
from config import *

# =========================================================
# APP
# =========================================================
class EEGEyeMouseApp:
    def __init__(self, root: tk.Tk, state: EEGState) -> None:
        self.root = root
        self.state = state
        self.eye_mouse = EyeContourMouseController()

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        self.last_click_time = 0.0
        self.consecutive_windows = 0

        self.root.title("MindWave + Eye Contour Mouse")
        self.root.geometry("1280x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        info_frame = tk.Frame(root)
        info_frame.pack(fill="x", padx=10, pady=10)

        self.status_var = tk.StringVar(value="Status: starting...")
        self.signal_var = tk.StringVar(value="Poor Signal: -")
        self.attention_var = tk.StringVar(value="Attention: -")
        self.meditation_var = tk.StringVar(value="Meditation: -")
        self.blink_var = tk.StringVar(value="Blink: -")

        self.detector_var = tk.StringVar(value="Detector: idle")
        self.std_var = tk.StringVar(value="STD: -")
        self.ptp_var = tk.StringVar(value="Peak-to-Peak: -")
        self.active_var = tk.StringVar(value="Active Samples: -")
        self.cam_var = tk.StringVar(value="Camera: starting...")

        tk.Label(info_frame, textvariable=self.status_var, font=("Arial", 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=4, columnspan=5
        )
        tk.Label(info_frame, textvariable=self.signal_var, font=("Arial", 11)).grid(
            row=1, column=0, sticky="w", padx=8, pady=2
        )
        tk.Label(info_frame, textvariable=self.attention_var, font=("Arial", 11)).grid(
            row=1, column=1, sticky="w", padx=8, pady=2
        )
        tk.Label(info_frame, textvariable=self.meditation_var, font=("Arial", 11)).grid(
            row=1, column=2, sticky="w", padx=8, pady=2
        )
        tk.Label(info_frame, textvariable=self.blink_var, font=("Arial", 11)).grid(
            row=1, column=3, sticky="w", padx=8, pady=2
        )
        tk.Label(info_frame, textvariable=self.cam_var, font=("Arial", 11)).grid(
            row=1, column=4, sticky="w", padx=8, pady=2
        )

        tk.Label(info_frame, textvariable=self.detector_var, font=("Arial", 11, "bold")).grid(
            row=2, column=0, sticky="w", padx=8, pady=4
        )
        tk.Label(info_frame, textvariable=self.std_var, font=("Arial", 11)).grid(
            row=2, column=1, sticky="w", padx=8, pady=4
        )
        tk.Label(info_frame, textvariable=self.ptp_var, font=("Arial", 11)).grid(
            row=2, column=2, sticky="w", padx=8, pady=4
        )
        tk.Label(info_frame, textvariable=self.active_var, font=("Arial", 11)).grid(
            row=2, column=3, sticky="w", padx=8, pady=4
        )

        fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Raw EEG Signal")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_ylim(-2200, 2200)

        initial_data = [0] * RAW_BUFFER_SIZE
        (self.line,) = self.ax.plot(initial_data)

        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.update_loop()

    def eeg_click_detector(self, raw_values, poor_signal):
        if len(raw_values) < DETECTION_WINDOW:
            self.consecutive_windows = 0
            return "warming up", 0.0, 0.0, 0

        window = np.array(raw_values[-DETECTION_WINDOW:], dtype=np.float32)

        # Αφαιρούμε DC offset / αργή μετατόπιση
        mean_val = float(np.mean(window))
        centered = window - mean_val

        # Βασικά metrics έντασης
        std_val = float(np.std(centered))
        rms_val = float(np.sqrt(np.mean(centered ** 2)))
        ptp_val = float(np.max(window) - np.min(window))

        # Πόσα samples έχουν ουσιαστική ένταση
        active_mask = np.abs(centered) >= ACTIVE_SAMPLE_THRESHOLD
        active_count = int(np.sum(active_mask))
        active_ratio = active_count / len(window)

        # Zero-crossings: μετράμε εναλλαγές πρόσημου μόνο όταν το σήμα είναι αρκετά έντονο
        # για να μη μετράμε μικροθόρυβο γύρω από το 0.
        zc_threshold = ACTIVE_SAMPLE_THRESHOLD * 0.45
        gated = np.where(np.abs(centered) >= zc_threshold, centered, 0.0)

        zero_crossings = 0
        prev_sign = 0
        for x in gated:
            sign = 1 if x > 0 else (-1 if x < 0 else 0)
            if sign != 0:
                if prev_sign != 0 and sign != prev_sign:
                    zero_crossings += 1
                prev_sign = sign

        # Συνθήκες:
        # 1) ένταση
        burst_energy_ok = std_val >= MIN_STD and ptp_val >= MIN_PEAK_TO_PEAK and rms_val >= MIN_RMS

        # 2) διάρκεια
        sustained_ok = active_count >= MIN_SUSTAINED_SAMPLES and active_ratio >= MIN_ACTIVE_RATIO

        # 3) ταλαντωτική μορφή, όχι ένα spike
        oscillation_ok = zero_crossings >= MIN_ZERO_CROSSINGS

        signal_ok = True
        if REQUIRE_GOOD_SIGNAL:
            signal_ok = poor_signal is not None and poor_signal <= MAX_POOR_SIGNAL

        detected = signal_ok and burst_energy_ok and sustained_ok and oscillation_ok

        if detected:
            self.consecutive_windows += 1
        else:
            self.consecutive_windows = 0

        cooldown_ok = (time.time() - self.last_click_time) >= CLICK_COOLDOWN_SEC

        if self.consecutive_windows >= MIN_CONSECUTIVE_WINDOWS and cooldown_ok:
            try:
                pyautogui.click(button=CLICK_BUTTON)
                self.last_click_time = time.time()
                self.consecutive_windows = 0
                return f"CLICK! zc={zero_crossings}", std_val, ptp_val, active_count
            except pyautogui.FailSafeException:
                return f"failsafe zc={zero_crossings}", std_val, ptp_val, active_count

        if not signal_ok:
            return f"bad signal zc={zero_crossings}", std_val, ptp_val, active_count

        if burst_energy_ok and sustained_ok and not oscillation_ok:
            return f"non-oscillatory burst zc={zero_crossings}", std_val, ptp_val, active_count

        if burst_energy_ok and not sustained_ok:
            return f"short activity zc={zero_crossings}", std_val, ptp_val, active_count

        if detected:
            return f"armed zc={zero_crossings}", std_val, ptp_val, active_count

        return f"idle zc={zero_crossings}", std_val, ptp_val, active_count

    def update_loop(self):
        with self.state.lock:
            raw_values = list(self.state.raw_buffer)
            connected = self.state.connected
            last_error = self.state.last_error
            poor_signal = self.state.poor_signal
            attention = self.state.attention
            meditation = self.state.meditation
            blink = self.state.blink

        if connected:
            self.status_var.set("Status: connected")
        else:
            msg = "Status: disconnected"
            if last_error:
                msg += f" | {last_error}"
            self.status_var.set(msg)

        self.signal_var.set(f"Poor Signal: {poor_signal if poor_signal is not None else '-'}")
        self.attention_var.set(f"Attention: {attention if attention is not None else '-'}")
        self.meditation_var.set(f"Meditation: {meditation if meditation is not None else '-'}")
        self.blink_var.set(f"Blink: {blink if blink is not None else '-'}")

        detector_state, std_val, ptp_val, active_count = self.eeg_click_detector(raw_values, poor_signal)
        self.detector_var.set(f"Detector: {detector_state}")
        self.std_var.set(f"STD: {std_val:.1f}")
        self.ptp_var.set(f"Peak-to-Peak: {ptp_val:.1f}")
        self.active_var.set(f"Active Samples: {active_count}/{DETECTION_WINDOW}")

        frame = self.eye_mouse.step()
        self.cam_var.set(f"Camera: {self.eye_mouse.last_debug}")

        if frame is not None:
            cv2.imshow("Eye Contour Mouse Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                self.on_close()
                return
            elif key == ord("a"):
                self.eye_mouse.cal_x_min -= 0.01
                print("cal_x_min =", self.eye_mouse.cal_x_min)
            elif key == ord("d"):
                self.eye_mouse.cal_x_max += 0.01
                print("cal_x_max =", self.eye_mouse.cal_x_max)
            elif key == ord("w"):
                self.eye_mouse.cal_y_min -= 0.01
                print("cal_y_min =", self.eye_mouse.cal_y_min)
            elif key == ord("s"):
                self.eye_mouse.cal_y_max += 0.01
                print("cal_y_max =", self.eye_mouse.cal_y_max)
            elif key == ord("r"):
                self.eye_mouse.hist_x.clear()
                self.eye_mouse.hist_y.clear()
                print("Smoothing reset.")

        self.line.set_ydata(raw_values)
        self.line.set_xdata(range(len(raw_values)))
        self.ax.set_xlim(0, len(raw_values) - 1)
        self.canvas.draw_idle()

        self.root.after(UI_UPDATE_MS, self.update_loop)

    def on_close(self):
        self.eye_mouse.close()
        cv2.destroyAllWindows()
        self.root.destroy()
