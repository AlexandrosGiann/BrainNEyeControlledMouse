import numpy as np
import mediapipe as mp
import cv2
import pyautogui
from collections import deque
from config import *

# =========================================================
# MEDIAPIPE EYE CONTOUR CONTROLLER
# =========================================================

def lm_to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def avg_landmarks(face_landmarks, ids, w, h):
    pts = [lm_to_px(face_landmarks.landmark[i], w, h) for i in ids]
    return np.mean(pts, axis=0)


def eye_features(face_landmarks, inner_id, outer_id, top_id, bottom_id, region_ids, w, h):
    inner = lm_to_px(face_landmarks.landmark[inner_id], w, h)
    outer = lm_to_px(face_landmarks.landmark[outer_id], w, h)
    top = lm_to_px(face_landmarks.landmark[top_id], w, h)
    bottom = lm_to_px(face_landmarks.landmark[bottom_id], w, h)
    center = avg_landmarks(face_landmarks, region_ids, w, h)

    width = np.linalg.norm(outer - inner) + 1e-6
    height = np.linalg.norm(bottom - top) + 1e-6
    openness = height / width
    angle = np.arctan2(outer[1] - inner[1], outer[0] - inner[0])

    return {
        "inner": inner,
        "outer": outer,
        "top": top,
        "bottom": bottom,
        "center": center,
        "width": width,
        "height": height,
        "openness": openness,
        "angle": angle,
    }


def remap(v, in_min, in_max):
    return (v - in_min) / (in_max - in_min + 1e-6)


class EyeContourMouseController:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        self.hist_x = deque(maxlen=SMOOTHING_WINDOW)
        self.hist_y = deque(maxlen=SMOOTHING_WINDOW)

        self.cal_x_min = 0.40
        self.cal_x_max = 0.60
        self.cal_y_min = 0.42
        self.cal_y_max = 0.58

        self.last_debug = "camera starting"

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def step(self):
        if not self.cap.isOpened():
            self.last_debug = "camera not opened"
            return None

        ok, frame = self.cap.read()
        if not ok:
            self.last_debug = "camera frame failed"
            return None

        if MIRROR_IMAGE:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            re = eye_features(
                face_landmarks,
                RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                RIGHT_EYE_POINTS, w, h
            )

            le = eye_features(
                face_landmarks,
                LEFT_EYE_INNER, LEFT_EYE_OUTER,
                LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                LEFT_EYE_POINTS, w, h
            )

            eyes_mid = (re["center"] + le["center"]) / 2.0
            x_ratio = eyes_mid[0] / w

            avg_openness = (re["openness"] + le["openness"]) / 2.0
            y_ratio = eyes_mid[1] / h

            avg_angle = (re["angle"] + le["angle"]) / 2.0
            angle_norm = np.clip(avg_angle / 0.35, -1.0, 1.0)

            gx = x_ratio + 0.03 * angle_norm
            gy = y_ratio - 0.10 * (avg_openness - 0.23)

            if abs(gx - 0.5) < DEADZONE_X:
                gx = 0.5
            if abs(gy - 0.5) < DEADZONE_Y:
                gy = 0.5

            sx = remap(gx, self.cal_x_min, self.cal_x_max)
            sy = remap(gy, self.cal_y_min, self.cal_y_max)

            sx = clamp(sx, 0.0, 1.0)
            sy = clamp(sy, 0.0, 1.0)

            sx = 0.5 + (sx - 0.5) * MOVE_SCALE_X
            sy = 0.5 + (sy - 0.5) * MOVE_SCALE_Y

            sx = clamp(sx, 0.0, 1.0)
            sy = clamp(sy, 0.0, 1.0)

            mouse_x = int(sx * (self.screen_w - 2 * SAFE_MARGIN) + SAFE_MARGIN)
            mouse_y = int(sy * (self.screen_h - 2 * SAFE_MARGIN) + SAFE_MARGIN)

            self.hist_x.append(mouse_x)
            self.hist_y.append(mouse_y)

            final_x = int(np.mean(self.hist_x))
            final_y = int(np.mean(self.hist_y))

            final_x = max(SAFE_MARGIN, min(self.screen_w - SAFE_MARGIN, final_x))
            final_y = max(SAFE_MARGIN, min(self.screen_h - SAFE_MARGIN, final_y))

            try:
                pyautogui.moveTo(final_x, final_y)
            except pyautogui.FailSafeException:
                self.last_debug = "mouse failsafe"

            #for p in [re["center"], le["center"], re["inner"], re["outer"], le["inner"], le["outer"]]:
                #cv2.circle(frame, tuple(p.astype(int)), 3, (0, 255, 0), -1)

            #cv2.line(frame, tuple(re["center"].astype(int)), tuple(le["center"].astype(int)), (255, 0, 0), 2)

            cv2.putText(frame, f"gx: {gx:.3f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"gy: {gy:.3f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"open: {avg_openness:.3f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"mouse: {final_x}, {final_y}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            self.last_debug = "tracking"
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.last_debug = "no face"

        return frame

    def close(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

