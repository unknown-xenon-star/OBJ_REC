import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SMOOTHING = 0.2
PINCH_THRESHOLD = 0.05


def screen_clicker(x1: float, y1: float, x2: float, y2: float, h1: float, h2: float) -> None:
    target_x = np.minimum(1900 * (1 - x1) * 1.5, 1880)
    target_y = np.minimum(1190 * y1 * 1.5, 1190)

    if not hasattr(screen_clicker, "cursor_x"):
        screen_clicker.cursor_x = target_x
        screen_clicker.cursor_y = target_y
    else:
        screen_clicker.cursor_x += (target_x - screen_clicker.cursor_x) * SMOOTHING
        screen_clicker.cursor_y += (target_y - screen_clicker.cursor_y) * SMOOTHING

    pg.moveTo(
        x=h1,
        y=h2,
        duration=0.01,
    )

    pinch_distance = np.hypot(x1 - x2, y1 - y2)
    is_pinched = pinch_distance <= PINCH_THRESHOLD
    was_pinched = getattr(screen_clicker, "was_pinched", False)

    if is_pinched and not was_pinched:
        pg.click()
    screen_clicker.was_pinched = is_pinched


def main() -> None:
    results = None

    def callback(result, _output_image, _timestamp_ms):
        nonlocal results
        results = result

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=callback,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1900)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count = (frame_count + 1) % 10
        if frame_count == 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(image, int(time.time() * 1000))

        if results and results.hand_landmarks:
            tip = results.hand_landmarks[0][8]
            thumb = results.hand_landmarks[0][4]
            hand_base = results.hand_landmarks[0][0]
            screen_clicker(tip.x, tip.y, thumb.x, thumb.y, hand_base.x, hand_base.y)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
