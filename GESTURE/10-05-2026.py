import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SMOOTHING = 0.2
PINCH_THRESHOLD = 0.05
WINDOW_NAME = "Hand Mouse Control"
SCREEN_WIDTH, SCREEN_HEIGHT = pg.size()
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def move_cursor(index_x: float, index_y: float) -> None:
    target_x = min((1 - index_x) * SCREEN_WIDTH, SCREEN_WIDTH - 1)
    target_y = min(index_y * SCREEN_HEIGHT, SCREEN_HEIGHT - 1)

    if not hasattr(move_cursor, "cursor_x"):
        move_cursor.cursor_x = target_x
        move_cursor.cursor_y = target_y
    else:
        move_cursor.cursor_x += (target_x - move_cursor.cursor_x) * SMOOTHING
        move_cursor.cursor_y += (target_y - move_cursor.cursor_y) * SMOOTHING

    pg.moveTo(x=move_cursor.cursor_x, y=move_cursor.cursor_y, duration=0.01)


def handle_left_click(index_tip, thumb_tip) -> None:
    pinch_distance = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
    is_pinched = pinch_distance <= PINCH_THRESHOLD
    was_pinched = getattr(handle_left_click, "was_pinched", False)
    drag_started_at = getattr(handle_left_click, "drag_started_at", None)
    mouse_down = getattr(handle_left_click, "mouse_down", False)
    now = time.monotonic()

    if is_pinched and not was_pinched:
        handle_left_click.drag_started_at = now
    elif is_pinched and was_pinched and drag_started_at is not None:
        if not mouse_down and now - drag_started_at >= 0.25:
            pg.mouseDown()
            handle_left_click.mouse_down = True
    elif not is_pinched and was_pinched:
        if mouse_down:
            pg.mouseUp()
            handle_left_click.mouse_down = False
        else:
            pg.click()
        handle_left_click.drag_started_at = None

    handle_left_click.was_pinched = is_pinched


def reset_click_state() -> None:
    if getattr(handle_left_click, "mouse_down", False):
        pg.mouseUp()
        handle_left_click.mouse_down = False
    handle_left_click.was_pinched = False
    handle_left_click.drag_started_at = None


def draw_hand_annotations(frame, hand_landmarks_list, handedness_list) -> None:
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        points = []
        for lm in hand_landmarks:
            px = int((1 - lm.x) * frame.shape[1])
            py = int(lm.y * frame.shape[0])
            points.append((px, py))
            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 200, 0), 2)

        label = handedness[0].category_name
        x = int((1 - max(lm.x for lm in hand_landmarks)) * frame.shape[1])
        y = int(min(lm.y for lm in hand_landmarks) * frame.shape[0]) - 10
        cv2.putText(
            frame,
            label,
            (x, max(y, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            display_frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            results = landmarker.detect_for_video(image, timestamp_ms)

            right_hand_found = False
            left_hand_found = False

            if results.hand_landmarks and results.handedness:
                draw_hand_annotations(display_frame, results.hand_landmarks, results.handedness)

                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    hand_label = handedness[0].category_name.lower()

                    if hand_label == "right":
                        index_tip = hand_landmarks[8]
                        move_cursor(index_tip.x, index_tip.y)
                        right_hand_found = True
                    elif hand_label == "left":
                        index_tip = hand_landmarks[8]
                        thumb_tip = hand_landmarks[4]
                        handle_left_click(index_tip, thumb_tip)
                        left_hand_found = True

            if not left_hand_found:
                reset_click_state()

            status_text = "Move: right index finger | Click: left pinch | Drag: hold pinch"
            if not right_hand_found:
                status_text = "Show your right hand to move the cursor"
            elif not left_hand_found:
                status_text = "Left-hand pinch clicks, hold pinch to drag"

            cv2.putText(
                display_frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
