import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui as pg
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SMOOTHING = 0.2
PINCH_THRESHOLD = 0.05
PINCH_RELEASE_THRESHOLD = 0.07   # Hysteresis: release needs a wider gap than engage
DRAG_HOLD_DURATION = 0.25        # Seconds of held pinch before drag starts
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


# ── Cursor movement ──────────────────────────────────────────────────────────

_cursor_x: float | None = None
_cursor_y: float | None = None


def _smooth_move(norm_x: float, norm_y: float) -> None:
    """Convert normalised hand coords → screen coords and apply EMA smoothing.
    This is the single place that updates _cursor_x/_cursor_y and calls moveTo,
    so there is never a position discontinuity regardless of which landmark is
    being tracked."""
    global _cursor_x, _cursor_y

    target_x = min((1 - norm_x) * SCREEN_WIDTH, SCREEN_WIDTH - 1)
    target_y = min(norm_y * SCREEN_HEIGHT, SCREEN_HEIGHT - 1)

    if _cursor_x is None:
        _cursor_x, _cursor_y = target_x, target_y
    else:
        _cursor_x += (target_x - _cursor_x) * SMOOTHING
        _cursor_y += (target_y - _cursor_y) * SMOOTHING

    pg.moveTo(x=_cursor_x, y=_cursor_y, duration=0.01)


def move_cursor(index_x: float, index_y: float) -> None:
    """Normal (non-drag) cursor movement driven by the right-hand index tip."""
    _smooth_move(index_x, index_y)


# ── Click / drag state machine ───────────────────────────────────────────────

class ClickDragState:
    """
    States
    ------
    IDLE        : fingers apart, nothing happening
    PINCH_DOWN  : pinch just started, waiting to decide click vs drag
    DRAGGING    : pinch held long enough → mouseDown, user is dragging
    """

    IDLE = "IDLE"
    PINCH_DOWN = "PINCH_DOWN"
    DRAGGING = "DRAGGING"

    def __init__(self) -> None:
        self.state = self.IDLE
        self.pinch_start_time: float | None = None

    # ── public interface ──────────────────────────────────────────────────────

    def update(self, index_tip, thumb_tip) -> None:
        """Left hand is trigger-only. It never moves the cursor — that is
        always the right hand's job. This method purely detects pinch geometry
        and fires the appropriate mouse event."""
        distance = np.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
        now = time.monotonic()

        if self.state == self.IDLE:
            if distance <= PINCH_THRESHOLD:
                self.state = self.PINCH_DOWN
                self.pinch_start_time = now

        elif self.state == self.PINCH_DOWN:
            if distance > PINCH_RELEASE_THRESHOLD:
                # Released quickly → plain click at current cursor position
                pg.click()
                self.state = self.IDLE
                self.pinch_start_time = None
            elif now - self.pinch_start_time >= DRAG_HOLD_DURATION:
                # Held long enough → begin drag at current cursor position
                pg.mouseDown()
                self.state = self.DRAGGING

        elif self.state == self.DRAGGING:
            if distance > PINCH_RELEASE_THRESHOLD:
                # Pinch released → drop whatever is being dragged
                pg.mouseUp()
                self.state = self.IDLE
            # Cursor continues to be driven by the right hand in the main loop

    def reset(self) -> None:
        """Call when the hand disappears from frame."""
        if self.state == self.DRAGGING:
            pg.mouseUp()
        self.state = self.IDLE
        self.pinch_start_time = None

    @property
    def is_dragging(self) -> bool:
        return self.state == self.DRAGGING

    @property
    def is_pinched(self) -> bool:
        return self.state in (self.PINCH_DOWN, self.DRAGGING)

# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_hand_annotations(
    frame,
    hand_landmarks_list,
    handedness_list,
    drag_state: ClickDragState,
) -> None:
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        hand_label = handedness[0].category_name.lower()
        is_left = hand_label == "left"

        points = []
        for lm in hand_landmarks:
            px = int((1 - lm.x) * frame.shape[1])
            py = int(lm.y * frame.shape[0])
            points.append((px, py))
            dot_color = (0, 128, 255) if (is_left and drag_state.is_dragging) else (0, 255, 255)
            cv2.circle(frame, (px, py), 4, dot_color, -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (255, 200, 0), 2)

        label = handedness[0].category_name
        if is_left and drag_state.is_dragging:
            label += " [DRAG]"
        elif is_left and drag_state.is_pinched:
            label += " [PINCH]"

        x = int((1 - max(lm.x for lm in hand_landmarks)) * frame.shape[1])
        y = int(min(lm.y for lm in hand_landmarks) * frame.shape[0]) - 10
        cv2.putText(
            frame,
            label,
            (x, max(y, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if not drag_state.is_dragging else (0, 128, 255),
            2,
            cv2.LINE_AA,
        )


# ── Main loop ────────────────────────────────────────────────────────────────

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

    drag_state = ClickDragState()

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
                draw_hand_annotations(display_frame, results.hand_landmarks, results.handedness, drag_state)

                for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                    hand_label = handedness[0].category_name.lower()

                    if hand_label == "right":
                        index_tip = hand_landmarks[8]
                        # Right hand drives the cursor at all times —
                        # including while the left hand holds a drag.
                        move_cursor(index_tip.x, index_tip.y)
                        right_hand_found = True

                    elif hand_label == "left":
                        index_tip = hand_landmarks[8]
                        thumb_tip = hand_landmarks[4]
                        # Left hand is trigger-only: pinch = click/drag,
                        # but cursor position is always the right hand's job.
                        drag_state.update(index_tip, thumb_tip)
                        left_hand_found = True

            if not left_hand_found:
                drag_state.reset()

            # ── Status overlay ────────────────────────────────────────────────
            if drag_state.is_dragging:
                status_text = "DRAGGING — open pinch to release"
                status_color = (0, 128, 255)
            elif not right_hand_found:
                status_text = "Show your right hand to move the cursor"
                status_color = (255, 255, 255)
            elif not left_hand_found:
                status_text = "Left-hand pinch to click  |  hold pinch to drag"
                status_color = (255, 255, 255)
            else:
                status_text = "Move: right index  |  Click: left pinch  |  Drag: hold pinch"
                status_color = (255, 255, 255)

            cv2.putText(
                display_frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        drag_state.reset()  # Ensure mouseUp on exit
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
