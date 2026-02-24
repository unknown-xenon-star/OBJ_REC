#!/usr/bin/env python3
"""Naruto Shadow Clone app in Python.

Modes:
- train: collect hand-sign samples, train a binary classifier, and save model/data.
- run: load a trained model and trigger clone/smoke effects on gesture detection.

Dependencies:
  pip install opencv-python mediapipe tensorflow numpy
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DATA_JSON = Path("gesture-data.json")
MODEL_PATH = Path("gesture-model.keras")

FINGER_SEGMENTS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

CUSTOM_CLONES = [
    {"x": -100, "y": 100, "scale": 0.9, "delay": 1000},
    {"x": 120, "y": 100, "scale": 0.85, "delay": 1150},
    {"x": -180, "y": 140, "scale": 0.8, "delay": 1300},
    {"x": -140, "y": 140, "scale": 0.45, "delay": 1320},
    {"x": 180, "y": 160, "scale": 0.7, "delay": 1450},
    {"x": 140, "y": 160, "scale": 0.4, "delay": 1470},
    {"x": -250, "y": 140, "scale": 0.7, "delay": 1600},
    {"x": -220, "y": 140, "scale": 0.35, "delay": 1620},
    {"x": 260, "y": 160, "scale": 0.65, "delay": 1750},
    {"x": -100, "y": 150, "scale": 0.6, "delay": 2500},
    {"x": 100, "y": 150, "scale": 0.6, "delay": 2650},
    {"x": -120, "y": 70, "scale": 0.55, "delay": 2800},
    {"x": 100, "y": 70, "scale": 0.5, "delay": 2950},
    {"x": -200, "y": 85, "scale": 0.55, "delay": 3100},
    {"x": 230, "y": 85, "scale": 0.5, "delay": 3250},
    {"x": -280, "y": 100, "scale": 0.4, "delay": 3400},
]


@dataclass
class SmokeAnim:
    x: int
    y: int
    scale: float
    start_ms: float
    frames: List[np.ndarray]


def normalize_hand(landmarks: List[Tuple[float, float, float]]) -> List[float]:
    wrist = landmarks[0]
    mcp = landmarks[9]
    scale = math.sqrt(
        (mcp[0] - wrist[0]) ** 2 + (mcp[1] - wrist[1]) ** 2 + (mcp[2] - wrist[2]) ** 2
    )
    if scale == 0:
        scale = 1.0

    out: List[float] = []
    for i in range(21):
        out.append((landmarks[i][0] - wrist[0]) / scale)
        out.append((landmarks[i][1] - wrist[1]) / scale)
        out.append((landmarks[i][2] - wrist[2]) / scale)
    return out


def extract_features(
    right: List[Tuple[float, float, float]], left: List[Tuple[float, float, float]]
) -> List[float]:
    return normalize_hand(right) + normalize_hand(left)


def mp_hand_to_list(hand_lms) -> List[Tuple[float, float, float]]:
    return [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]


def draw_hand(frame: np.ndarray, hand_lms, mirrored: bool = False) -> None:
    h, w = frame.shape[:2]
    for seg in FINGER_SEGMENTS:
        points = []
        for i in seg:
            x = int(hand_lms.landmark[i].x * w)
            y = int(hand_lms.landmark[i].y * h)
            if mirrored:
                x = w - x
            points.append((x, y))
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (34, 197, 94), 2)

    for lm in hand_lms.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        if mirrored:
            x = w - x
        cv2.circle(frame, (x, y), 3, (68, 68, 239), -1)


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(126,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def fit_model(samples: Dict[str, List[List[float]]], epochs: int = 50) -> tf.keras.Model:
    xs = samples["clone_sign"] + samples["not_sign"]
    ys = [1] * len(samples["clone_sign"]) + [0] * len(samples["not_sign"])

    idx = list(range(len(xs)))
    random.shuffle(idx)
    x_arr = np.array([xs[i] for i in idx], dtype=np.float32)
    y_arr = np.array([ys[i] for i in idx], dtype=np.float32)

    model = build_model()
    model.fit(x_arr, y_arr, epochs=epochs, batch_size=16, verbose=0, shuffle=True)
    return model


def save_samples(path: Path, samples: Dict[str, List[List[float]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(samples, f)


def load_samples(path: Path) -> Dict[str, List[List[float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "clone_sign": list(data.get("clone_sign", [])),
        "not_sign": list(data.get("not_sign", [])),
    }


def put_text(frame: np.ndarray, text: str, y: int, color=(255, 255, 255), scale: float = 0.6) -> None:
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def blend_rgba(bg: np.ndarray, fg_rgba: np.ndarray, center_x: int, center_y: int, scale: float = 1.0) -> None:
    if fg_rgba is None:
        return

    if scale != 1.0:
        new_w = max(1, int(fg_rgba.shape[1] * scale))
        new_h = max(1, int(fg_rgba.shape[0] * scale))
        fg_rgba = cv2.resize(fg_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w = fg_rgba.shape[:2]
    x1 = center_x - w // 2
    y1 = center_y - h // 2
    x2 = x1 + w
    y2 = y1 + h

    bg_h, bg_w = bg.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= bg_w or y1 >= bg_h:
        return

    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(bg_w, x2)
    cy2 = min(bg_h, y2)

    fx1 = cx1 - x1
    fy1 = cy1 - y1
    fx2 = fx1 + (cx2 - cx1)
    fy2 = fy1 + (cy2 - cy1)

    fg_crop = fg_rgba[fy1:fy2, fx1:fx2]
    alpha = fg_crop[:, :, 3:4].astype(np.float32) / 255.0
    fg_bgr = fg_crop[:, :, :3].astype(np.float32)

    roi = bg[cy1:cy2, cx1:cx2].astype(np.float32)
    out = fg_bgr * alpha + roi * (1.0 - alpha)
    bg[cy1:cy2, cx1:cx2] = out.astype(np.uint8)


def extract_person(frame_bgr: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
    mask = (seg_mask > 0.1).astype(np.uint8)
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def load_smoke_frames() -> Dict[str, List[np.ndarray]]:
    out: Dict[str, List[np.ndarray]] = {}
    for folder in ("smoke_1", "smoke_2", "smoke_3"):
        frames = []
        for i in range(1, 6):
            p = ASSETS_DIR / folder / f"{i}.png"
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 3:
                alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=2)
            frames.append(img)
        out[folder] = frames
    return out


def train_mode(data_path: Path, model_out: Path, camera_id: int) -> None:
    samples: Dict[str, List[List[float]]] = {"clone_sign": [], "not_sign": []}
    model: Optional[tf.keras.Model] = None

    if data_path.exists():
        samples = load_samples(data_path)

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    recording_label: Optional[str] = None
    countdown_until = 0.0
    record_until = 0.0
    record_seconds = 4

    print("Controls: 1=record clone sign, 2=record other, t=train, s=save model, e=export data, c=clear, q=quit")

    with mp_holistic.Holistic(model_complexity=1, smooth_landmarks=True) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            right = res.right_hand_landmarks
            left = res.left_hand_landmarks

            if right:
                draw_hand(frame, right, mirrored=False)
            if left:
                draw_hand(frame, left, mirrored=False)

            now = time.time()
            if countdown_until > now:
                remaining = int(math.ceil(countdown_until - now))
                put_text(frame, f"GET READY... {remaining}", 30, (0, 255, 255))
            elif recording_label and record_until > now:
                remaining = int(math.ceil(record_until - now))
                put_text(frame, f"REC {recording_label}: {remaining}s", 30, (0, 0, 255))
                if right and left:
                    feat = extract_features(mp_hand_to_list(right), mp_hand_to_list(left))
                    samples[recording_label].append(feat)
            elif recording_label and record_until <= now:
                recording_label = None

            if model and right and left:
                feat = np.array([extract_features(mp_hand_to_list(right), mp_hand_to_list(left))], dtype=np.float32)
                prob = float(model.predict(feat, verbose=0)[0][0])
                put_text(frame, f"Confidence: {prob * 100:.1f}%", 55, (0, 255, 0))

            put_text(
                frame,
                f"clone_sign: {len(samples['clone_sign'])} | not_sign: {len(samples['not_sign'])}",
                frame.shape[0] - 20,
                (200, 200, 200),
                0.55,
            )

            cv2.imshow("Naruto Trainer (Python)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("1"):
                recording_label = "clone_sign"
                countdown_until = time.time() + 3
                record_until = countdown_until + record_seconds
            elif key == ord("2"):
                recording_label = "not_sign"
                countdown_until = time.time() + 3
                record_until = countdown_until + record_seconds
            elif key == ord("t"):
                if len(samples["clone_sign"]) < 5 or len(samples["not_sign"]) < 5:
                    print("Need at least 5 samples each before training")
                else:
                    print("Training model...")
                    model = fit_model(samples)
                    print("Training complete")
            elif key == ord("s"):
                if model is None:
                    print("Train a model first")
                else:
                    model.save(model_out)
                    print(f"Model saved: {model_out}")
            elif key == ord("e"):
                save_samples(data_path, samples)
                print(f"Data saved: {data_path}")
            elif key == ord("c"):
                samples = {"clone_sign": [], "not_sign": []}
                print("Cleared sample data")

    cap.release()
    cv2.destroyAllWindows()


def run_mode(model_path: Path, threshold: float, camera_id: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    smoke_frames = load_smoke_frames()
    overlay_1 = cv2.imread(str(ASSETS_DIR / "state-1.png"), cv2.IMREAD_UNCHANGED)
    overlay_2 = cv2.imread(str(ASSETS_DIR / "state-2.png"), cv2.IMREAD_UNCHANGED)

    mp_holistic = mp.solutions.holistic
    mp_seg = mp.solutions.selfie_segmentation

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    clones_triggered = False
    clone_start_ms = 0.0
    smoke_spawned = [False] * len(CUSTOM_CLONES)
    active_smokes: List[SmokeAnim] = []

    smoke_duration_ms = 600

    def spawn_smoke(x: int, y: int, scale: float) -> None:
        folder = random.choice(list(smoke_frames.keys()))
        frames = smoke_frames[folder]
        active_smokes.append(SmokeAnim(x=x, y=y, scale=scale * 1.2, start_ms=time.time() * 1000.0, frames=frames))

    with mp_holistic.Holistic(model_complexity=1, smooth_landmarks=True) as holistic, mp_seg.SelfieSegmentation(model_selection=1) as seg:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            seg_res = seg.process(rgb)
            hol_res = holistic.process(rgb)

            person = extract_person(frame, seg_res.segmentation_mask)
            out = frame.copy()

            right = hol_res.right_hand_landmarks
            left = hol_res.left_hand_landmarks

            if not clones_triggered and right and left:
                feat = np.array([extract_features(mp_hand_to_list(right), mp_hand_to_list(left))], dtype=np.float32)
                prob = float(model.predict(feat, verbose=0)[0][0])
                put_text(out, f"Confidence: {prob * 100:.1f}%", 30, (255, 255, 255))
                if prob > threshold:
                    clones_triggered = True
                    clone_start_ms = time.time() * 1000.0

            if clones_triggered:
                now_ms = time.time() * 1000.0

                # Draw delayed clones from farthest delay to nearest so main person can be on top.
                for i, cl in sorted(enumerate(CUSTOM_CLONES), key=lambda x: x[1]["delay"], reverse=True):
                    if now_ms - clone_start_ms >= cl["delay"]:
                        cimg = cv2.resize(person, None, fx=cl["scale"], fy=cl["scale"], interpolation=cv2.INTER_LINEAR)
                        x = int(cl["x"] + (w * (1 - cl["scale"]) / 2))
                        y = int(cl["y"])
                        x2 = min(w, x + cimg.shape[1])
                        y2 = min(h, y + cimg.shape[0])
                        x1 = max(0, x)
                        y1 = max(0, y)
                        if x1 < x2 and y1 < y2:
                            out[y1:y2, x1:x2] = cimg[(y1 - y):(y2 - y), (x1 - x):(x2 - x)]

                        if not smoke_spawned[i]:
                            smoke_spawned[i] = True
                            center_x = int(cl["x"] + w / 2)
                            center_y = int(cl["y"] + h / 2 - 40)
                            spawn_smoke(center_x - 15, center_y, cl["scale"])
                            spawn_smoke(center_x + 15, center_y, cl["scale"])

                # Main person stays on top.
                mask = np.any(person > 0, axis=2)
                out[mask] = person[mask]

                # Smoke animation frames.
                for i in range(len(active_smokes) - 1, -1, -1):
                    smoke = active_smokes[i]
                    elapsed = now_ms - smoke.start_ms
                    frame_duration = smoke_duration_ms / 5
                    idx = int(elapsed // frame_duration)
                    if idx >= len(smoke.frames):
                        active_smokes.pop(i)
                        continue
                    blend_rgba(out, smoke.frames[idx], smoke.x, smoke.y, smoke.scale)

                if overlay_2 is not None:
                    blend_rgba(out, overlay_2, w // 2, h - 50, 0.5)
            else:
                out = person
                if overlay_1 is not None:
                    blend_rgba(out, overlay_1, w // 2, h - 50, 0.5)

            if right:
                draw_hand(out, right, mirrored=False)
            if left:
                draw_hand(out, left, mirrored=False)

            put_text(out, "Press q to quit", h - 10, (180, 180, 180), 0.5)
            cv2.imshow("Naruto Shadow Clone (Python)", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naruto Shadow Clone in Python")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Collect data and train model")
    p_train.add_argument("--data", type=Path, default=DATA_JSON)
    p_train.add_argument("--model-out", type=Path, default=MODEL_PATH)
    p_train.add_argument("--camera", type=int, default=0)

    p_run = sub.add_parser("run", help="Run clone effect")
    p_run.add_argument("--model", type=Path, default=MODEL_PATH)
    p_run.add_argument("--threshold", type=float, default=0.999)
    p_run.add_argument("--camera", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_mode(data_path=args.data, model_out=args.model_out, camera_id=args.camera)
    else:
        run_mode(model_path=args.model, threshold=args.threshold, camera_id=args.camera)


if __name__ == "__main__":
    main()