import math

import cv2
import numpy as np

LOWER_BLUE = np.array([100, 150, 50], dtype=np.uint8)
UPPER_BLUE = np.array([140, 255, 255], dtype=np.uint8)
MIN_CONTOUR_AREA = 2
MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)


def _find_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA
    ]
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    centers: list[tuple[int, int]] = []
    for cnt in valid_contours[:2]:
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        centers.append((cx, cy))

    return centers

def plot(img: np.ndarray) -> tuple[np.ndarray, float | None]:
    annotated = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    centers = _find_centers(mask)

    for cx, cy in centers:
        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

    if len(centers) < 2:
        return annotated, None

    p1, p2 = centers
    distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    cv2.line(annotated, p1, p2, (255, 0, 0), 2)

    return annotated, distance


