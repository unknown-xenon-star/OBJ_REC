import cv2
import numpy as np
import math

img = cv2.imread("images.png")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Red color ranges (red wraps around HSV hue scale)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Create masks
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = mask1 + mask2
# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centers = []

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 20:  # ignore tiny noise
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            centers.append((cx, cy))

            # Draw center
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

# Need exactly 2 dots
if len(centers) == 2:
    p1 = centers[0]
    p2 = centers[1]

    # Euclidean distance
    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    print("Distance:", distance)

    # Draw line
    cv2.line(img, p1, p2, (255, 0, 0), 2)

# Show result
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
