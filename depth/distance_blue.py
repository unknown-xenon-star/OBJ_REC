import cv2
import numpy as np
import math

# Load image
img = cv2.imread("blue_dots.jpg")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Blue color range
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centers = []

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 20:
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            centers.append((cx, cy))

            # Draw center point
            cv2.circle(img, (cx, cy), 6, (0, 255, 0), -1)

# If exactly 2 blue dots found
if len(centers) == 2:

    p1 = centers[0]
    p2 = centers[1]

    # Calculate distance
    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    # Draw line between dots
    cv2.line(img, p1, p2, (0, 0, 255), 2)

    # Midpoint for text
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)

    # Show distance on image
    cv2.putText(
        img,
        f"{distance:.2f}px",
        (mid_x, mid_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    print("Distance:", distance)

# Show output
cv2.imshow("Result", img)
cv2.imshow("Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
