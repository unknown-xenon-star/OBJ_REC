import cv2
import numpy as np
import math

# Load image

def plot(img,distance=0):
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

        if area > 20:  # ignore noise
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

        # print("Distance:", distance)

        # Draw line
        cv2.line(img, p1, p2, (255, 0, 0), 2)

    # Show result
    # cv2.imshow("Result", img)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img, distance

if __name__ == "__main__":
    img = cv2.imread("dots.png")
