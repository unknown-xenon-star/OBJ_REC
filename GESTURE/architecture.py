import time

import cv2
import numpy as np
import mediapipe as mp

# def process(frame):
#     rbg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



def main() -> None:
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
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
        
        cv2.imshow("Cam", frame)
        # Convert the BGR image to RGB before processing with MediaPipe
        
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
