import time

import cv2
import numpy as np
import mediapipe as mp

# def process(frame):
#     rbg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



def main() -> None:
    cap = cv2.VideoCapture(0)
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
        
            ok, frame = cap.read()
            if not ok: break
            
            
            frame_count = (frame_count + 1) % 10
            if frame_count == 0:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            frame.flags.writeable = True
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        new_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
            
            cv2.imshow("Cam", new_frame)
            # Convert the BGR image to RGB before processing with MediaPipe

            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # process(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# frame.flags.writeable = False