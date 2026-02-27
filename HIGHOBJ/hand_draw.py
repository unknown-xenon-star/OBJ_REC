import numpy as np
import cv2
import mediapipe as mp

FINGER_SEGMENTS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]

def draw_hand(frame: np.ndarray, hand_lms, mirrored: bool = False) -> None:
    h,w = frame.shape[:2]
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


def run_mode(camera_id):
    
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
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
            
            cv2.imshow("Naruto Trainer (Python)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()