import cv2
import queue

def start_capture(frame_queue, stop_flag):
    cap = cv2.VideoCapture(0)       # Laptop webcam

    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)


    cap.release() 