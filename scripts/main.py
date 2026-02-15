import cv2
import threading
import time
from ultralytics import YOLO


class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        # this gets the video from cam
        # 0 -> deafult webcam
        self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # set size of cam 
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # get frame and respone
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                return
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()


def run_inference():
    # 🔹 Use nano or small for real-time
    model = YOLO("models/yolov8x.pt")

    vs = VideoStream(src=0).start()

    prev_time = time.time()

    while True:
        frame = vs.read()
        if frame is None:
            continue

        # 🔹 YOLO inference (optimized)
        results = model.predict(
            source=frame,
            imgsz=320*2,
            conf=0.5,
            verbose=False,
            device=0 if model.device.type != "cpu" else "cpu"
        )

        # 🔹 FPS calculation (safe)
        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        annotated = results[0].plot()

        cv2.putText(
            annotated,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Optimized Stream", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
