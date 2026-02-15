import cv2

from ultralytics import YOLO

model = YOLO("yolov8m.pt")          # n matlab nano (fast)

def main() -> None:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0).")
    
    # better optimization kar 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        result = model.predict(
            source=frame,
            conf=0.5,
            imgsz = 640,
            verbose = False
        )
        cv2.imshow("Camera Feed", result[0].plot())

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
