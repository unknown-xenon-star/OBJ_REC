import cv2

from one_way import plot
from app_depth import real_distance

def main(camera_index: int) -> None:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Received an empty frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            annotated_frame, distance = plot(frame)

            
            center_text = (
                f"Distance: {real_distance(distance):.2f}" if distance is not None else "Distance: N/A"
            )
            cv2.putText(
                annotated_frame,
                center_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Depth", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)
