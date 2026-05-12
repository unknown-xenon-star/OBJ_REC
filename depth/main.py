import cv2


def measure_depth(frame, focal_length_px, real_object_width_cm):
    x, y, width, height = cv2.selectROI("Camera", frame, fromCenter=False, showCrosshair=False)
    if width <= 0 or height <= 0:
        return None

    distance_cm = (real_object_width_cm * focal_length_px) / float(width)
    return {
        "distance_cm": distance_cm,
        "roi": (x, y, width, height),
    }


def format_distance(distance_cm):
    return f"{distance_cm:.2f} cm"


def main():
    camera_index = 0
    focal_length_px = 700.0
    real_object_width_cm = 10.0

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    last_depth = None

    print("Press 's' to select an object and estimate depth.")
    print("Press 'q' or ESC to quit.")
    print(
        f"Current assumptions: focal length = {focal_length_px} px, "
        f"real object width = {real_object_width_cm} cm."
    )

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("Received an empty frame from the camera.")
            break

        display = frame.copy()

        if last_depth is not None:
            x, y, width, height = last_depth["roi"]
            cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"Estimated depth: {format_distance(last_depth['distance_cm'])}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            display,
            "Press s to measure depth, q to quit",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Camera", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

        if key == ord("s"):
            measured_depth = measure_depth(frame, focal_length_px, real_object_width_cm)
            if measured_depth is not None:
                last_depth = measured_depth
                print(f"Estimated depth: {format_distance(last_depth['distance_cm'])}")
            else:
                print("Depth measurement cancelled.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

