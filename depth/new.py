import cv2
import numpy as np
# from ex import depth
from blue_dots import plot


def main(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    while True:
        ok, frame = camera.read()
        
        if not ok or frame is None:
            print("Received an empty frame from the camera.")
            break

        frame = cv2.flip(frame,1)

        # raw_depth_map = depth(frame)
        # depth_map = cv2.normalize(raw_depth_map, None, 0, 255, cv2.NORM_MINMAX)
        # depth_map = depth_map.astype(np.uint8)
        
        # depth_view = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        # center_y = raw_depth_map.shape[0] // 2
        # center_x = raw_depth_map.shape[1] // 2
        # center_depth = float(raw_depth_map[center_y, center_x])

        img, Distance = plot(frame)
        center_text = f"Distance: {Distance:.2f}"

        # cv2.drawMarker(
        #     depth_view,
        #     (center_x, center_y),
        #     (255, 255, 255),
        #     markerType=cv2.MARKER_CROSS,
        #     markerSize=20,
        #     thickness=2,
        # )
        cv2.putText(
            # depth_view,
            img,
            center_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        
        # cv2.imshow("Camera", frame)
        cv2.imshow("Depth", img)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
