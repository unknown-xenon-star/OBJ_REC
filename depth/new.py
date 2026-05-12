import cv2
import numpy as np
from ex import depth

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

        # cv2.imshow("Camera", Frame)

        depth_map = depth(frame)
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)
        
        depth_view = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        
        cv2.imshow("Camera", frame)
        cv2.imshow("Depth", depth_view)
        
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
