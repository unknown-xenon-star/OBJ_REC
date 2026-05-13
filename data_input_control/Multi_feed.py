import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_URL = "http://192.168.1.9:8080/video"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stereo depth estimation from an IP camera feed and a laptop camera."
    )
    parser.add_argument(
        "--ip-url",
        default=DEFAULT_URL,
        help="IP camera stream URL.",
    )
    parser.add_argument(
        "--laptop-camera",
        type=int,
        default=0,
        help="OpenCV device index for the laptop camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Processing width for both camera frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Processing height for both camera frames.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("stereo_calibration.npz"),
        help="Optional stereo calibration file with rectification maps.",
    )
    return parser.parse_args()


def open_stream(source, width, height):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def load_rectification_maps(calibration_path):
    if not calibration_path.exists():
        return None

    data = np.load(str(calibration_path))
    required = ("left_map_x", "left_map_y", "right_map_x", "right_map_y")
    if not all(key in data for key in required):
        raise ValueError(
            f"Calibration file {calibration_path} is missing one of: {', '.join(required)}"
        )

    return tuple(data[key] for key in required)


def rectify_frames(left_frame, right_frame, rectification_maps):
    if rectification_maps is None:
        return left_frame, right_frame

    left_map_x, left_map_y, right_map_x, right_map_y = rectification_maps
    left_rectified = cv2.remap(left_frame, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_frame, right_map_x, right_map_y, cv2.INTER_LINEAR)
    return left_rectified, right_rectified


def crop_to_common_size(left_frame, right_frame):
    height = min(left_frame.shape[0], right_frame.shape[0])
    width = min(left_frame.shape[1], right_frame.shape[1])
    return left_frame[:height, :width], right_frame[:height, :width]


def build_stereo_matcher():
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16 * 8,
        blockSize=5,
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def normalize_disparity(disparity):
    disparity = disparity.astype(np.float32) / 16.0
    normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def add_status(frame, text, line=30, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (10, line),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    ip_cap = open_stream(args.ip_url, args.width, args.height)
    laptop_cap = open_stream(args.laptop_camera, args.width, args.height)

    if not ip_cap.isOpened():
        raise RuntimeError(f"Could not open IP feed: {args.ip_url}")
    if not laptop_cap.isOpened():
        raise RuntimeError(f"Could not open laptop camera index: {args.laptop_camera}")

    try:
        rectification_maps = load_rectification_maps(args.calibration)
    except ValueError as exc:
        ip_cap.release()
        laptop_cap.release()
        raise exc

    stereo = build_stereo_matcher()
    using_calibration = rectification_maps is not None

    while True:
        ip_ok, ip_frame = ip_cap.read()
        laptop_ok, laptop_frame = laptop_cap.read()

        if not ip_ok or not laptop_ok:
            break

        ip_frame = cv2.resize(ip_frame, (args.width, args.height))
        laptop_frame = cv2.resize(laptop_frame, (args.width, args.height))
        ip_frame, laptop_frame = crop_to_common_size(ip_frame, laptop_frame)
        ip_frame, laptop_frame = rectify_frames(ip_frame, laptop_frame, rectification_maps)

        left_gray = cv2.cvtColor(ip_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(laptop_frame, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(left_gray, right_gray)
        disparity_view = normalize_disparity(disparity)
        disparity_view = cv2.applyColorMap(disparity_view, cv2.COLORMAP_TURBO)

        preview = np.hstack((ip_frame, laptop_frame))
        add_status(preview, "Left: IP feed | Right: laptop camera")

        if using_calibration:
            add_status(preview, "Rectified with stereo_calibration.npz", line=55)
        else:
            add_status(
                preview,
                "No calibration loaded: depth is approximate only",
                line=55,
                color=(0, 165, 255),
            )

        cv2.imshow("Stereo Inputs", preview)
        cv2.imshow("Stereo Depth", disparity_view)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    ip_cap.release()
    laptop_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
