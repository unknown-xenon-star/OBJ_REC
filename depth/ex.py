import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


MODEL_REPO = "isl-org/ZoeDepth"
MODEL_NAME = "ZoeD_NK"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_zoe_model = None


def _get_model():
    global _zoe_model

    if _zoe_model is None:
        _zoe_model = torch.hub.load(MODEL_REPO, MODEL_NAME, pretrained=True)
        _zoe_model = _zoe_model.to(DEVICE).eval()

    return _zoe_model


def depth(img_data):
    rgb_image = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    model = _get_model()
    pil_image = Image.fromarray(rgb_image)

    with torch.no_grad():
        depth_map = model.infer_pil(pil_image)

    return np.asarray(depth_map, dtype=np.float32)


def depth_of_img(img_data_file):
    img = cv2.imread(img_data_file)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_data_file}")

    depth_map = depth(img)

    plt.imshow(depth_map)
    plt.colorbar()
    plt.show()

    return depth_map


if __name__ == "__main__":
    depth_of_img("image.jpg")
