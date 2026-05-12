import cv2
import torch
import matplotlib.pyplot as plt

# Load model
model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()

# Transforms
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = transforms.small_transform

def depth(img_data):    
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    # Prepare input
    input_batch = transform(img)

    # Prediction
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    return depth_map


def depth_of_img(img_data_file):

    # Read image
    img = cv2.imread(img_data_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare input
    input_batch = transform(img)

    # Prediction
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Show
    plt.imshow(depth_map)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    depth("image.jpg")
