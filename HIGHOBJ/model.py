import numpy as np

def infer(input_tensor):
    # Simulate ML inference
    prediction = {
        "label": "person",
        "confidence": float(np.random.uniform(0.8, 0.99))
    }
    return prediction