import cv2
import numpy as np

def preprocess(frame , DIMENSIONS=(224,224)):
    HEIGHT, WEIGHT = DIMENSIONS 
    frame = cv2.resize(frame, (HEIGHT, WEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)