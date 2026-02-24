def postprocess(prediction):
    if prediction["confidence"] > 0.82:
        return f'Detected {prediction["label"]}'
    return "No Strong detection"