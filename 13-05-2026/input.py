import cv2

def flip(img_data, axis=1):
    return cv2.flip(img_data, axis)

def main(device_id=0):
    capture = cv2.VideoCapture(device_id)

    while True:
        isTrue, frame = capture.read()
        
        if not isTrue:
            break

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        cv2.imshow("Live Feed", flip(frame))

    capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main(device_id=0)
