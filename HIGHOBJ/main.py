import threading
import queue
import cv2
from capture import start_capture
from preprocess import preprocess
from model import infer
from postprocess import postprocess

frame_queue = queue.Queue(maxsize=5)
stop_flag = threading.Event()

capture_thread = threading.Thread(
    target=start_capture,
    args=(frame_queue, stop_flag),
    daemon=True
)
capture_thread.start()

while True:
    if not frame_queue.empty():
        frame = frame_queue.get()

        input_tensor = preprocess(frame)
        # prediction = infer(input_tensor)
        # result = postprocess(prediction)
        result = 


        # cv2.putText(frame, result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ML Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_flag.set()
        break

cv2.destroyAllWindows()