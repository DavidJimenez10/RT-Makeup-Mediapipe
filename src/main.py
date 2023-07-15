import numpy as np
import cv2
import queue
import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import draw_landmarks_on_image, apply_makeup

MODEL_PATH = "model/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode

    def pass_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result:
            queue.put(result)

    options = FaceLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = MODEL_PATH),
        running_mode = VisionRunningMode.LIVE_STREAM,
        result_callback = pass_result
    )


    with FaceLandmarker.create_from_options(options) as landmarker:

        stream = cv2.VideoCapture(0)
        i = 0
        while True:
            ret, frame = stream.read()
            if ret:
                frame_timestamp = stream.get(cv2.CAP_PROP_POS_MSEC)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format = mp.ImageFormat.SRGB,
                    data = rgb_frame
                )
                i += 1
                landmarker.detect_async(mp_image, i)#int(frame_timestamp))

                face_landmarks_result = queue.get()

                #print(face_landmarks_result)
                #annotated_image = draw_landmarks_on_image(mp_image.numpy_view(),face_landmarks_result)
                annotated_image = apply_makeup(mp_image.numpy_view(),face_landmarks_result)
                cv2.imshow('video', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        stream.release()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    queue = queue.Queue() 

    # Create a thread for running the FaceLandmarker
    face_landmarker_thread = threading.Thread(target=main)

    face_landmarker_thread.start()
    