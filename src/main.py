import numpy as np
import cv2
import queue
import threading

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "model/face_landmarker.task"

def main():


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
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(),face_landmarks_result)

                cv2.imshow('video', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        stream.release()
        cv2.destroyAllWindows()


def draw_landmarks_on_image(rgb_image, detection_result):
  annotated_image = np.copy(rgb_image)
  if detection_result.face_landmarks:
    face_landmarks_list = detection_result.face_landmarks

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
  return annotated_image

if __name__ == '__main__':
    queue = queue.Queue() 

    # Create a thread for running the FaceLandmarker
    face_landmarker_thread = threading.Thread(target=main)

    face_landmarker_thread.start()
    


"""
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
MODEL_PATH = "model/face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode
def draw_landmarks(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    annotated_image = np.copy(output_image.numpy_view())
    if result.face_landmarks:
        face_landmarks_list = result.face_landmarks
    image_result = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("frame", image_result)
    cv2.waitKey(1)
options = FaceLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = MODEL_PATH),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = draw_landmarks
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    stream.release()
    cv2.destroyAllWindows()
"""
"""
  @classmethod
  def create_from_options(
      cls, options: FaceLandmarkerOptions
  ) -> 'FaceLandmarker':
    def packets_callback(output_packets: Mapping[str, packet_module.Packet]):
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return
      image = packet_getter.get_image(output_packets[_IMAGE_OUT_STREAM_NAME])
      if output_packets[_IMAGE_OUT_STREAM_NAME].is_empty():
        return
      if output_packets[_NORM_LANDMARKS_STREAM_NAME].is_empty():
        empty_packet = output_packets[_NORM_LANDMARKS_STREAM_NAME]
        options.result_callback(
            FaceLandmarkerResult([], [], []),
            image,
            empty_packet.timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
        )
        return
      face_landmarks_result = _build_landmarker_result(output_packets)
      timestamp = output_packets[_NORM_LANDMARKS_STREAM_NAME].timestamp
      options.result_callback(
          face_landmarks_result,
          image,
          timestamp.value // _MICRO_SECONDS_PER_MILLISECOND,
      )
    output_streams = [
        ':'.join([_NORM_LANDMARKS_TAG, _NORM_LANDMARKS_STREAM_NAME]),
        ':'.join([_IMAGE_TAG, _IMAGE_OUT_STREAM_NAME]),
    ]
    if options.output_face_blendshapes:
      output_streams.append(
          ':'.join([_BLENDSHAPES_TAG, _BLENDSHAPES_STREAM_NAME])
      )
    if options.output_facial_transformation_matrixes:
      output_streams.append(
          ':'.join([_FACE_GEOMETRY_TAG, _FACE_GEOMETRY_STREAM_NAME])
      )
    task_info = _TaskInfo(
        task_graph=_TASK_GRAPH_NAME,
        input_streams=[
            ':'.join([_IMAGE_TAG, _IMAGE_IN_STREAM_NAME]),
            ':'.join([_NORM_RECT_TAG, _NORM_RECT_STREAM_NAME]),
        ],
        output_streams=output_streams,
        task_options=options,
    )
    return cls(
        task_info.generate_graph_config(
            enable_flow_limiting=options.running_mode
            == _RunningMode.LIVE_STREAM
        ),
        options.running_mode,
        packets_callback if options.result_callback else None,
    )
"""