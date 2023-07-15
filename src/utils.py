import cv2
import numpy as np


import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from typing import List, Dict

from mediapipe.tasks.python.vision import FaceLandmarkerResult

LANDMARKS_UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 310, 311, 312, 13, 82, 81, 80, 191, 78,]
LANDMARKS_BOTTOM_LIP =  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]            

LANDMARKS_CHEEKS = [425, 205]

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3
def draw_landmarks_on_image(rgb_image: np.array, detection_result: FaceLandmarkerResult):

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
        print(face_landmarks_proto)
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


def apply_makeup(img: np.ndarray, detection_result: FaceLandmarkerResult):#, feature: str, show_landmarks: bool = False):
    """
    Takes in a source image and applies effects onto it.
    """
    output = np.copy(img)

    if detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            #print('*'*50)
            """
            mask = mask_landmarks(image=img,
                        landmark_list=face_landmarks_proto,
                        connections=upper_lip,#mp.solutions.face_mesh.FACEMESH_LIPS,
                        connection_drawing_spec=DrawingSpec(color=(0,255,0)))
            """
            idx_to_coordinates = landmarks_to_px(image=img,
                        landmark_list=face_landmarks_proto)
            
            points_bottom_lip = filter_points(idx_to_coordinates, LANDMARKS_BOTTOM_LIP)
            points_upper_lip = filter_points(idx_to_coordinates, LANDMARKS_UPPER_LIP)

            points_cheeks = filter_points(idx_to_coordinates, LANDMARKS_CHEEKS)

            mask = lipstick(img, points_bottom_lip, (93,33,44))
            output = cv2.addWeighted(img, 1.0, mask, 0.4, 0.0)

            mask = lipstick(img, points_upper_lip, (193,104,115))
            output = cv2.addWeighted(output, 1.0, mask, 0.4, 0.0)

    return output
def landmarks_to_px(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList) -> Dict:

    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    return idx_to_coordinates

def filter_points(idx_to_coordinates: dict, connections: list[int]) -> np.ndarray:

    points = [idx_to_coordinates[idx] for idx in connections]
        
    return np.array(points)

def lipstick(src, points: np.ndarray, color) -> np.ndarray:
    mask = np.zeros_like(src)
    cv2.fillPoly(mask, [points], color)
    cv2.GaussianBlur(mask, (7,7), 5)
    return mask

