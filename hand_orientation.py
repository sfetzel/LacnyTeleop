import cv2
import numpy as np
import pytest

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os

from scipy.spatial.transform import Rotation as R

from orientation import calculate_normal, convert_hand_landmarks


def calc_normal(a, b, c):
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)
    v1 = b-a
    v2 = c-a
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def calculate_rotation_matrix(unit_normal: np.ndarray, rotate_into: np.ndarray = None) -> np.ndarray:
    if rotate_into is None:
        rotate_into = np.array([0,0,1])
    v = np.cross(unit_normal, rotate_into)
    c = np.dot(unit_normal, rotate_into)

    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * 1/(1+c)
    return R

def cart2sph(vec):
    x,y,z = vec
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    polar = np.arccos(z/r) /np.pi * 180     # theta
    az = np.arctan2(y,x) /np.pi*180                           # phi
    return r, polar, az


VisionRunningMode = mp.tasks.vision.RunningMode
options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
                                       running_mode=VisionRunningMode.IMAGE,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.3,
                                       min_hand_presence_confidence=0.3,
                                       min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)
test_folder = "test_images"
images = os.listdir(test_folder)

def get_hand_landmarks(image):
    img = cv2.imread(os.path.join(test_folder, image))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    hand_landmarker_result = detector.detect(mp_image)

    if len(hand_landmarker_result.hand_landmarks) == 0:
        raise Exception(f"no hand found in {image}")

    return img, hand_landmarker_result.hand_landmarks[0], hand_landmarker_result.handedness[0]


def to_text(vector):
    return f"({vector[0]}, {vector[1]}, {vector[2]})"

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
CAPTION_TEXT_COLOR = (0, 0, 0) # vibrant green
def create_debug_image(image, hand_landmarks, handedness, filename):
    annotated_image = image.copy()
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
              (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
              FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    for i, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.putText(annotated_image, f"{i}",
                (x, y), cv2.FONT_HERSHEY_DUPLEX,
                0.5, CAPTION_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    cv2.imwrite(os.path.join(test_folder, filename), annotated_image)
z_dir = np.array([0,0,1])
y_dir = np.array([0,1,0])
x_dir = np.array([1,0,0])

test_data = [("hand-posz.png", z_dir),
             ("hand-posz2.png", z_dir),
             ("hand-posz3.png", z_dir),
             ("hand-posz4.png", z_dir),
             ("hand-posz5.png", z_dir),
             ("hand-negy.png", -y_dir),
             ("hand-negy2.png", -y_dir),
             ("hand-posy.png", y_dir),
             ("hand-posx.png", x_dir),]

@pytest.mark.parametrize("image,expected_normal", test_data)
def test_direction(image, expected_normal):
    img, landmarks, handedness = get_hand_landmarks(image)
    points = convert_hand_landmarks(landmarks)
    normal = calculate_normal(points)
    distance = np.linalg.norm(expected_normal - normal)

    create_debug_image(img, landmarks, handedness, f"{image}-debug.png")
    assert distance < 8e-1, f"Distance is too large: {distance}, normal: {to_text(normal)}"

@pytest.mark.parametrize("image,expected_polar,expected_azimuth", [
    ("hand-negx-45.png", 45, 180),
    ("hand-negx2-45.png", 45, 180),
    ("hand-posx-45.png", 45, 0),
])
def test_angles(image, expected_polar, expected_azimuth):
    img, landmarks, handedness = get_hand_landmarks(image)
    points = convert_hand_landmarks(landmarks)
    normal = calculate_normal(points)
    _, polar, az = cart2sph(normal)
    assert abs(polar - expected_polar) < 20, f"Polar is {polar}, but expected is {expected_polar}"
    assert abs(az - expected_azimuth) < 50, f"Azimuth is {az}, but expected is {expected_azimuth}"

