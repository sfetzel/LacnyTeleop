import numpy as np
import cv2
import time
import torch

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.spatial.transform import Rotation as R
from pose_estimator import PoseEstimator

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from opencv_capture import BufferlessCapture

device = torch.device('cuda')
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-small-hf", use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-small-hf").to(device)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


world_R = np.array([[ 0.99425761,  0.04218685, -0.09834668],
 [ 0.04218685,  0.69007138,  0.72251073],
 [ 0.09834668, -0.72251073,  0.68432899]])


def calc_normal(a, b, c):
    v1 = a-b
    v2 = a-c
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def landmark_dist(first, second):
    difference = np.array([first.x-second.x, first.y-second.y, first.z-second.z])
    return np.linalg.norm(difference)

def calculate_rotation_matrix(unit_normal: np.ndarray) -> np.ndarray:
    rotate_into = np.array([0,0,1])
    v = np.cross(unit_normal, rotate_into)
    s = np.linalg.norm(v)
    c = np.dot(unit_normal, rotate_into)

    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * 1/(1+c)
    return R

def get_depth(image):
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # interpolate to original size and visualize the prediction
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.shape[0], image.shape[1])],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())

    depth = depth.detach().cpu().numpy()
    return depth

class HandPoseEstimator(PoseEstimator):
   
  def __init__(self, model_path_full) -> None:
    super(HandPoseEstimator, self).__init__()
    # Initialization of the image
    self.annotated_image = np.zeros((640,480,3), np.uint8)

    self.cap = cv2.VideoCapture(2)
    #cap = cv2.VideoCapture("hand_forward.mp4")
    
    #
    
    # Create an HandLandmarker object
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path_full),
                                        running_mode=VisionRunningMode.VIDEO,
                                        num_hands=2,
                                        min_hand_detection_confidence=0.5,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.5)
    self.detector = vision.HandLandmarker.create_from_options(options)
    self.last_timestamp = None
    self.max_size = None
    self.min_size = None
    self.last_normal = None
    self.decay = 0.95
    self.normal_rot = None

    self.position = None
    self.rotation = None

  #

  def run(self):
    last_timestamp = self.detect_body()
  #

  def draw_hand_landmarks_on_live(self, detection_result, rgb_image, timestamp):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    rgb_image = rgb_image.numpy_view()
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]

      hand_points = [ np.array([l.x, l.y, l.z]) for l in hand_landmarks ]

      normals = [
        calc_normal(hand_points[0], hand_points[5], hand_points[9]),
        calc_normal(hand_points[0], hand_points[9], hand_points[13]),
        calc_normal(hand_points[0], hand_points[13], hand_points[17]),
        calc_normal(hand_points[0], hand_points[5], hand_points[17]),
      ]
      normal = np.array(normals).mean(axis=0)
      normal = normal / np.linalg.norm(normal)
      #normal = np.round(normal * 8) / 8
      
      if not self.last_normal is None:
        normal = self.decay*self.last_normal + (1-self.decay)*normal

      self.last_normal = normal
      if not self.normal_rot is None:
        normal = self.normal_rot @ normal

      rotation = R.from_matrix(calculate_rotation_matrix(normal))
#      print(normal)
      angles = rotation.as_euler("xyz")
      
      self.depth = get_depth(annotated_image)
      self.update_depth = 0
      
      depth_height, depth_width = self.depth.shape
      print(f"Depth shape: {self.depth.shape}")
      print(f"input shape: {rgb_image.shape}")
      points = np.array([ [ l.x for l in hand_landmarks],
                [ l.y for l in hand_landmarks],
                [ self.depth[int(min(l.y, 1.0)*(depth_height-1)),int(min(1.0, l.x)*(depth_width-1))] for l in hand_landmarks]  ])
      transformed_points = world_R @ points
      
      new_position = points[:, 0]
      new_rotation = angles
      
      if not self.position is None and not self.rotation is None:
        delta_pos = new_position - self.position
        delta_rot = new_rotation - self.rotation
        
        delta = np.concatenate([delta_pos, delta_rot])
        if self.latest_deltas is None:
            self.latest_deltas = delta
        else:
            self.latest_deltas += delta

      print(f"Pos: {new_position}")
      print(f"Rot: {new_rotation}")
      
      self.position = new_position
      self.rotation = new_rotation
      
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

    self.annotated_image = annotated_image
    self.last_timestamp = timestamp
    return

  #
  
  def detect_body(self):
    last_timestamp = None
    while not self.stop_requested:
      success, img = self.cap.read()
      
      if not success:
        print("fail")
        break
      img = cv2.flip(img, 1)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

      timestamp = int(round(time.time()*1000))

      hand_landmarker_result = self.detector.detect_for_video(mp_image, timestamp) #detect_async
      self.draw_hand_landmarks_on_live(hand_landmarker_result, mp_image, timestamp)
      last_timestamp = timestamp

      cv2.imshow('Hand detection', self.annotated_image)
      
      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
          break
      if key == ord('c') and not self.last_normal is None:
        self.normal_rot = calculate_rotation_matrix(self.last_normal)
    return last_timestamp

if __name__ == "__main__":
    m = HandPoseEstimator("hand_landmarker.task")
    m.run()
