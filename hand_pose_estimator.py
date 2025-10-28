import cv2
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode
from orientation import convert_hand_landmarks, calculate_normal
from pose_estimator import PoseEstimator

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from opencv_capture import BufferlessCapture
from utils import calculate_rotation_matrix

import torch
device = torch.device('cuda')
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-small-hf", use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-small-hf").to(device)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


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
   
    def __init__(self, capture_name) -> None:
        super(HandPoseEstimator, self).__init__()
        self.cap = BufferlessCapture(capture_name)
        self.detector = MediaPipeHandPose(running_mode=VisionRunningMode.VIDEO)
        self.last_timestamp = None
        self.max_size = None
        self.min_size = None
        self.last_normal = None
        self.decay = 0.5
        self.normal_rot = None
        self.last_depth_image = None

        self.zero_pos = np.array([0,-0.5,0])

        self.position = None
        self.rotation = None

    def run(self):
        self.detect_body()

    def process_result(self, detection_result, rgb_image: np.ndarray):
        hand_landmarks, handedness = detection_result
        current_depth_image = get_depth(rgb_image)

        points = convert_hand_landmarks(hand_landmarks)
        normal = calculate_normal(points)

        if not self.last_normal is None:
            normal = self.decay*self.last_normal + (1-self.decay)*normal

        self.last_normal = normal
        if not self.normal_rot is None:
            normal = self.normal_rot @ normal

        rotation = R.from_matrix(calculate_rotation_matrix(normal))
        angles = rotation.as_euler("xyz")

        depth = current_depth_image
        if not self.last_depth_image is None:
            depth = self.last_depth_image * self.decay + (1-self.decay) * current_depth_image

        depth_height, depth_width = depth.shape
        self.last_depth_image = depth
        depth_values = np.array([depth[int(min(l.y, 1.0)*(depth_height-1)),int(min(1.0, l.x)*(depth_width-1))] for l in hand_landmarks])

        points = np.array([ [ (1-l.x) for l in hand_landmarks],
                            ((depth_values-0.5)*5).tolist(),
                            [ (1 - l.y) - 0.4 for l in hand_landmarks ]])
        transformed_points = points

        if not self.zero_pos is None:
            transformed_points -= self.zero_pos.reshape(3, 1)

        pts = np.array([ [l.x, l.y, l.z] for l in hand_landmarks ])

        center_of_palm = (pts[0, :] + pts[5, :] + pts[9, :] + pts[13, :] + pts[17, :]) / 5.0
        print(center_of_palm)
        center_depth = depth[int(min(center_of_palm[1], 1.0)*(depth_height-1)),int(min(1.0, center_of_palm[0])*(depth_width-1))]
        new_position = np.array([
            1 - center_of_palm[0],
            (center_depth - 0.5) * 2,
            (1 - center_of_palm[1]) - 0.4,
        ])
        new_rotation = angles

        self.current_position = np.concatenate([new_position, new_rotation])

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

    def detect_body(self):
        last_timestamp = None
        while not self.stop_requested:
            img = self.cap.get_frame()
      
            if img is None:
                time.sleep(0.1)
                continue

            hand_landmarker_result = self.detector.detect(img)
            if hand_landmarker_result is not None:
                self.process_result(hand_landmarker_result, img)
                hand_landmarks, handedness = hand_landmarker_result
                annotated_image = MediaPipeHandPose.annotate_image(img, hand_landmarks, handedness)
                cv2.imshow('Hand detection', annotated_image)
            else:
                cv2.imshow('Hand detection', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c') and not self.last_normal is None:
                self.normal_rot = calculate_rotation_matrix(self.last_normal)
            if key == ord('p') and not self.position is None:
                self.zero_pos = np.zeros(3)
                self.zero_pos[1] = self.position[1]
        return last_timestamp

if __name__ == "__main__":
    m = HandPoseEstimator("hand_landmarker.task")
    m.run()
