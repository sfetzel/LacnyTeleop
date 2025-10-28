import cv2
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from .depth_estimator import DepthAnythingEstimator
from .hands_detection.mp_hands import MediaPipeHandPose, VisionRunningMode
from .orientation import convert_hand_landmarks, calculate_normal
from .pose_estimator import PoseEstimator, GripperState

from .utils.opencv_capture import BufferlessCapture
from .utils.utils import calculate_rotation_matrix, to_image_indices


class HandPoseEstimator(PoseEstimator):

    def __init__(self, capture_name, stretch_factors: list = None) -> None:
        super(HandPoseEstimator, self).__init__()
        self.cap = BufferlessCapture(capture_name)
        self.detector = MediaPipeHandPose(running_mode=VisionRunningMode.VIDEO)
        self.last_normal = None
        self.decay = 0.25
        self.normal_rot = None
        self.is_gripper_closed = False
        self.depth_estimator = DepthAnythingEstimator(True, self.decay)
        self.finger_distance_threshold = 0.07

        self.zero_pos = np.array([0, 0.5, 0.5])
        self.stretch_factors = np.array(stretch_factors if stretch_factors is not None else [1.0, 2.0, 1.0])

    def process_result(self, detection_result, rgb_image: np.ndarray):
        hand_landmarks, handedness = detection_result
        depth = self.depth_estimator.get_depth(rgb_image)

        points = convert_hand_landmarks(hand_landmarks)
        normal = calculate_normal(points)
        if not self.last_normal is None:
            normal = self.decay * self.last_normal + (1 - self.decay) * normal

        self.last_normal = normal
        if not self.normal_rot is None:
            normal = self.normal_rot @ normal

        rotation = R.from_matrix(calculate_rotation_matrix(normal))
        angles = rotation.as_euler("xyz")

        depth_height, depth_width = depth.shape
        pts = np.array([[l.x, l.y, l.z] for l in hand_landmarks])

        index_to_thumb_distance = np.linalg.norm(pts[8, :] - pts[4, :])
        middle_to_thumb_distance = np.linalg.norm(pts[4, :] - pts[12, :])
        ring_to_thumb_distance = np.linalg.norm(pts[4, :] - pts[16, :])
        mean_distance = (index_to_thumb_distance + middle_to_thumb_distance + ring_to_thumb_distance) / 3.0
        new_is_gripper_closed = mean_distance < self.finger_distance_threshold
        self.is_gripper_closed = new_is_gripper_closed

        center_of_palm = (pts[0, :] + pts[5, :] + pts[9, :] + pts[13, :] + pts[17, :]) / 5.0
        depth_y = to_image_indices(center_of_palm[1], depth_height)
        depth_x = to_image_indices(center_of_palm[0], depth_width)
        center_depth = depth[depth_y, depth_x]
        new_location = np.array([
            1 - center_of_palm[0],
            center_depth,
            (1 - center_of_palm[1]),
        ])
        new_location -= self.zero_pos
        new_location = new_location * self.stretch_factors
        gripper_value = GripperState.Closed.value if self.is_gripper_closed else GripperState.Open.value
        new_rotation = angles

        new_position = np.concatenate([new_location, new_rotation, np.array([gripper_value])])
        self.set_position_and_update_deltas(new_position)

    def run(self):
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
            if key == ord('p') and not self.current_position is None:
                self.zero_pos = self.current_position[:3]
        return last_timestamp


if __name__ == "__main__":
    m = HandPoseEstimator(0)
    m.run()
