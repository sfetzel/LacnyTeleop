import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pose_estimator import MockEstimator, CircleEstimator, RotatorEstimator
from hand_pose_estimator import HandPoseEstimator

estimator = HandPoseEstimator()
#estimator = RotatorEstimator(np.array([0.1,0.1,0.1]))
estimator.start()

visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
visualizer.add_geometry(mesh_frame)

box_size = 0.1
target_box :o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh().create_box(box_size*0.95, box_size*0.95, box_size*0.95)
target_box.translate(np.array([-box_size / 2, -box_size / 2, -box_size / 2]), relative=True)
target_pos = np.array([0.25, 0.25, 0])
target_box.translate(target_pos, relative=False)
target_box.paint_uniform_color(np.array([1.0, 0, 0]))

target_box.rotate(target_box.get_rotation_matrix_from_xyz([0, 25.0, 0]))
target_box.rotate(target_box.get_rotation_matrix_from_xyz([0, 25.0, 0]))
target_box.rotate(target_box.get_rotation_matrix_from_xyz([0, 25.0, 0]))

pose_box = o3d.geometry.TriangleMesh().create_box(box_size, box_size, box_size)
pose_box.paint_uniform_color(np.array([0.0, 0, 1.0]))

prev_rotation = None

def update_box(vis) -> bool:
    global prev_rotation
    target = estimator.current_position
    if not target is None:
        pose_box.translate(target[:3], relative=False)
        if prev_rotation is not None:
            pose_box.rotate(np.linalg.inv(prev_rotation))
        rot = R.from_euler('xyz', target[3:], degrees=False)
        rot_matrix = rot.as_matrix()
        pose_box.rotate(rot_matrix)
        prev_rotation = rot_matrix
        vis.update_geometry(pose_box)

        distance = np.linalg.norm(target[:3] - target_pos)
        if distance < 0.4:
            print(f"Distance to target: {distance}")

    time.sleep(0.1)
    return not target is None

visualizer.add_geometry(target_box)
visualizer.add_geometry(pose_box)
visualizer.register_animation_callback(update_box)
visualizer.poll_events()
visualizer.update_renderer()
visualizer.run()
estimator.stop()