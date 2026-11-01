"""
工具：PyBullet 可视化辅助。
"""
import pybullet as p
import pybullet_data


def connect_gui(camera_distance=1.5, camera_yaw=50, camera_pitch=-35):
    """连接 GUI 并设置相机视角。"""
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, [0, 0, 0.3])
    return None


def load_plane():
    """加载地面（pybullet_data 自带）。"""
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return p.loadURDF("plane.urdf")
