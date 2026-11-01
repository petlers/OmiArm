"""
机械臂统一接口（核心抽象）。
算法层（运动学 / 状态机 / 任务）只依赖这个接口，
不关心底层是仿真、舵机还是 DM 电机。

换硬件时只需实现 ArmInterface 的一个新子类：
    - drivers/simulated.py      仿真
    - drivers/servo_driver.py   舵机（串口）
    - drivers/dm_motor_driver.py DM 电机（CAN）
"""
from abc import ABC, abstractmethod
import numpy as np


class ArmInterface(ABC):
    """7-DOF 机械臂统一接口。"""

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """自由度数量（如 7）。"""

    @abstractmethod
    def goto(self, joint_angles, duration=None) -> None:
        """
        位置伺服：把机械臂移动到指定关节角。
        这是最常用的接口 —— 舵机和 DM 电机都支持位置模式。
        Args:
            joint_angles: 形状 (n_dof,) 的目标关节角（弧度）
            duration: 可选的移动时间，None 表示立即执行
        """

    @abstractmethod
    def get_joint_state(self) -> tuple:
        """
        读当前关节状态。
        Returns:
            (positions, velocities) —— 两个形状 (n_dof,) 的数组（弧度 / 弧度每秒）
        """

    @abstractmethod
    def set_torque(self, torques) -> None:
        """
        力矩控制（DM 电机支持；舵机一般不支持，可 raise NotImplementedError）。
        Args:
            torques: 形状 (n_dof,) 的目标关节力矩（Nm）
        """

    @abstractmethod
    def reset(self) -> None:
        """复位到初始位姿。"""

    # ---------------- 派生工具 ----------------

    def goto_pose(self, kine, pos, orn=None):
        """
        位姿伺服：结合运动学模块，把末端移动到目标位姿。
        这层才是算法真正常用的入口 —— 上层只给"末端想去哪"，
        IK 和关节驱动在这里串起来。
        """
        q = kine.inverse(pos, orn, q_init=self.get_joint_state()[0])
        self.goto(q)
        return q

    def get_joint_positions(self):
        """便捷方法：只取当前关节角。"""
        return np.array(self.get_joint_state()[0], dtype=float)
