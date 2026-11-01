"""
驱动层：电机基类。
舵机（servo）和 DM 电机（达妙无刷）都继承 MotorDriver，
把"发一条指令给电机"的通信细节封装在子类里。
"""
from abc import ABC, abstractmethod
import numpy as np


class MotorDriver(ABC):
    """单个关节电机的抽象。"""

    def __init__(self, motor_id):
        self.motor_id = motor_id

    @abstractmethod
    def set_position(self, angle):
        """设目标角度（弧度）。"""

    @abstractmethod
    def read_position(self):
        """读当前角度（弧度）。"""

    @abstractmethod
    def set_velocity(self, vel):
        """设目标速度（弧度/秒）。"""

    @abstractmethod
    def set_torque(self, torque):
        """设目标力矩（Nm）。舵机一般不支持，可 raise NotImplementedError。"""

    @abstractmethod
    def enable(self):
        """使能电机。"""

    @abstractmethod
    def disable(self):
        """失能电机（力矩释放）。"""
