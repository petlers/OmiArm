"""
驱动层：舵机驱动（真机 stub）。

舵机特点：
    - 控制指令是【目标角度】（位置模式），内部自带位置闭环
    - 通信：单线串行总线（如 Feetech SCS / STServo、PX 系列）
    - 反馈：只能读回角度（位置），读不到速度/力矩
    - 典型型号：SCS0009, ST3215（可做 7-DOF 臂）

协议示例（Feetech SCS 串行总线，波特率 1Mbps，数据帧结构）：
    [0xFF 0xFF] [ID] [LEN] [CMD] [PARAM...] [CHECKSUM]
    写位置指令 CMD=0x03：PARAM = [角速度低, 角速度高, 位置低, 位置高]
    角度 → 编码值换算：encoder = angle_deg * (4096 / 360)

本 stub 用 pyserial 占位。真机接线后按你的舵机型号填协议即可。
"""
import numpy as np

from petlersArm.drivers.base import MotorDriver


class ServoDriver(MotorDriver):
    """单总线舵机驱动。"""

    def __init__(self, motor_id, port=None, baudrate=1_000_000):
        super().__init__(motor_id)
        self.port = port
        self.baudrate = baudrate
        self._ser = None
        self._connect()

    def _connect(self):
        """打开串口（stub：真机时用 serial.Serial 连接）。"""
        if self.port is None:
            return
        # import serial
        # self._ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        pass

    # 角度换算：舵机编码值 = 角度(rad) → deg → 编码
    @staticmethod
    def _deg_to_encoder(angle_rad):
        deg = np.degrees(angle_rad)
        return int(round(deg * 4096 / 360))

    def set_position(self, angle):
        """发送目标角度（弧度）。"""
        enc = self._deg_to_encoder(angle)
        # 构造 SCS 数据帧并写入串口（stub）
        # frame = bytes([0xFF, 0xFF, self.motor_id, 0x06, 0x03,
        #                *self._speed_bytes, *enc.to_bytes(2, 'little'),
        #                checksum])
        # self._ser.write(frame)
        pass

    def read_position(self):
        """读当前角度（弧度）。stub 返回 0。"""
        # 发送读位置指令，解析返回帧
        return 0.0

    def set_velocity(self, vel):
        """舵机通过"目标速度 + 位置"组合控制，速度作为附加参数。"""
        pass

    def set_torque(self, torque):
        """普通舵机无力矩模式。"""
        raise NotImplementedError("普通舵机不支持力矩控制")

    def enable(self):
        pass

    def disable(self):
        pass


class ServoArmMixin:
    """舵机机械臂：ArmInterface 的舵机实现骨架。

    用 7 个 ServoDriver 组合成一个 7-DOF 臂。
    算法层调用 goto(q) 时，这里把关节角分发给每个舵机。
    """

    def __init__(self, servo_ids, port):
        self.motors = [ServoDriver(i, port=port) for i in servo_ids]

    def goto(self, joint_angles, duration=None):
        for m, q in zip(self.motors, joint_angles):
            m.set_position(q)

    def get_joint_state(self):
        pos = [m.read_position() for m in self.motors]
        return np.array(pos), np.zeros(len(pos))

    def set_torque(self, torques):
        raise NotImplementedError("舵机不支持力矩控制")
