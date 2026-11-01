"""
驱动层：DM 电机（达妙无刷伺服电机）驱动。

DM 电机特点（对比舵机）：
    - 无刷电机 + 谐波减速器，扭矩大、精度高
    - 通信：CAN 总线（菊花链，多关节一条总线串联）
    - 支持位置/速度/力矩三种模式，反馈完整（角度+速度+电流）
    - 典型型号：DM-J4310, DM-J4340（常用于 7-DOF 协作臂）

CAN 协议（MIT 模式，经典 8 字节报文）：
    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
    │  p   │  p   │  v   │  v   │  Kp  │  Kp  │  Kd  │  Kd  │
    │ low  │ high │ low  │ high │ low  │ high │ low  │ high │
    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
    位域：p(12bit) v(12bit) Kp(12bit) Kd(12bit) t(12bit) —— 共 64bit
    归一化换算：
        p_int = (pos_rad / π)  * 32767            （±π 对应 ±32767）
        v_int = (vel_rad/s / 30) * 32767          （±30rad/s 对应 ±32767）
        Kp_int = (Kp / 500)  * 32767
        Kd_int = (Kd / 5)    * 32767
        t_int = (torque / 18) * 32767             （±18Nm 对应 ±32767）

本 stub 用 python-can 占位。真机时按你的 CAN 板卡型号配置接口即可。
"""
import struct
import numpy as np

from petlersArm.drivers.base import MotorDriver


# MIT 模式数据打包/解包
_P_MAX = 32767.0


def pack_mit_cmd(p, v, kp, kd, t):
    """
    将物理量打包成 MIT 模式 8 字节 CAN 数据。
    Args:
        p, v, kp, kd, t: 归一化整数（-32767 ~ 32767）
    """
    p = int(np.clip(p, -_P_MAX, _P_MAX))
    v = int(np.clip(v, -_P_MAX, _P_MAX))
    kp = int(np.clip(kp, -_P_MAX, _P_MAX))
    kd = int(np.clip(kd, -_P_MAX, _P_MAX))
    t = int(np.clip(t, -_P_MAX, _P_MAX))
    return struct.pack('<HHHH', (p << 4) | (v >> 8), (v & 0xFF) << 8 | (kp >> 4),
                       (kp & 0x0F) << 12 | (kd >> 0) & 0xFFF, 0)  # 简化示例


class DmMotorDriver(MotorDriver):
    """达妙无刷电机驱动（CAN 总线）。"""

    def __init__(self, motor_id, can_bus=None, p_min=-np.pi, p_max=np.pi,
                 v_max=30.0, kp_max=500.0, kd_max=5.0, t_max=18.0):
        super().__init__(motor_id)
        self.can_bus = can_bus
        # 量程（归一化范围）
        self.p_min, self.p_max = p_min, p_max
        self.v_max, self.kp_max, self.kd_max, self.t_max = v_max, kp_max, kd_max, t_max
        self._pos = 0.0  # 本地缓存当前位置

    # ---------------- 通信 ----------------
    def _send(self, data_bytes, can_id=None):
        """发送一条 CAN 报文（stub：真机用 can.Bus.send）。"""
        if self.can_bus is not None:
            # import can
            # msg = can.Message(arbitration_id=can_id or self.motor_id, data=data_bytes)
            # self.can_bus.send(msg)
            pass

    def _recv(self):
        """接收 CAN 报文并解析状态（stub）。"""
        # import can
        # msg = self.can_bus.recv(timeout=0.1)
        # return self._unpack_status(msg.data)
        return self._pos, 0.0, 0.0

    # ---------------- 归一化换算 ----------------
    def _norm(self, x, lo, hi):
        return 2.0 * (x - lo) / (hi - lo) - 1.0  # 映射到 [-1, 1]

    def _denorm(self, x, lo, hi):
        return (x + 1.0) * (hi - lo) / 2.0 + lo

    # ---------------- MotorDriver 实现 ----------------
    def set_position(self, angle):
        """位置模式：发送目标角度 + PD 增益（Kp/Kd 决定关节刚度）。"""
        # MIT 模式位置控制：p 目标位置，v=0，Kp/Kd 设定
        p_int = self._norm(angle, self.p_min, self.p_max) * _P_MAX
        kp_int = 1.0 * _P_MAX   # Kp = 1.0（低刚度，防止抖动）
        kd_int = 0.0
        self._send(pack_mit_cmd(p_int, 0, kp_int, kd_int, 0))

    def set_torque(self, torque):
        """力矩模式：直接控制输出力矩（舵机做不到这个）。"""
        t_int = self._norm(torque, -self.t_max, self.t_max) * _P_MAX
        self._send(pack_mit_cmd(0, 0, 0, 0, t_int))

    def set_velocity(self, vel):
        v_int = self._norm(vel, -self.v_max, self.v_max) * _P_MAX
        self._send(pack_mit_cmd(0, v_int, 0, 0, 0))

    def read_position(self):
        pos, _, _ = self._recv()
        return pos

    def read_full_state(self):
        """读完整状态：位置/速度/电流（DM 电机的完整反馈）。"""
        return self._recv()

    def enable(self):
        # 发送使能指令（如 0xFF 使能帧）
        pass

    def disable(self):
        pass


class DmArmMixin:
    """DM 电机机械臂：ArmInterface 的 DM 实现骨架。

    用 7 个 DmMotorDriver 组合成 7-DOF 臂，CAN 菊花链。
    算法层调用 goto(q) / set_torque(t) 时分发到各电机。
    """

    def __init__(self, motor_ids, can_bus=None):
        self.motors = [DmMotorDriver(i, can_bus=can_bus) for i in motor_ids]

    def goto(self, joint_angles, duration=None):
        for m, q in zip(self.motors, joint_angles):
            m.set_position(q)

    def get_joint_state(self):
        pos = [m.read_position() for m in self.motors]
        vel = [0.0] * len(self.motors)
        return np.array(pos), np.array(vel)

    def set_torque(self, torques):
        for m, t in zip(self.motors, torques):
            m.set_torque(t)
