"""
驱动层：MuJoCo 真实物理仿真驱动。

真实物理特性（与 PyBullet 方案的本质区别）：
  - 臂关节用 MuJoCo 的 actuator 位置伺服（data.ctrl[:7] = 目标关节角），
    由真实 PD 动力学驱动，而非 resetJointState 直接设位置
  - 夹爪用 tendon 力控制（data.ctrl[7]，0=张开，255=夹紧），真实接触夹持
  - 抓取靠真实接触 + 摩擦，不使用 hold/pin/禁用碰撞等技巧
"""
import numpy as np
import mujoco

from petlersArm.core.arm_interface import ArmInterface


class MuJoCoArm(ArmInterface):
    """MuJoCo Panda 机械臂驱动（真实物理）。"""

    def __init__(self, 
                 model, 
                 data, 
                 joint_qposadr, 
                 gripper_ctrl_idx,
                 finger_bodies=('left_finger', 'right_finger'),
                 grip_step=2.0, 
                 step_rad=0.05, 
                 dt=0.002, 
                 grip_stiffness=500.0):
        """
        Args:
            model: mujoco.MjModel
            data: mujoco.MjData
            joint_qposadr: 臂关节在 qpos 中的地址（[0..6]）
            gripper_ctrl_idx: 夹爪 actuator 的 ctrl 索引（Panda = 7）
            finger_bodies: 指尖 body 名
            grip_step: 夹爪每次收拢的 ctrl 增量（0-255）
            step_rad: goto 每次伺服的最大关节角变化
            dt: 仿真步长
            grip_stiffness: 夹爪 tendon 执行器刚度（N/m）。Panda 模型默认 100 N/m，
                对 44mm 杯子只能压出 ~1N 法向力，摩擦握力不够、会滑脱；
                500 N/m ≈ 10N 握力，接近真机夹爪的夹持能力。
                ctrl → 开口的映射（0..255 ↔ 0..80mm）保持不变。
        """
        self.model = model
        self.data = data
        self.qposadr = list(joint_qposadr)
        self.n_dof_ = len(self.qposadr)
        self.grip_ctrl = gripper_ctrl_idx
        self.finger_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                                for b in finger_bodies]
        self.grip_step = grip_step
        self.step_rad = step_rad
        self.dt = dt
        self._grip_target = None
        self._cmd_q = None
        self.set_grip_stiffness(grip_stiffness)

    # ---------------- ArmInterface 实现 ----------------
    @property
    def n_dof(self):
        return self.n_dof_

    def set_grip_stiffness(self, stiffness):
        """调整夹爪 tendon 执行器刚度，保持 ctrl → 开口映射不变。

        原模型：force = gain*ctrl - 100*length，即 kp=100 N/m。
        改后：  force = (gain/100*K)*ctrl - K*length —— 同样的 ctrl 对应同样的开口，
                但接触时的闭合力放大 K/100 倍。
        """
        a = self.grip_ctrl
        kp0 = -self.model.actuator_biasprm[a][1]
        if kp0 <= 0:
            return
        gain0 = self.model.actuator_gainprm[a][0]
        self.model.actuator_gainprm[a][0] = gain0 / kp0 * stiffness
        self.model.actuator_biasprm[a][1] = -stiffness

    def goto(self, joint_angles, duration=None, hold_steps=400, tol=0.02,
             step_cb=None):
        """
        位置伺服：把完整目标关节角发给 actuator（真实 PD 驱动），
        持续 step 直到关节接近目标（actuator 有动力学延迟，需多步收敛）。

        轨迹插值（step_rad waypoint）由调用方（任务层）负责，
        goto 本身不做 step_rad 限制 —— 否则 actuator 追不上，机械臂几乎不动。

        Args:
            tol: 到达判定（关节角误差，rad）
            step_cb: 每个仿真步后的回调（如 viewer.sync），用于可视化
        """
        q = np.asarray(joint_angles, dtype=float)
        self._cmd_q = q.copy()
        # 发送完整目标到 actuator（PD 位置伺服）
        for k, adr in enumerate(self.qposadr):
            self.data.ctrl[adr] = q[k]
        # 持续 step 直到关节接近目标（或达到步数上限）
        for _ in range(hold_steps):
            cur = self.get_joint_positions()
            if np.max(np.abs(cur - q)) < tol:
                break
            mujoco.mj_step(self.model, self.data)
            if step_cb is not None:
                step_cb()

    def commanded_q(self):
        """最近命令的关节位置（用于到达判断）。"""
        return self._cmd_q if self._cmd_q is not None else self.get_joint_positions()
    
    def get_joint_state(self):
        """读臂关节位置 + 速度。"""
        pos = np.array([self.data.qpos[a] for a in self.qposadr], dtype=float)
        # qvel 地址：7 个旋转关节每个 1 维速度
        vel = np.array([self.data.qvel[a] for a in self.qposadr], dtype=float)
        return pos, vel

    def set_torque(self, torques):
        """力矩控制（真实 actuator 力）。"""
        for k, adr in enumerate(self.qposadr):
            # 用 qfrc_applied 施加外力矩（绕过 actuator 的位置伺服）
            self.data.qfrc_applied[adr] = torques[k]
        mujoco.mj_step(self.model, self.data)

    def reset(self, q_home=None):
        """复位到 home 位：同时设 qpos 和 ctrl（位置伺服目标 = home 角）。
        关键：只设 qpos 而 ctrl 保持 0，位置伺服的 ctrl=0 会把机械臂拉到零位（塌陷）。"""
        if q_home is None:
            q_home = np.deg2rad([0, 0, 0, -90, 0, 90, -45])
        for k, adr in enumerate(self.qposadr):
            self.data.qpos[adr] = q_home[k]
            self.data.ctrl[adr] = q_home[k]   # 位置伺服目标 = home 角（保持不塌）
        # 张开夹爪（ctrl=255 = 张开）
        self.data.ctrl[self.grip_ctrl] = 255.0
        self._cmd_q = None
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)



    # ---------------- 夹爪 ----------------
    def gripper_center(self):
        """两个指尖中点（世界坐标）—— 夹爪真实抓取中心。"""
        if len(self.finger_body_ids) < 2:
            return None
        c = (self.data.xpos[self.finger_body_ids[0]] + self.data.xpos[self.finger_body_ids[1]]) / 2.0
        return c.copy()

    def gripper_open(self, hold_steps=60):
        """张开夹爪（ctrl → 255 = 最大张开，实测指尖间距 78mm）。"""
        self.data.ctrl[self.grip_ctrl] = 255.0
        for _ in range(hold_steps):
            mujoco.mj_step(self.model, self.data)

    def gripper_close(self, hold_steps=300):
        """夹紧夹爪（ctrl → 0 = 完全闭合，实测指尖间距 0mm）。
        真实接触：手指闭合压住物体，靠摩擦夹持。"""
        self.data.ctrl[self.grip_ctrl] = 0.0
        for _ in range(hold_steps):
            mujoco.mj_step(self.model, self.data)
