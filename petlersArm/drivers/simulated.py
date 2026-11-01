"""
驱动层：仿真驱动。
把 Dofbot 项目的 dofbot.py 封装提炼成 ArmInterface 的实现。
用 PyBullet 的 setJointMotorControl2(POSITION_CONTROL) 做关节位置伺服。
"""
import numpy as np
import pybullet as p

from petlersArm.core.arm_interface import ArmInterface


class SimulatedArm(ArmInterface):
    """PyBullet 仿真机械臂驱动。"""

    # ---------------- ArmInterface 实现 ----------------
    @property
    def n_dof(self):
        return self.n_dof_

    def goto(self, joint_angles, duration=None):
        """
        位置伺服：用 resetJointState 逐步逼近目标关节角（每步最多转 self.step_rad）。

        为什么不用 setJointMotorControl2(POSITION_CONTROL)？
        实测 PyBullet 位置控制在多关节联动时存在 bug：个别关节（如 joint0）
        会完全卡死不动（速度 0）。而 resetJointState 逐步逼近：
          - 每个关节都能到位（误差 ~0.000）
          - 步长可控 → 完全平滑、不抽搐（无 PID 振荡）
          - 速度由 step_rad 控制（等效 maxVelocity）

        注意：resetJointState 设的位置会【保持】（物理引擎不会改写它），
        所以每步从物理位置向目标推进 step_rad 即可 —— 物理位置即命令位置，
        轨迹平滑单调，无震荡、无脱节。
        """
        q = np.asarray(joint_angles, dtype=float)
        cur = self.get_joint_positions()
        # 每步最多移动 step_rad，实现慢速平滑运动（从物理位置单调推进）
        delta = np.clip(q - cur, -self.step_rad, self.step_rad)
        target = cur + delta
        self._cmd_q = target.copy()   # 记录命令位置（用于到达判断）
        for i, j in enumerate(self.joint_indices):
            p.resetJointState(self.uid, j, float(target[i]))
        # 先 hold（把物体放到夹爪位置），再 step —— 避免"物理先下坠、再拽回"的抖动
        self.hold()
        p.stepSimulation()

    def commanded_q(self):
        """返回最近一次命令的关节位置（resetJointState 设的当帧位置）。
        用于到达判断 —— 物理引擎反馈的位置会被动力学修正，命令位置才是轨迹的真相。"""
        cmd = getattr(self, "_cmd_q", None)
        return cmd if cmd is not None else self.get_joint_positions()

    def gripper_center(self):
        """返回两个指尖的中点位置（世界坐标）—— 夹爪的实际抓取对称中心。
        Panda 的 link11 不是指尖对称中心，抓取对齐必须用这个真实中心。"""
        if len(self.gripper_indices) < 2:
            return None
        lf = p.getLinkState(self.uid, self.gripper_indices[0])[0]
        rf = p.getLinkState(self.uid, self.gripper_indices[1])[0]
        return (np.array(lf) + np.array(rf)) / 2.0

    def get_joint_state(self):
        pos, vel = [], []
        for j in self.joint_indices:
            s = p.getJointState(self.uid, j)
            pos.append(s[0])
            vel.append(s[1])
        return np.array(pos), np.array(vel)

    def set_torque(self, torques):
        """力矩控制（DM 电机路径的仿真版）。"""
        for i, j in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyUniqueId=self.uid, jointIndex=j,
                controlMode=p.TORQUE_CONTROL, force=float(torques[i]),
            )
        p.stepSimulation()

    def reset(self, q_home=None):
        if q_home is None:
            q_home = [0.0] * self.n_dof_
        for i, j in enumerate(self.joint_indices):
            p.resetJointState(self.uid, j, q_home[i])
        self._cmd = None
        self._grip_cmd = None
        self._cmd_q = None  # 重置命令位置（避免残留导致瞬移）
        p.stepSimulation()

    # ---------------- 夹爪控制 ----------------
    def __init__(self, body_uid, joint_indices, gripper_indices=None,
                 dt=1 / 240, grip_step=2.0, step_rad=0.02):
        """SimulatedArm。

        Args:
            grip_step: 夹爪闭合时每次最多收拢的宽度（mm）。
                小值 = 夹得慢（渐进式夹紧），默认 2mm/步。
            step_rad: goto 时每个关节每步最多转动的弧度（rad）。
                控制整个机械臂的运动速度：越小越慢、越平滑（不抽搐）。
                默认 0.02 rad/步 ≈ 1.15°/步。
        """
        self.uid = body_uid
        self.joint_indices = list(joint_indices)
        self.n_dof_ = len(self.joint_indices)
        self.gripper_indices = list(gripper_indices or [])
        self.dt = dt
        self.grip_step = grip_step
        self.step_rad = step_rad
        self._grip_target = None  # 当前夹爪目标宽度
        self._grip_cmd = None     # 最近一次下发的夹爪目标
        self._cmd = None          # 最近一次下发的关节目标

    # ---------------- 夹爪控制 ----------------
    # Panda 夹爪：joint9 和 joint10 是两个 prismatic 关节（官方同时设正值）。
    # 两个指尖沿 Y 方向对称开合：
    #   joint=0.0  → 指尖间距 14mm（最小，接近合拢）
    #   joint=0.04 → 指尖间距 70mm（最大张开）
    # 线性映射：joint = (width_mm - 14) / 56 * 0.04
    GRIP_JOINT_MAX = 0.04
    GRIP_OPEN_MIN_MM = 14.0   # 指尖间距下限 (mm)
    GRIP_OPEN_MAX_MM = 70.0   # 指尖间距上限 (mm)

    def _mm_to_joint(self, width_mm):
        """指尖间距(mm) → finger 关节位置。"""
        w = np.clip(width_mm, self.GRIP_OPEN_MIN_MM, self.GRIP_OPEN_MAX_MM)
        return (w - self.GRIP_OPEN_MIN_MM) / (self.GRIP_OPEN_MAX_MM - self.GRIP_OPEN_MIN_MM) * self.GRIP_JOINT_MAX

    def _joint_to_mm(self, joint_pos):
        """finger 关节位置 → 指尖间距(mm)。"""
        return self.GRIP_OPEN_MIN_MM + joint_pos / self.GRIP_JOINT_MAX * (self.GRIP_OPEN_MAX_MM - self.GRIP_OPEN_MIN_MM)

    # ---------------- 抓取（PyBullet 标准 pick-and-place 方式） ----------------
    # 说明：Panda 两指夹爪 + PyBullet 接触求解器，物理上很难稳定夹住并抬起小物体
    # （指尖闭合时带 X 分量滑动）。学术/工业仿真里标准做法是"抓稳"后让物体跟随末端：
    #   pin_object() → 抓取前禁用机械臂-物体碰撞 + 固定物体到台面
    #                  （机械臂怎么动都不会碰/推动物体，零物理干涉）
    #   grasp()      → 解除固定，记录夹爪与物体的相对偏移，之后物体跟随夹爪
    #   hold()       → 每步调用，把物体钉在夹爪上（跟随移动）
    #   release()    → 停止跟随 + 重新启用碰撞（物体自由落下）
    # 这样抓取/放置任务可靠、可复现，算法框架（状态机/IK/插值）不受影响。

    def pin_object(self, object_uid, fixed_pos, mass=0.0):
        """把物体钉在台面 + 禁用机械臂-物体碰撞。
        抓取前调用 —— 机械臂如何接近都不会碰/推动物体。
        grasp() 时恢复质量；release() 时恢复碰撞。"""
        self._pinned_obj = object_uid
        self._pin_mass = mass
        p.changeDynamics(object_uid, -1, mass=mass)
        # 禁用机械臂（所有 link）与物体之间的碰撞 → 零物理干涉
        for j in range(p.getNumJoints(self.uid) + 1):
            p.setCollisionFilterPair(self.uid, object_uid, j, -1, enableCollision=0)
        p.resetBasePositionAndOrientation(object_uid, fixed_pos, [0, 0, 0, 1])
        return True

    def unpin_object(self):
        """解除钉住（grasp 前调用）：恢复物体原质量。"""
        if getattr(self, "_pinned_obj", None) is not None:
            p.changeDynamics(self._pinned_obj, -1, mass=0.15)
            self._pinned_obj = None
            self._pin_mass = None

    def enable_object_collision(self):
        """重新启用机械臂-物体碰撞（release 时调用，让物体自由落下）。"""
        if getattr(self, "_pinned_obj", None) is not None:
            for j in range(p.getNumJoints(self.uid) + 1):
                p.setCollisionFilterPair(self.uid, self._pinned_obj, j, -1, enableCollision=1)

    def grasp(self, object_uid):
        """抓取：以【指尖中心】为锚记录物体偏移，之后物体跟随夹爪。
        用指尖中心（而非 link11）作锚 —— 夹爪包住杯身时，杯子精确在指尖中心，
        跟随移动时物体不偏移。保持碰撞禁用（抓取期间物体不碰机械臂）。"""
        self.unpin_object()
        gc = self.gripper_center()
        obj_pos, obj_orn = p.getBasePositionAndOrientation(object_uid)
        self._grasp_obj = object_uid
        self._grasp_offset = np.array(obj_pos) - np.array(gc)
        return True

    def hold(self):
        """保持抓取：每步把物体位置设为指尖中心 + 抓取偏移（跟随移动）。"""
        if getattr(self, "_grasp_obj", None) is None:
            return
        gc = self.gripper_center()
        obj_pos = np.array(gc) + self._grasp_offset
        p.resetBasePositionAndOrientation(self._grasp_obj, obj_pos, [0, 0, 0, 1])

    def release(self):
        """释放：停止跟随 + 重新启用碰撞，物体自由落下（和地面/垃圾桶正常碰撞）。"""
        obj = getattr(self, "_grasp_obj", None)
        self._grasp_obj = None
        self._grasp_offset = None
        if obj is not None:
            # 重新启用机械臂-物体碰撞
            for j in range(p.getNumJoints(self.uid) + 1):
                p.setCollisionFilterPair(self.uid, obj, j, -1, enableCollision=1)
            # 物体仍需与地面/垃圾桶碰撞（它们没被禁用）

    def gripper_move(self, width_mm, instant_open=True):
        """
        控制夹爪开合宽度（指尖间距，单位 mm）。
        同时控制 joint9 和 joint10（都是正值），位置控制产生夹持力。
        闭合渐进式慢夹，张开快速。

        Args:
            width_mm: 目标指尖间距（毫米）。Panda 有效范围约 14~70mm。
                闭合到 ~14mm = 指尖基本合拢。
            instant_open: True=张开立即到位；闭合总是渐进式。
        """
        if len(self.gripper_indices) < 2:
            return
        self._grip_target = float(width_mm)
        target_joint = self._mm_to_joint(width_mm)
        # 当前指尖间距（mm），用 joint9 近似
        current_joint = p.getJointState(self.uid, self.gripper_indices[0])[0]
        current_mm = self._joint_to_mm(current_joint)

        # 张开方向：立即到位；闭合方向：渐进式（每次收拢 grip_step mm）
        if instant_open and width_mm > current_mm:
            target_now = target_joint
        else:
            step_mm = np.clip(width_mm - current_mm, -self.grip_step, 0.0)
            target_now = self._mm_to_joint(current_mm + step_mm)

        # 目标没变就不重设控制器（避免夹爪抖动）
        if self._grip_cmd is None or abs(self._grip_cmd - target_now) > 1e-5:
            self._grip_cmd = target_now
            for g in self.gripper_indices[:2]:
                p.setJointMotorControl2(
                    bodyUniqueId=self.uid, jointIndex=g,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=float(target_now),
                    force=200.0, maxVelocity=0.3,
                    positionGain=1.0, velocityGain=1.0,
                )
        p.stepSimulation()
