"""
运动学模块：基于 MuJoCo 的 FK / IK（解析 Jacobian + 阻尼最小二乘）。

MuJoCo 没有内置逆运动学，因此用经典数值方法：
  - FK：mj_forward + 读 body 的 xpos
  - Jacobian：mj_jacBody（解析，精确且快）
  - IK：阻尼最小二乘（DLS）迭代  q += J^T (J J^T + λI)^-1 Δx

关键能力：可以对【指尖中心】（两手指中点）做 IK ——
直接让夹爪真实抓取中心对准目标，无需闭环修正。

用法：
    kine = MuJoCoKinematics(model, data, joint_qposadr, ee_body='hand',
                            finger_bodies=['left_finger','right_finger'])
    pos, quat = kine.forward(q)                 # FK
    q = kine.inverse(pos)                       # 末端(hand) IK
    q = kine.inverse_fingertip(pos)             # 指尖中心 IK（抓取用）
"""
import numpy as np
import mujoco


class MuJoCoKinematics:
    def __init__(self, model, data, joint_qposadr, ee_body, finger_bodies=()):
        self.model = model
        self.data = data
        self.qposadr = list(joint_qposadr)
        self.n_dof = len(self.qposadr)
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self.finger_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                           for b in finger_bodies]
        # 关节限位
        self.lower_limits, self.upper_limits = [], []
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and name.startswith('joint') and name[5:].isdigit() and int(name[5:]) <= self.n_dof:
                self.lower_limits.append(model.jnt_range[j][0])
                self.upper_limits.append(model.jnt_range[j][1])

    # ---------------- 基础 ----------------
    def _set_q(self, q):
        """把臂关节角写入 qpos 并 forward。"""
        for k, adr in enumerate(self.qposadr):
            self.data.qpos[adr] = q[k]
        mujoco.mj_forward(self.model, self.data)

    def _current_q(self):
        """当前臂关节角（作为 IK 初值 —— 别从全零起步，容易跑到关节极限）。"""
        return np.array([self.data.qpos[a] for a in self.qposadr], dtype=float)

    def _get_pos(self, body_id):
        return self.data.xpos[body_id].copy()

    # ---------------- 正运动学 ----------------
    def forward(self, q):
        """FK：末端(hand)位置 + 四元数。纯计算，不改真实状态。"""
        q = np.asarray(q, dtype=float)
        saved = self.data.qpos.copy()
        self._set_q(q)
        pos = self.data.xpos[self.ee_body_id].copy()
        quat = self.data.xquat[self.ee_body_id].copy()
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        return pos, quat

    def gripper_center(self):
        """指尖中心（世界坐标）—— 夹爪真实抓取中心。"""
        if len(self.finger_ids) < 2:
            return None
        return ((self.data.xpos[self.finger_ids[0]]
                 + self.data.xpos[self.finger_ids[1]]) / 2.0).copy()

    # ---------------- 解析 Jacobian ----------------
    def _jac_for(self, body_ids, q):
        """对一组 body 的位置 Jacobian 取平均（3×n）。
        body_ids 给两个手指 → 指尖中心的 Jacobian。"""
        self._set_q(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        J_avg = np.zeros((3, self.n_dof))
        for bid in body_ids:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bid)
            J_avg += jacp[:, :self.n_dof]
        return J_avg / len(body_ids)

    def _ik_iter(self, target, body_ids, q_init, iters, tol, lam):
        """通用 DLS IK：让一组 body 的平均位置到达 target。

        纯计算：IK 迭代会把解写进 data.qpos 来算 FK/Jacobian，
        结束后必须还原仿真状态 —— 否则调用 IK 就等于把机械臂"瞬移"，
        后面再谈物理伺服就没有意义了。
        """
        saved = self.data.qpos.copy()
        q = np.asarray(q_init if q_init is not None else self._current_q(), dtype=float)
        try:
            for _ in range(iters):
                # 当前平均位置
                self._set_q(q)
                cur = np.mean([self.data.xpos[b] for b in body_ids], axis=0)
                err = np.asarray(target) - cur
                if np.linalg.norm(err) < tol:
                    break
                J = self._jac_for(body_ids, q)
                Jt = J.T
                delta = Jt @ np.linalg.solve(J @ Jt + lam * np.eye(3), err)
                q = q + delta
                for k in range(self.n_dof):
                    if k < len(self.lower_limits):
                        q[k] = np.clip(q[k], self.lower_limits[k], self.upper_limits[k])
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_forward(self.model, self.data)
        return q

    # ---------------- 逆运动学 ----------------
    def inverse(self, pos, quat=None, q_init=None, iters=200, tol=0.002, lam=0.01):
        """IK：末端(hand)到目标位置。"""
        return self._ik_iter(pos, [self.ee_body_id], q_init, iters, tol, lam)

    def inverse_fingertip(self, pos, q_init=None, iters=200, tol=0.002, lam=0.01):
        """IK：指尖中心（夹爪抓取中心）到目标位置。抓取对齐用这个。"""
        return self._ik_iter(pos, self.finger_ids, q_init, iters, tol, lam)

    def inverse_pose(self, pos, quat, q_init=None, iters=300, tol=0.001, lam=0.01):
        """
        6-DOF IK：同时约束末端(hand)位置 + 姿态。
        用于抓取 —— 约束 hand 朝向使两个手指对称包住物体（Panda 手指沿 X 闭合）。

        Args:
            pos: 目标位置 [x, y, z]
            quat: 目标姿态四元数（[w, x, y, z]）
            q_init: 初始关节角
        """
        quat = np.asarray(quat, dtype=float)
        quat /= np.linalg.norm(quat)  # 归一化
        saved = self.data.qpos.copy()
        q = np.asarray(q_init if q_init is not None else self._current_q(), dtype=float)
        try:
            for _ in range(iters):
                self._set_q(q)
                # 位置误差
                pos_err = np.asarray(pos) - self.data.xpos[self.ee_body_id]
                # 姿态误差（用旋转向量）
                cur_quat = self.data.xquat[self.ee_body_id]
                # 相对旋转 = quat * cur_quat^-1 → 转旋转向量
                q_rel = _quat_multiply(quat, _quat_conj(cur_quat))
                # 旋转向量 ≈ 2*[x,y,z]（小角度近似），用完整公式
                rot_vec = _quat_to_rotvec(q_rel)
                # 拼成 6-DOF 误差
                err6 = np.concatenate([pos_err, rot_vec])
                if np.linalg.norm(err6) < tol:
                    break
                # 6-DOF Jacobian: 位置 J (3xn) + 旋转 J (3xn)
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
                J = np.vstack([jacp[:3, :self.n_dof], jacr[:3, :self.n_dof]])
                Jt = J.T
                delta = Jt @ np.linalg.solve(J @ Jt + lam * np.eye(6), err6)
                q = q + delta
                for k in range(self.n_dof):
                    if k < len(self.lower_limits):
                        q[k] = np.clip(q[k], self.lower_limits[k], self.upper_limits[k])
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_forward(self.model, self.data)
        return q

    def is_reachable(self, pos, tol=0.05):
        q = self.inverse_fingertip(pos)
        saved = self.data.qpos.copy()
        self._set_q(q)
        try:
            return np.linalg.norm(self.gripper_center() - np.asarray(pos)) < tol
        finally:
            self.data.qpos[:] = saved
            mujoco.mj_forward(self.model, self.data)


def _quat_multiply(a, b):
    """四元数乘法（w,x,y,z）。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_rotvec(q):
    """四元数 → 旋转向量（用于姿态误差）。"""
    w, x, y, z = q
    if abs(w) > 0.999:
        return np.zeros(3)
    s = np.sqrt(1 - w*w)
    return 2.0 * np.arctan2(s, w) * np.array([x, y, z]) / s
