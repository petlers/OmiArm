"""
运动学模块：基于 PyBullet 的通用 FK / IK。
与 Dofbot 项目的 dofbot.setInverseKine 同一思路，但泛化为 N 关节，
并支持 7-DOF 机械臂的冗余度优化（null-space 自运动）。

用法：
    kine = UrdfKinematics(urdf_path, end_effector_link=6, joint_indices=[0..6])
    T = kine.forward(q)                        # FK: 关节角 → 末端位姿
    q = kine.inverse(pos, orn=None, q_init=None)  # IK: 位姿 → 关节角
"""
import numpy as np
import pybullet as p


class UrdfKinematics:
    def __init__(self, body_uid, joint_indices, end_effector_link):
        """
        Args:
            body_uid: PyBullet 加载后的 body id
            joint_indices: 参与运动学计算的关节索引列表（不含夹爪）
            end_effector_link: 末端连杆索引（IK 的目标连杆）
        """
        self.uid = body_uid
        self.joint_indices = list(joint_indices)
        self.n_dof = len(self.joint_indices)
        self.ee_link = end_effector_link

        # 读关节限位，构造 IK 需要的 null-space 参数
        self.lower_limits, self.upper_limits = [], []
        self.joint_ranges = []
        self.rest_poses = []
        for j in self.joint_indices:
            info = p.getJointInfo(self.uid, j)
            lo, hi = info[8], info[9]
            self.lower_limits.append(lo)
            self.upper_limits.append(hi)
            self.joint_ranges.append(hi - lo)
            # 默认 resting pose 取关节范围中点
            self.rest_poses.append(0.5 * (lo + hi))

    # ---------------- 正运动学 ----------------
    def forward(self, q):
        """
        FK：给定关节角，求末端位姿。
        关键：计算前保存当前关节状态，算完恢复 —— FK 是【纯计算】，
        绝不能污染机械臂的真实关节状态（否则控制循环全部紊乱）。
        """
        q = np.asarray(q, dtype=float)
        assert len(q) == self.n_dof, f"期望 {self.n_dof} 个关节角，收到 {len(q)}"
        saved = [p.getJointState(self.uid, j)[0] for j in self.joint_indices]
        for i, j in enumerate(self.joint_indices):
            p.resetJointState(self.uid, j, q[i])
        pos, orn = p.getLinkState(self.uid, self.ee_link)[:2]
        for j, v in zip(self.joint_indices, saved):
            p.resetJointState(self.uid, j, v)
        return np.array(pos, dtype=float), np.array(orn, dtype=float)

    # ---------------- 逆运动学 ----------------
    def _ik_once(self, pos, orn, rest_poses):
        """单次 IK 求解（内部），返回 (关节角, 末端位置误差)。"""
        if orn is None:
            q = p.calculateInverseKinematics(
                self.uid, self.ee_link, list(pos),
                self.lower_limits, self.upper_limits,
                self.joint_ranges, rest_poses,
            )
        else:
            q = p.calculateInverseKinematics(
                self.uid, self.ee_link, list(pos), list(orn),
                self.lower_limits, self.upper_limits,
                self.joint_ranges, rest_poses,
            )
        q = np.asarray(q, dtype=float)[: self.n_dof]
        # 用 FK 验证误差（forward 现在不污染关节状态）
        fk_pos, _ = self.forward(q)
        err = np.linalg.norm(fk_pos - np.asarray(pos, dtype=float))
        return q, err

    def _within_limits(self, q, tol=1e-4):
        """检查关节角是否在限位内（工业 IK 的必要约束）。"""
        q = np.asarray(q)
        for i in range(self.n_dof):
            if q[i] < self.lower_limits[i] - tol or q[i] > self.upper_limits[i] + tol:
                return False
        return True

    def inverse(self, pos, orn=None, q_init=None, multi_start=True):
        """
        IK：给定末端目标位姿，求关节角。

        PyBullet 的 IK 是数值法，严重依赖初始猜测（可能陷入局部最优）。
        因此默认做【多重启动】：从多个初始猜测各解一次，
        **只接受关节角在限位内的解**，再选末端误差最小的。

        Args:
            pos: 目标位置 [x, y, z]
            orn: 目标姿态四元数（None 表示只约束位置）
            q_init: 首选初始猜测（通常是上一次的关节角，连续运动时收敛最好）
            multi_start: 是否多重启动（True=更稳，False=只用 q_init）

        Returns:
            ndarray 形状 (n_dof,)，IK 求得的关节角（限位内）
        """
        starts = []
        if q_init is not None:
            starts.append(list(q_init))              # 1. 上次解（连续运动首选）
        starts.append(self.rest_poses)               # 2. 零位/中位
        if multi_start:
            # 3. 随机散布 1 个初始猜测（覆盖局部最优；多次很慢）
            rng = np.random.default_rng(0)
            noise = rng.uniform(-1.5, 1.5, self.n_dof)
            starts.append(list(np.asarray(self.rest_poses) + noise))

        best_q, best_err = None, np.inf
        for rest in starts:
            q, err = self._ik_once(pos, orn, rest)
            # 关键：跳过关节角超限位的解（否则位置控制到不了，末端位置完全错）
            if not self._within_limits(q):
                continue
            if err < best_err:
                best_q, best_err = q, err
        if best_q is None:
            # 全都不在限位内 → 返回误差最小的那个（调用方自行判断）
            best_q, best_err = None, np.inf
            for rest in starts:
                q, err = self._ik_once(pos, orn, rest)
                if err < best_err:
                    best_q, best_err = q, err
        return best_q

    # ---------------- 冗余度工具（7-DOF 特性） ----------------
    def solve_with_restpose(self, pos, orn, rest_pose):
        """
        带指定 resting pose 的 IK。
        7-DOF 机械臂对同一末端位姿有无穷多解（肘部可以绕腕-肩轴线转）。
        指定 rest_pose 可以挑选不同的肘部姿态 —— 例如避障、改变抓取姿态。
        """
        return self.inverse(pos, orn, q_init=rest_pose)

    def is_reachable(self, pos, tol=0.05):
        """判断目标位置是否在机械臂工作空间内（正运动学试解）"""
        q = self.inverse(pos)
        fk_pos, _ = self.forward(q)
        return np.linalg.norm(fk_pos - np.asarray(pos)) < tol
