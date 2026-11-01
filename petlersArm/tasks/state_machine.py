"""
状态机模块：把 Dofbot main.py 的硬编码 5 状态提炼成通用状态表驱动。
每个状态描述四件事：
    - 如何计算目标（pos_fn）
    - 何时算到达/转移（trans_fn）
    - 转移后进入哪个状态（next_state）
    - 是否要执行夹爪动作（gripper_fn）

Dofbot 里的到达判断（np.isclose atol=1e-2）泛化为 Context.reached()。
"""
import numpy as np


class Context:
    """状态机运行上下文：持有当前关节角、目标、步数计时。"""

    def __init__(self, arm, kine, tol=1e-2, dt=1 / 240, interpolation=True,
                 step_size=0.02):
        """Context。

        Args:
            step_size: 插值每步最多转动的关节弧度（rad）。
                小值 = 机械臂整体动得越慢（每步走得更少）。默认 0.02（较慢）。
        """
        self.arm = arm
        self.kine = kine
        self.tol = tol
        self.dt = dt
        self.goal = None          # 当前目标关节角
        self._waypoints = None    # 插值轨迹（平滑移动，防止甩飞物体）
        self._timer = None        # 状态停留起始步数
        self.gripper = None       # 当前夹爪宽度
        self.step = 0             # 已执行的仿真步数
        self.interpolation = interpolation
        self.step_size = step_size

    # 读当前状态（优先用命令位置 —— resetJointState 设的当帧位置是轨迹的真相）
    def current_q(self):
        if hasattr(self.arm, "commanded_q"):
            return self.arm.commanded_q()
        return self.arm.get_joint_positions()

    # 目标跟踪：若开启插值，把目标角拆成逐步逼近的路径
    def set_goal(self, q_goal, pose_goal=None):
        q_goal = np.asarray(q_goal, dtype=float)
        self.goal = q_goal
        self.pose_goal = np.asarray(pose_goal, dtype=float) if pose_goal is not None else None
        self._waypoints = None
        if self.interpolation:
            start = self.current_q()
            # 关节空间线性插值，N 步逼近（每步最多转 step_size rad，控制移动速度）
            diff = q_goal - start
            n = max(1, int(np.max(np.abs(diff)) / self.step_size))
            self._waypoints = np.linspace(start, q_goal, n)

    def servo_goal(self, budget=None):
        """
        向关节伺服目标，并同步夹爪动作（对应 Dofbot 的 dofbot_control）。

        关键：waypoint 按【命令位置】推进（每次 goto 都前进 step_rad，
        累计到达 waypoint 后推进到下一个），而不是依赖物理引擎反馈的位置
        （resetJointState 设的位置 step 后会被物理修正，不能作为推进依据）。

        Args:
            budget: 本次调用最多推进的仿真步数。默认 20（一个状态较快完成）；
                设为 1 配合每步停顿可实现慢速观察完整规划过程。
        """
        if budget is None:
            budget = getattr(self, "step_budget", 20)
        # 记录本步累积的命令推进量（用于判断是否到达当前 waypoint）
        self._cmd_advance = getattr(self, "_cmd_advance", 0.0)
        while budget > 0:
            if self._waypoints is not None and len(self._waypoints) > 1:
                target = self._waypoints[0]
                # 按命令推进量判断：累计走了 step_size 就推进到下一个 waypoint
                self._cmd_advance += self.arm.step_rad
                if self._cmd_advance >= self.step_size:
                    self._cmd_advance = 0.0
                    self._waypoints = self._waypoints[1:]
                    if len(self._waypoints) > 1:
                        target = self._waypoints[0]
                    else:
                        target = self.goal
            else:
                target = self.goal
            self.arm.goto(target)
            if self.gripper is not None and hasattr(self.arm, "gripper_move"):
                self.arm.gripper_move(self.gripper)
            self.step += 1
            budget -= 1

    # 到达判据（Dofbot 同款：关节角接近目标）
    def reached(self, pos_tol=None):
        """到达判据。默认看关节角；pos_tol 给定时改用 FK 验证末端位置，
        更可靠（7-DOF 冗余臂关节角接近但末端可能没到目标）。
        若 set_goal 时给了 pose_goal，则直接比较当前末端 vs 目标位姿。"""
        if self.pose_goal is not None:
            ee_pos, _ = self.kine.forward(self.current_q())
            return np.linalg.norm(ee_pos - self.pose_goal[:3]) < (pos_tol or self.tol)
        if pos_tol is not None and self.kine is not None:
            ee_pos, _ = self.kine.forward(self.current_q())
            goal_pos, _ = self.kine.forward(self.goal)
            return np.linalg.norm(ee_pos - goal_pos) < pos_tol
        return np.sum(np.isclose(self.current_q(), self.goal, atol=self.tol)) >= self.arm.n_dof

    # 计时判据：按仿真步数计数（比墙钟 time.time 更稳定，不受计算速度影响）
    def timed(self, seconds):
        n_required = int(seconds / self.dt)
        if self._timer is None:
            self._timer = self.step
        if self.step - self._timer >= n_required:
            self._timer = None
            return True
        return False

    # 夹爪（gripper_fn 返回非数值（如 True）时忽略，防止误设宽度）
    def set_gripper(self, width):
        # 注意：bool 是 int 子类，必须排除（否则 set_gripper(True) 会设成 1.0）
        if isinstance(width, (int, float)) and not isinstance(width, bool):
            self.gripper = float(width)


class StateMachine:
    """通用状态表驱动状态机。

    用法：
        sm = StateMachine([
            State("MOVE_TO_CUP",  pos_fn,  ctx.reached, "GRASP"),
            State("GRASP",        pos_fn,  lambda c: c.timed(2.0), "LIFT"),
            ...
        ])
        sm.run(ctx)   # 循环直到 DONE
    """

    def __init__(self, states, done_state="DONE"):
        self.states = {s.name: s for s in states}
        self.done_state = done_state
        self.current = states[0].name
        self._entered = set()  # 已初始化目标的状态（避免每轮重建 waypoints 导致抽搐）

    def run(self, ctx, max_iters=None, verbose=True, slow=False, slow_delay=0.03):
        """运行状态机直到进入 DONE 状态。
        verbose=True 时打印每个阶段的规划过程（观察完整流程）。
        slow=True 时每仿真步之间停顿 slow_delay 秒，GUI 里清晰展示完整运动。"""
        import time
        i = 0
        while self.current != self.done_state:
            if max_iters and i >= max_iters:
                print(f"[StateMachine] 达到最大迭代 {max_iters}，提前结束")
                break
            s = self.states[self.current]
            # 1) 首次进入状态时计算目标位姿 → IK → 建 waypoints（只做一次！）
            if self.current not in self._entered:
                pose = s.pos_fn(ctx)
                q = ctx.kine.inverse(pose[0], pose[1], q_init=ctx.current_q())
                ctx.set_goal(q, pose_goal=pose[0])
                if verbose:
                    print(f"  ┌─ 状态 [{self.current}]：目标位姿 {np.round(pose[0], 3)} → IK 解算")
                # 2) 夹爪动作（进入时执行一次）
                if s.gripper_fn:
                    result = s.gripper_fn(ctx)
                    if isinstance(result, (int, float)) and not isinstance(result, bool):
                        ctx.set_gripper(result)
                    if verbose:
                        action = "闭合夹爪抓取" if result is not False else "张开夹爪释放"
                        print(f"  │   夹爪动作: {action}")
                self._entered.add(self.current)
            # 3) 伺服执行（持续逼近目标，不重建 waypoints）
            ctx.servo_goal()
            if slow:
                time.sleep(slow_delay)
            # 4) 转移判断
            if s.trans_fn(ctx):
                if verbose:
                    print(f"  └─ 状态 [{self.current}] 完成，进入 [{s.next_state}]")
                self.current = s.next_state
            i += 1


class State:
    """单个状态的描述。"""

    def __init__(self, name, pos_fn, trans_fn, next_state, gripper_fn=None):
        self.name = name
        self.pos_fn = pos_fn            # (ctx) -> (pos, orn)
        self.trans_fn = trans_fn        # (ctx) -> bool，是否转移到下一状态
        self.next_state = next_state
        self.gripper_fn = gripper_fn    # (ctx) -> gripper_width 或 None
