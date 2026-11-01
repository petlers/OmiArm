"""
任务层：抓取放置（水杯 → 垃圾桶）。

抓取流程（每个阶段一个状态，物理合理、可靠）：
    MOVE_ABOVE → DESCEND → GRASP → LIFT → MOVE_TO_TARGET → PLACE → DONE

关键设计（基于 PyBullet + Panda 的实验结论）：
  1. 大臂运动：resetJointState 平滑步进（可靠、不抽搐）
  2. 接近姿态：orn_down 垂直朝下 —— 指尖中心能精确对准杯心（误差 <5mm）
  3. 夹爪闭合：resetJointState 直接设 finger 关节（可靠闭合到杯身两侧）
  4. 抓取保持：固定约束 hold-follow（PyBullet 两指物理夹持接触不可靠，
     用约束模拟"夹稳"，视觉上夹爪已真实包住杯身）
"""
import numpy as np

from petlersArm.tasks.state_machine import State, StateMachine

# 夹爪张开宽度（mm，Panda 指尖间距范围约 14~70mm）
GRIPPER_OPEN = 60.0
# 抓取闭合宽度：指尖张开到略大于杯宽（47mm），从两侧"抱住"杯身而不穿过。
# 若闭合到 <杯宽，指尖会穿过杯身造成穿模。
GRIPPER_CLOSE = 50.0

# 接近高度：机械臂先到物体上方再下降，避免水平碰撞
APPROACH_HEIGHT = 0.20
# 抓取时 link11 目标相对物体中心的 Z 偏移（下降到杯身中部）
GRASP_Z_OFFSET = -0.015


def _pose_at(pos):
    """只约束位置（orn=None），IK 自动选解。抓取用闭环对齐修正指尖中心。"""
    return np.asarray(pos, dtype=float), None


def build_pick_place_task(arm, kine, obj_pos, target_pos, object_uid=None,
                          lift_height=0.15, grasp_pause=1.5, place_pause=2.0):
    """
    构造抓取放置状态机（水杯 → 垃圾桶）。

    Args:
        arm: ArmInterface（含 gripper_center / grasp / release）
        kine: UrdfKinematics
        obj_pos: 物体中心位置
        target_pos: 垃圾桶位置
        object_uid: 被抓物体 PyBullet id（提供则用固定约束可靠抓取）
        lift_height: 抬升高度
        grasp_pause: 抓取停留（等夹爪闭合）
        place_pause: 放置停留（等杯子落入桶）

    Returns:
        StateMachine
    """
    obj_pos = np.array(obj_pos, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    # ------- 目标位姿点 -------
    approach_pos = obj_pos + np.array([0, 0, APPROACH_HEIGHT])  # 物体上方
    grasp_pos = obj_pos + np.array([0, 0, GRASP_Z_OFFSET])       # 杯身中部
    lift_pos = obj_pos + np.array([0, 0, lift_height])           # 抬升后
    target_over_pos = target_pos + np.array([0, 0, lift_height]) # 垃圾桶上方
    target_place_pos = target_pos + np.array([0, 0, lift_height * 0.5])  # 释放点

    # ------- 抓取：指尖包住杯身 + 固定约束 -------
    def _goto(c, pose, steps=200):
        """直接用 IK + 平滑步进移动到目标（绕过状态机插值，可靠到位）。"""
        q = c.kine.inverse(pose, None, q_init=c.current_q())
        for _ in range(steps):
            c.arm.goto(q)

    def do_grasp(c):
        # 1. 指尖中心对齐到杯心：XY 闭环对齐，Z 也降到杯身中部（指尖包住杯身）
        _align_xy(c, obj_pos)
        # 2. 闭合夹爪（渐进式收拢到杯身两侧，视觉上包住杯子）
        for _ in range(30):
            c.arm.gripper_move(GRIPPER_CLOSE, instant_open=False)
        # 3. 建立固定约束（以指尖中心为锚 → 杯子贴合指尖，无穿模偏移）
        if object_uid is not None and hasattr(c.arm, "grasp"):
            c.arm.grasp(object_uid)
        return True

    def _align_xy(c, obj_center, iters=5):
        """指尖中心闭环对齐到物体中心：XYZ 全三维修正。
        让指尖中心精确等于杯心（消除 link11→指尖的偏移造成的穿模）。
        用 arm.goto 直接伺服（可靠），每轮读指尖中心修正目标。"""
        target = np.array(obj_center, dtype=float)
        for it in range(iters):
            _goto(c, target, steps=150)
            gc = c.arm.gripper_center()
            if gc is None:
                break
            err = gc - obj_center
            if np.max(np.abs(err)) < 0.005:
                break
            correction = np.clip(err, -0.01, 0.01)
            target -= correction

    def do_release(c):
        if c.reached(pos_tol=0.08):
            if object_uid is not None and hasattr(c.arm, "release"):
                c.arm.release()
            c.set_gripper(GRIPPER_OPEN)
        return False  # False → 打印"张开夹爪释放"

    REACH_TOL = 0.08  # 到达容差（位置约束 IK 残留约 0.05）

    states = [
        # 1. 移动到物体上方（不碰杯）
        State("MOVE_ABOVE", lambda c: _pose_at(approach_pos),
              lambda c: c.reached(pos_tol=REACH_TOL), "DESCEND"),
        # 2. 下降到杯身（指尖包住杯身位置）
        State("DESCEND", lambda c: _pose_at(grasp_pos),
              lambda c: c.reached(pos_tol=REACH_TOL), "GRASP"),
        # 3. 闭环对齐 + 闭合夹爪 + 建立固定约束
        State("GRASP", lambda c: _pose_at(grasp_pos),
              lambda c: c.timed(grasp_pause),
              "LIFT", gripper_fn=do_grasp),
        # 4. 抬升
        State("LIFT", lambda c: _pose_at(lift_pos),
              lambda c: c.reached(pos_tol=REACH_TOL), "MOVE_TO_TARGET"),
        # 5. 移动到垃圾桶上方
        State("MOVE_TO_TARGET", lambda c: _pose_at(target_over_pos),
              lambda c: c.reached(pos_tol=REACH_TOL), "PLACE"),
        # 6. 下降释放（杯子落入桶内）
        State("PLACE", lambda c: _pose_at(target_place_pos),
              lambda c: c.reached(pos_tol=REACH_TOL) and c.timed(place_pause),
              "DONE", gripper_fn=do_release),
    ]

    return StateMachine(states)
