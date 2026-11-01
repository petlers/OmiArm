"""
MuJoCo 真实物理仿真入口：Panda 机械臂 pick-and-place 演示。

架构：
    kinematics/mujoco_kine.py   运动学（解析 Jacobian IK，精度 <1mm）
    drivers/mujoco_arm.py      驱动（真实 PD 位置伺服 + 夹爪 ctrl）
    drivers/grasp_feedback.py  抓取反馈（接触/支撑面判断）

场景（models/panda_mj/pick_place.xml）：
    Panda 7-DOF + 桌子 + 水杯 + 垃圾桶，全部真实物理（重力/接触/摩擦）

演示内容（全程真实物理，不写回任何物体位置）：
    home → 杯子上方 → 下降到杯身中部 → 夹爪闭合压住杯壁 → 抬升 → 垃圾桶上方 → 张开释放

抓取原理：
    指尖碰撞体靠接触力压住杯子的两个侧面，杯子靠【摩擦】被带起来。
    没有 weld / 固定约束 / qpos 瞬移 —— 杯子跟不跟得动，完全由物理说了算。

运行：
    cd /home/phj/Desktop/petlersArm
    python3 examples/run_mujoco.py
"""
import sys
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from petlersArm.kinematics import MuJoCoKinematics
from petlersArm.drivers import MuJoCoArm, GraspFeedback

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models/panda_mj/pick_place.xml")

# ---------------- 参数 ----------------
# hand 原点相对杯心的抓取高度：指尖（pads）比 hand 原点低约 106mm，
# 取 115mm 时指尖落在杯心上方约 9mm，正好夹在杯身中上部，手掌也不碰杯口。
GRASP_ABOVE_CENTER = 0.115
# 接近高度：先到杯子上方再竖直下降，避免横扫撞倒杯子
APPROACH_ABOVE = 0.22
# 抬升高度 / 垃圾桶上方位置
LIFT_HEIGHT = 0.15
BIN_HAND_POS = np.array([0.50, -0.55, 0.42])
# 夹爪闭合力（Panda 模型默认 100 N/m 只有 ~1N 握力，会滑脱）
GRIP_STIFFNESS = 500.0
# 每仿真步的真实等待时间（慢放，看得清接触过程）
SLEEP = 0.006


def main():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    kine = MuJoCoKinematics(model, data, [0, 1, 2, 3, 4, 5, 6], ee_body='hand',
                            finger_bodies=['left_finger', 'right_finger'])
    arm = MuJoCoArm(model, data, [0, 1, 2, 3, 4, 5, 6], gripper_ctrl_idx=7,
                    grip_stiffness=GRIP_STIFFNESS)
    # 抓取反馈：杯子是否还压在桌面上 / 手指是否压住杯子
    fb = GraspFeedback(model, data, object_geom='cup_body',
                       support_bodies=('table', 'floor'),
                       finger_bodies=('left_finger', 'right_finger'))

    # 先让机械臂回 home 位并保持（设 qpos + ctrl，避免位置伺服把臂拉到零位塌陷）
    arm.reset(np.deg2rad([0, 0, 0, -90, 0, 90, -45]))

    # 等水杯落到桌面稳定（机械臂已保持在 home 位）
    for _ in range(3000):
        mujoco.mj_step(model, data)

    cup_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cup_body')
    cup_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cup')
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    cup_pos = data.geom_xpos[cup_geom].copy()     # 杯心（几何中心）
    home_quat = data.xquat[hand_id].copy()    # 抓取姿态：指尖朝下
    grip_idx = 7

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        def tick(steps, sleep=SLEEP):
            """推进物理仿真并刷新画面。"""
            for _ in range(steps):
                mujoco.mj_step(model, data)
                viewer.sync()
                if sleep:
                    time.sleep(sleep)

        def move_hand(target, waypoints=10, settle=20, label="", ease=False):
            """笛卡尔直线插值 → 6-DOF IK → PD 位置伺服到位。

            注意：IK 只算关节角，仿真状态由 actuator 真实驱动；
            路径点之间不做任何 qpos 写回。

            ease=True 用平滑加减速分布路径点 —— 搬运时起步/停止的
            加速度太大会把杯子从夹爪里甩出去。
            """
            start = data.xpos[hand_id].copy()
            s = np.linspace(0.0, 1.0, waypoints)
            if ease:
                s = s * s * (3.0 - 2.0 * s)     # smoothstep 缓入缓出
            path = start + np.outer(s, np.asarray(target, dtype=float) - start)
            if label:
                print(f"  [规划] {label}: {np.round(start, 3)} → "
                      f"{np.round(target, 3)}（{waypoints} 个路径点）")
            for wp in path:
                q = kine.inverse_pose(wp, home_quat,
                                      q_init=arm.get_joint_positions())
                def _sync():
                    viewer.sync()
                    time.sleep(SLEEP)
                arm.goto(q, hold_steps=1200, tol=0.01, step_cb=_sync)
                tick(settle)
            print(f"  [完成] {label} → hand {np.round(data.xpos[hand_id], 3)}"
                  f" 指尖中心 {np.round(arm.gripper_center(), 3)}")

        # ---------------- 演示开始 ----------------
        print("\n=== 机械臂 home 位 ===")
        print(f"夹爪位置: {np.round(arm.gripper_center(), 3)}")
        print(f"水杯位置: {np.round(cup_pos, 3)}（桌面上，尺寸 44×44×80mm）")
        print(f"垃圾桶位置: {np.round(BIN_HAND_POS, 3)}（远处）")
        tick(200)

        print("\n=== 1. 移动到水杯上方（夹爪张开） ===")
        data.ctrl[grip_idx] = 255.0
        tick(300)
        above = cup_pos + np.array([0, 0, APPROACH_ABOVE])
        move_hand(above, waypoints=10, settle=20, label="杯上方")

        print("=== 2. 竖直下降到杯身中部（指尖包住杯壁） ===")
        grasp_pose = cup_pos + np.array([0, 0, GRASP_ABOVE_CENTER])
        move_hand(grasp_pose, waypoints=8, settle=25, label="下降抓取")
        gap = abs(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')][0]
                  - data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')][0])
        print(f"  [对齐] 指尖间距 {gap*1000:.0f}mm > 杯宽 44mm，杯子在两指之间")

        print("=== 3. 闭合手指：接触力压住杯壁（真实接触） ===")
        for c in np.arange(255, -1, -15):
            data.ctrl[grip_idx] = float(c)
            tick(50)
            sep = abs(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right_finger')][0]
                      - data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')][0])
            ncon = fb.finger_contact_count()
            print(f"  [闭合] ctrl={c:5.1f} 指尖间距={sep*1000:5.1f}mm "
                  f"手指-杯接触点={ncon:2d} 杯心={np.round(data.xpos[cup_id], 3)}")

        print("=== 4. 抬升（杯子靠摩擦被夹起，无任何位置写回） ===")
        lift_pose = grasp_pose + np.array([0, 0, LIFT_HEIGHT])
        move_hand(lift_pose, waypoints=12, settle=25, label="抬升")
        r = fb.report()
        print(f"  [反馈] 杯心 z={r['object_z']:.3f}（桌面顶 z=0.22）"
              f" 仍接触桌面={r['touching_support']} 手指接触={r['finger_contact']}"
              f" → 被夹起={r['grasped']}")

        print("=== 5. 移动到垃圾桶上方（保持夹持） ===")
        move_hand(BIN_HAND_POS, waypoints=45, settle=50, label="搬运", ease=True)
        r = fb.report()
        print(f"  [反馈] 移动中保持夹持={r['grasped']} 杯心={np.round(data.xpos[cup_id], 3)}")

        print("=== 6. 张开夹爪释放（杯子落入垃圾桶） ===")
        data.ctrl[grip_idx] = 255.0
        tick(400)
        r = fb.report()
        print(f"  [释放] 杯心={np.round(data.xpos[cup_id], 3)}"
              f" 手指接触={r['finger_contact']} 杯心 z={r['object_z']:.3f}")

        print("\n=== 任务完成：抓取 → 抬升 → 搬运 → 放置（全程真实物理接触） ===")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
