# petlersArm — 7-DOF 机械臂算法框架（MuJoCo 真实物理）

参考 [Dofbot-Simulation-and-Control](../arm/Dofbot-Simulation-and-Control) 提炼的通用机械臂控制框架，
**仿真 + 真机两用**。算法层与硬件完全解耦。

**仿真引擎：MuJoCo** — 重力 / 接触 / 摩擦 / 动力学全部真实求解。
抓取不写回物体状态、不加 weld 约束、不禁用碰撞：杯子跟不跟得起来，完全由物理说了算。

## 架构

```
tasks/           任务层：状态机、抓取放置编排（硬件无关；基于旧 PyBullet 方案，见"已知限制"）
kinematics/      运动学：FK / IK（解析 Jacobian + 阻尼最小二乘）
core/            核心抽象：ArmInterface 接口（硬件无关）
drivers/         驱动层：MuJoCo / 舵机 / DM 电机（唯一硬件相关层）
examples/        入口示例
models/panda_mj/ Panda 官方 MJCF（DeepMind menagerie）+ 本项目场景 pick_place.xml
```

**核心思想**：算法只依赖 `ArmInterface` 抽象接口。换硬件 = 换驱动层 + 换模型，算法零改动。

## 快速开始

```bash
pip install mujoco numpy          # 实测 MuJoCo 3.11
python3 examples/run_mujoco.py    # MuJoCo viewer：Panda 物理抓取水杯 → 放进垃圾桶
```

场景：桌面 + 44×44×80mm 水杯（0.12kg）+ 垃圾桶。
演示内容（约 1 分钟，慢放）：

| 步骤 | 内容 |
|------|------|
| 1 | 夹爪张开，移到杯子上方 |
| 2 | 竖直下降到杯身中部，两指包住杯壁 |
| 3 | 夹爪闭合：手指-杯接触点数 0 → 32，指尖间距收到 43.5mm（杯宽 44mm）|
| 4 | 抬升：杯子离开桌面被夹起（`手指接触=True`、`仍接触桌面=False`）|
| 5 | 搬运到垃圾桶上方，全程保持夹持 |
| 6 | 张开夹爪，杯子落进桶里 |

## 运动学（MuJoCo，解析 Jacobian IK）

```python
from petlersArm.kinematics import MuJoCoKinematics

kine = MuJoCoKinematics(model, data, joint_qposadr=[0..6], ee_body='hand',
                        finger_bodies=['left_finger', 'right_finger'])

pos, quat = kine.forward(q)                       # FK
q = kine.inverse(pos)                             # IK：位置（末端 hand）
q = kine.inverse_fingertip(pos)                   # IK：指尖中心（抓取对齐）
q = kine.inverse_pose(pos, quat, q_init=q_now)    # 6-DOF IK：位置 + 姿态（抓取用）
```

- 迭代方法是阻尼最小二乘（DLS）。`lam` 越小姿态越准、越接近奇异时越抖；
  实测 **`lam=0.01 / tol=0.001`** 能把姿态残差压到 <1.5°（`lam=0.05` 只剩 ~4°）。
- **IK 是纯计算**：迭代过程会临时把解写进 `data.qpos` 算 FK/Jacobian，函数返回前会还原
  仿真状态。所以调用 IK 不会"瞬移"机械臂，后续运动仍由 actuator 真实驱动。
- `q_init` 默认取当前关节角 —— 别用全零起步，Panda 的 joint4 限位是 `[-3.07, -0.07]`，容易跑到限位上。

## 驱动（真实 PD 位置伺服）

```python
from petlersArm.drivers import MuJoCoArm

arm = MuJoCoArm(model, data, joint_qposadr=[0..6], gripper_ctrl_idx=7,
                grip_stiffness=500.0)

arm.reset(q_home)                     # 回 home 位（同时写 qpos + ctrl，防止塌陷）
arm.goto(q, tol=0.01, step_cb=sync)   # 位置伺服：PD 动力学收敛，不做位置瞬移
arm.gripper_open()                    # ctrl=255，指尖张开 ~80mm
arm.gripper_close()                   # ctrl=0，闭合到接触/最短
```

- `goto(..., step_cb=fn)`：每个仿真步后回调，传 `viewer.sync` 就能看到完整的伺服过程，
  而不是一帧跳到目标。
- `grip_stiffness`：夹爪 tendon 执行器的刚度。Panda 原模型只有 100 N/m，
  对 44mm 杯子只能压出 ~1N 法向力，摩擦握力不够、抬起来会滑脱。
  改成 500 N/m（≈10N 握力）后稳定夹持，而 `ctrl → 开口` 的映射保持不变。

## 抓取反馈（真实物理信号）

```python
from petlersArm.drivers import GraspFeedback

fb = GraspFeedback(model, data, object_geom='cup_body',
                   support_bodies=('table', 'floor'),
                   finger_bodies=('left_finger', 'right_finger'))

fb.finger_contact_count()    # 手指-物体接触点数（0 = 没夹住）
fb.object_touching_support() # 物体是否还压在桌面上
fb.report()                  # {'object_z', 'touching_support', 'finger_contact', 'grasped'}
```

Panda 的指尖碰撞 geom 全是匿名 mesh / box，所以手指判定按 **body id** 做，不要按名字匹配。

## 物理抓取的三个必要前提

并行两指夹住 44mm 方杯不是"摆好位置就行"，实测有三个必要条件：

| 前提 | 说明 |
|------|------|
| 夹爪闭合力 | 刚度 100 → 500 N/m（见上），否则握力 < 摩擦需求，抬升即滑脱 |
| 姿控精度 | 6-DOF IK 必须收敛到 <1.5°，否则两指高度不一致，杯子被推歪、只夹到上沿 |
| 抓取高度 | hand 原点比指尖高约 106mm；取 `杯心 + 0.115` 时指尖落在杯身中上部，手掌不碰杯口 |

另外搬运阶段要**限速 + 缓入缓出**（demo 里 `move_hand(..., ease=True)` 用 smoothstep）：
横向加速度过大会把杯子从夹爪里甩出去。

## 换真机（关键）

| 硬件 | 驱动实现 | 说明 |
|------|---------|------|
| 仿真（MuJoCo） | `drivers/mujoco_arm.py` | ✅ 已完成，真实物理 |
| 仿真（PyBullet，旧） | `drivers/simulated.py` | 已废弃（物理不可靠） |
| 舵机（SCS/串口） | `drivers/servo_driver.py` | stub，需填协议 |
| DM 电机（CAN） | `drivers/dm_motor_driver.py` | stub，MIT 模式协议已写 |


