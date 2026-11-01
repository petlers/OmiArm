"""
petlersArm —— 7-DOF 机械臂算法框架（仿真 + 真机两用）

架构（三层分离）：
    tasks/        任务层：状态机、抓取放置逻辑（与硬件无关）
    kinematics/   运动学：FK/IK（与硬件无关）
    drivers/      驱动层：仿真 / 舵机 / DM 电机（唯一的硬件相关层）

算法层只依赖 ArmInterface 抽象接口，换硬件 = 换驱动层 + 换 URDF，算法零改动。
"""
__version__ = "0.1.0"
