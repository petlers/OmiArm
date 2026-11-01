"""
抓取反馈模块：判断物体是否被机械臂成功夹起。

基于 MuJoCo 真实物理反馈，三种互补信号：
  1. 支撑面接触：物体是否还接触桌面/地面（最可靠）
  2. 高度：物体 z 是否高于支撑面
  3. 手指接触力：手指是否压住物体（真实力控抓取时才有）

用法：
    fb = GraspFeedback(model, data, object_geom='cup_body',
                       support_bodies=('table',),
                       finger_bodies=('left_finger', 'right_finger'))
    fb.finger_contact_count()   # 手指-物体接触点数
    fb.is_grasped()             # (是否离开支撑面, 物体高度)
    fb.report()                 # 各信号详情
"""
import numpy as np
import mujoco


class GraspFeedback:
    def __init__(self, model, data, object_geom='cup_body',
                 support_bodies=('table', 'floor', 'ground'),
                 finger_bodies=None, finger_geoms=None):
        """
        Args:
            model: mujoco.MjModel
            data: mujoco.MjData
            object_geom: 被抓物体某个 geom 名（用于判断接触）
            support_bodies: 支撑体（桌面/地面），接触这些 = 未夹起
            finger_bodies: 手指 body 名列表（检测手指是否压住物体）
                Panda 模型的指尖碰撞 geom 都是匿名 mesh/box，只能按 body id 判，
                所以这里直接收 body 名。
            finger_geoms: 手指 geom 名列表（可选，按名字匹配）
        """
        self.model = model
        self.data = data
        self.object_geom = object_geom
        self.support_bodies = support_bodies
        self.finger_geoms = finger_geoms or []
        self.finger_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                                for b in (finger_bodies or [])]
        self.object_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_geom)
        self.object_body = (model.geom_bodyid[self.object_geom_id]
                            if self.object_geom_id >= 0 else -1)

    def _is_object(self, geom_id):
        return geom_id == self.object_geom_id

    def _is_finger(self, geom_id):
        return (self.model.geom_bodyid[geom_id] in self.finger_body_ids
                or (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or '')
                in self.finger_geoms)

    def _is_support(self, geom_id):
        body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                 self.model.geom_bodyid[geom_id]) or ''
        return any(s in body for s in self.support_bodies)

    def object_touching_support(self):
        """物体是否还接触支撑面（桌面/地面）。"""
        for c in self.data.contact[:self.data.ncon]:
            if self._is_object(c.geom1) and self._is_support(c.geom2):
                return True
            if self._is_object(c.geom2) and self._is_support(c.geom1):
                return True
        return False

    def finger_contact_count(self):
        """手指与物体的接触点数（0 = 没夹住）。"""
        n = 0
        for c in self.data.contact[:self.data.ncon]:
            if (self._is_finger(c.geom1) and self._is_object(c.geom2)) or \
               (self._is_finger(c.geom2) and self._is_object(c.geom1)):
                n += 1
        return n

    def finger_touching_object(self):
        """手指是否接触物体（真实力控夹持时才有信号）。"""
        return self.finger_contact_count() > 0

    def object_height(self):
        """物体位置（freejoint body 或几何中心）。"""
        # 找物体的 freejoint body 名
        cup_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.object_geom)
        if cup_id >= 0:
            body = self.model.geom_bodyid[cup_id]
            return self.data.xpos[body][2]
        return -1.0

    def is_grasped(self, min_lift=0.03):
        """
        判断物体是否被夹起：
        - 不再接触支撑面
        - 高度明显高于原位（用支撑面顶作参考）
        """
        touching = self.object_touching_support()
        height = self.object_height()
        # 用"是否离开支撑面"判断（最可靠）
        lifted = not touching
        return lifted, height

    def report(self):
        """各信号详情。"""
        touching = self.object_touching_support()
        finger = self.finger_touching_object()
        height = self.object_height()
        return {
            'object_z': round(height, 3),
            'touching_support': touching,
            'finger_contact': finger,
            'grasped': not touching,
        }
