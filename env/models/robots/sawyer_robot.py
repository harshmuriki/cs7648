import numpy as np

from env.mjcf_utils import array_to_string, xml_path_completion
from env.models.robots.robot import Robot


class Sawyer(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(
        self, use_torque=False, xml_path="robots/sawyer/robot.xml",
    ):
        if use_torque:
            xml_path = "robots/sawyer/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path))

        self.bottom_offset = np.array([0, 0, -0.913])

        # self._init_qpos = np.array([-0.23429241 - 0.4, -1.1364233, 0.336434, 2.18, -0.16150611, 0.31906261 + 0.2, 0])
        self._init_qpos = np.array([-0.28, -0.60, 0.00, 1.86, 0.00, 0.3, 1.57])

        self._model_name = "sawyer"

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.
        """
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """
        Places the robot on position @quat.
        """
        node = self.worldbody.find("./body[@name='base']")
        node.set("quat", array_to_string(quat))

    def set_joint_damping(
        self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
    ):
        """Set joint damping """

        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(
        self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
    ):
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        # return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
        # 0: base, 1: 1st elbow, 3: 2nd elbow 5: 3rd elbow
        return self._init_qpos

    @init_qpos.setter
    def init_qpos(self, init_qpos):
        self._init_qpos = init_qpos

    @property
    def contact_geoms(self):
        return ["right_l{}_collision".format(x) for x in range(2, 7)]

    # @property
    # def _base_body(self):
    #     node = self.worldbody.find("./body[@name='base']")
    #     body = node.find("./body[@name='right_arm_base_link']")
    #     return body

    @property
    def _link_body(self):
        return [
            "right_l0",
            "right_l1",
            "right_l2",
            "right_l3",
            "right_l4",
            "right_l5",
            "right_l6",
        ]

    @property
    def _joints(self):
        return [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]

    @property
    def contact_geoms(self):
        return [
            "controller_box_col",
            "pedestal_feet_col",
            "torso_col",
            "pedestal_col1",
            "pedestal_col2",
            "right_arm_base_link_col1",
            "right_arm_base_link_col2",
            "right_l0_col1",
            "right_l0_col2",
            "head_col1",
            "head_col2",
            "screen_col1",
            "screen_col2",
            "right_l1_col1",
            "right_l1_col2",
            "right_l2_col1",
            "right_l2_col2",
            "right_l3_col1",
            "right_l3_col2",
            "right_l4_col1",
            "right_l4_col2",
            "right_l5_col1",
            "right_l5_col2",
            "right_l6_col1",
            "right_l6_col2",
            "right_l4_2_col",
            "right_l2_2_col",
            "right_l1_2_col",
        ]
