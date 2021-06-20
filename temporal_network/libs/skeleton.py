import numpy as np
from .joint import Joint2d, Joint3d

class Skeleton():
    def __init__(self):
        
        # Skeleton Info
        self.num_joints = 17

        self.root_joint = 'lower_spine'
        self.body_parts = {
            'torso': ['lower_spine', 'mid_spine', 'upper_spine'],
            'left_leg': ['lower_spine', 'left_hip', 'left_knee', 'left_ankle'],
            'right_leg': ['lower_spine', 'right_hip', 'right_knee', 'right_ankle'],
            'left_arm': ['upper_spine', 'left_shoulder', 'left_elbow', 'left_hand'],
            'right_arm': ['upper_spine', 'right_shoulder', 'right_elbow', 'right_hand'],
            'head': ['upper_spine', 'neck', 'nose']
        }

        self.dict_joint_2_idx = {
            'lower_spine': 0,
            'right_hip': 1,
            'right_knee': 2,
            'right_ankle': 3,
            'left_hip': 4,
            'left_knee': 5,
            'left_ankle': 6,
            'mid_spine': 7,
            'upper_spine': 8,
            'neck': 9,
            'nose': 10,
            'left_shoulder': 11,
            'left_elbow': 12,
            'left_hand': 13,
            'right_shoulder': 14,
            'right_elbow': 15,
            'right_hand': 16
        }

        self.dict_idx_2_joint = {
            0: 'lower_spine',
            1: 'right_hip',
            2: 'right_knee',
            3: 'right_ankle',
            4: 'left_hip',
            5: 'left_knee',
            6: 'left_ankle',
            7: 'mid_spine',
            8: 'upper_spine',
            9: 'neck',
            10: 'nose',
            11: 'left_shoulder',
            12: 'left_elbow',
            13: 'left_hand',
            14: 'right_shoulder',
            15: 'right_elbow',
            16: 'right_hand'
        }

class Skeleton2d(Skeleton):
    def __init__(self, joints, is_cartesian=True):
        super(Skeleton2d, self).__init__()
        self._build_skeleton(joints, is_cartesian)
        self.joints = joints

    def _build_skeleton(self, joints, is_cartesian):

        if is_cartesian:
            setattr(self, self.root_joint, Joint2d(self.root_joint, cartesian_coord=joints[self.dict_joint_2_idx[self.root_joint]]))
            getattr(self, self.root_joint).set_polar_coord()
            for body_part in self.body_parts.keys():
                for i, joint_name in enumerate(self.body_parts[body_part]):
                    if i == 0:
                        continue
                    else:
                        joint = Joint2d(joint_name, cartesian_coord=joints[self.dict_joint_2_idx[joint_name]])
                        joint.set_parent(getattr(self, self.body_parts[body_part][i-1]))
                        joint.set_polar_coord()
                        setattr(self, joint_name, joint)
        else:
            setattr(self, self.root_joint, Joint2d(self.root_joint, polar_coord=joints[self.dict_joint_2_idx[self.root_joint]]))
            getattr(self, self.root_joint).set_cartesian_coord()
            for body_part in self.body_parts.keys():
                for i, joint_name in enumerate(self.body_parts[body_part]):
                    if i == 0:
                        continue
                    else:
                        joint = Joint2d(joint_name, polar_coord=joints[self.dict_joint_2_idx[joint_name]])
                        joint.set_parent(getattr(self, self.body_parts[body_part][i-1]))
                        joint.set_cartesian_coord()
                        setattr(self, joint_name, joint)

    def get_cartesian_joints(self):
        cartesian_joints = np.zeros((self.num_joints, 2))
        for i in range(self.num_joints):
            cartesian_joints[i] = getattr(self, self.dict_idx_2_joint[i]).cartesian_coord

        return cartesian_joints

    def get_polar_joints(self):
        polar_joints = np.zeros((self.num_joints, 2))
        for i in range(self.num_joints):
            polar_joints[i] = getattr(self, self.dict_idx_2_joint[i]).polar_coord

        return polar_joints


class Skeleton3d(Skeleton):
    def __init__(self, joints, is_cartesian=True):
        super(Skeleton3d, self).__init__()
        self._build_skeleton(joints, is_cartesian)
        self.joints = joints


    def _build_skeleton(self, joints, is_cartesian):
        if is_cartesian:
            setattr(self, self.root_joint, Joint3d(self.root_joint, cartesian_coord=joints[self.dict_joint_2_idx[self.root_joint]]))
            getattr(self, self.root_joint).set_spherical_coord()
            for body_part in self.body_parts.keys():
                for i, joint_name in enumerate(self.body_parts[body_part]):
                    if i == 0:
                        continue
                    else:
                        joint = Joint3d(joint_name, cartesian_coord=joints[self.dict_joint_2_idx[joint_name]])
                        joint.set_parent(getattr(self, self.body_parts[body_part][i-1]))
                        joint.set_spherical_coord()
                        setattr(self, joint_name, joint)

        else:
            setattr(self, self.root_joint, Joint3d(self.root_joint, spherical_coord=joints[self.dict_joint_2_idx[self.root_joint]]))
            getattr(self, self.root_joint).set_cartesian_coord()
            for body_part in self.body_parts.keys():
                for i, joint_name in enumerate(self.body_parts[body_part]):
                    if i == 0:
                        continue
                    else:
                        joint = Joint3d(joint_name, spherical_coord=joints[self.dict_joint_2_idx[joint_name]])
                        joint.set_parent(getattr(self, self.body_parts[body_part][i-1]))
                        joint.set_cartesian_coord()
                        setattr(self, joint_name, joint)

    def get_cartesian_joints(self):
        cartesian_joints = np.zeros((self.num_joints, 3))
        for i in range(self.num_joints):
            cartesian_joints[i] = getattr(self, self.dict_idx_2_joint[i]).cartesian_coord

        return cartesian_joints

    def get_spherical_joints(self):
        spherical_joints = np.zeros((self.num_joints, 3))
        for i in range(self.num_joints):
            spherical_joints[i] = getattr(self, self.dict_idx_2_joint[i]).spherical_coord

        return spherical_joints