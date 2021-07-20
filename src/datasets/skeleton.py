from typing import List, Tuple


class Skeleton:
    """Common 16 joint template for all datasets"""

    def __init__(self) -> None:
        self.joints = self.get_joints()
        self.connections = self.get_connections()
        self.root_idx = self.joints.index("Pelvis")

        self.bones = tuple(
            [
                (self.joints.index(i), self.joints.index(j))
                for (i, j) in self.connections
            ]
        )
        # without pelvis as its removed in the preprocessing step after zeroing
        self.joints_15 = self.joints.copy()
        self.joints_15.remove("Pelvis")

        self.flipped_indices = self.get_flipped_indices(self.joints_15)

    @staticmethod
    def get_flipped_indices(joints: List[str]):
        flipped_indices = []
        for idx, i in enumerate(joints):
            if "R_" in i:
                flipped_indices.append(joints.index(i.replace("R_", "L_")))
            elif "L_" in i:
                flipped_indices.append(joints.index(i.replace("L_", "R_")))
            else:
                flipped_indices.append(idx)
        return flipped_indices

    @staticmethod
    def get_joints() -> List[str]:
        """easy to read the joint ids"""
        joints: List[str] = [""] * 16
        joints[0] = "Pelvis"
        joints[1] = "R_Hip"
        joints[2] = "R_Knee"
        joints[3] = "R_Ankle"
        joints[4] = "L_Hip"
        joints[5] = "L_Knee"
        joints[6] = "L_Ankle"
        joints[7] = "Torso"
        joints[8] = "Neck"
        joints[9] = "Head"
        joints[10] = "L_Shoulder"
        joints[11] = "L_Elbow"
        joints[12] = "L_Wrist"
        joints[13] = "R_Shoulder"
        joints[14] = "R_Elbow"
        joints[15] = "R_Wrist"

        return joints



    @staticmethod
    def get_connections() -> Tuple[Tuple[str, str], ...]:
        connections = (
            ("Pelvis", "Torso"),
            ("Torso", "Neck"),
            ("Neck", "Head"),
            ("Neck", "L_Shoulder"),
            ("L_Shoulder", "L_Elbow"),
            ("L_Elbow", "L_Wrist"),
            ("Neck", "R_Shoulder"),
            ("R_Shoulder", "R_Elbow"),
            ("R_Elbow", "R_Wrist"),
            ("Pelvis", "R_Hip"),
            ("R_Hip", "R_Knee"),
            ("R_Knee", "R_Ankle"),
            ("Pelvis", "L_Hip"),
            ("L_Hip", "L_Knee"),
            ("L_Knee", "L_Ankle"),
        )
        return connections
