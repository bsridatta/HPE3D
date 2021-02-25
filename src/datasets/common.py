from typing import List


COMMON_JOINTS: List[str] = ['']*16
COMMON_JOINTS[0] = 'Pelvis'  
COMMON_JOINTS[1] = 'R_Hip'
COMMON_JOINTS[2] = 'R_Knee'
COMMON_JOINTS[3] = 'R_Ankle'
COMMON_JOINTS[4] = 'L_Hip'
COMMON_JOINTS[5] = 'L_Knee'
COMMON_JOINTS[6] = 'L_Ankle'
COMMON_JOINTS[7] = 'Torso'
COMMON_JOINTS[8] = 'Neck' 
COMMON_JOINTS[9] = 'Head'
COMMON_JOINTS[10] = 'L_Shoulder'
COMMON_JOINTS[11] = 'L_Elbow'
COMMON_JOINTS[12] = 'L_Wrist'
COMMON_JOINTS[13] = 'R_Shoulder'
COMMON_JOINTS[14] = 'R_Elbow'
COMMON_JOINTS[15] = 'R_Wrist'


JOINT_CONNECTIONS = (('Pelvis', 'Torso'), ('Torso', 'Neck'), ('Neck', 'Head'), ('Neck', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_Wrist'), ('Neck', 'R_Shoulder'), (
            'R_Shoulder', 'R_Elbow'), ('R_Elbow', 'R_Wrist'), ('Pelvis', 'R_Hip'), ('R_Hip', 'R_Knee'), ('R_Knee', 'R_Ankle'), ('Pelvis', 'L_Hip'), ('L_Hip', 'L_Knee'), ('L_Knee', 'L_Ankle'))

BONES = tuple([(COMMON_JOINTS.index(i),COMMON_JOINTS.index(j))
                            for (i, j) in JOINT_CONNECTIONS])