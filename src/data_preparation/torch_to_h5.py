# %%
from collections import defaultdict
from typing import Dict, Union, Tuple, List
from numpy.core.arrayprint import SubArrayFormat
import torch
import os
import h5py
import numpy as np
from src.data_preparation.get_metadata import action_to_id, camera_id_to_num


def split_metadata(metadata: Tuple[int, str, str]) -> Tuple[int, int, int, int]:
    """split metadata to components
    eg: (11, 'WalkDog', 'WalkDog 1.58860488.h5-sh')
        (9, 'Directions', 'Directions.55011271.h5-sh')

    Args:
        metadata (Tuple[int, str, str]): metadata from pickle

    Returns:
        Tuple[int, int, int, int]: subject, action, subaction, camera_num
    """

    subject: int = metadata[0]
    action: int = action_to_id(metadata[1])

    chunks: List[str] = metadata[2].split(".")
    camera_num: int = camera_id_to_num(chunks[1])

    chunks: List[str] = chunks[0].split(" ")
    subaction: int = int(chunks[1]) if len(chunks) > 1 else 0

    return (subject, action, subaction, camera_num)


path: str = f'{os.getenv("HOME")}/lab/HPE_datasets/torch_pickles/'

# filename:str = "test_2d_ft"

filename: str = "train_2d_ft"


pickle: Dict[Tuple[int, str, str], List[int]
             ] = torch.load(path+f"{filename}.pth.tar")

save_path = f'{os.getenv("HOME")}/lab/HPE3D/src/data/'

is_sh = 'sh' if "ft" in filename else ''

if "test" in filename:
    h5name = f'h36m17{is_sh}_911'
    skip_frame = 64

else:
    h5name = f'h36m17{is_sh}_15678'
    skip_frame = 5


data: Dict[str, Union[int, np.ndarray]] = defaultdict(list)

# %%
for recording_name, frames in pickle.items():
    poses = frames[::skip_frame]
    data['pose2d'].extend(poses)

    values = split_metadata(recording_name)
    data['subject'].extend(len(poses)*[values[0]])
    data['action'].extend(len(poses)*[values[1]])
    data['subaction'].extend(len(poses)*[values[2]])
    data['camera'].extend(len(poses)*[values[3]])

    print(len(data["pose2d"]), len(data["subject"]))

print(f"Number of samples: {len(data['pose2d'])}")

f = h5py.File(save_path+h5name+'.h5', 'w')
for key in data.keys():
    f[key] = data[key]
f.close()

print(f"Saved to {save_path+h5name+'.h5'}")


# %%
