import sys
import os

import torch
from torch.utils.data import SubsetRandomSampler

from src.additional_dataset.h36m_inf_dataset import H36M_MPII
from src.dataset import H36M
from src.additional_dataset.mpiinf_dataset import MPIINF
'''
Since the data is same across all subjects, 
the dataloader is same for test/train/val
'''


def h36m_inf_collate(batch):
    for sample in batch:
        for key in sample.keys():
            if key not in ['pose2d', 'pose3d']:
                sample.pop(key, None)
    return batch


def train_dataloader(config):
    print(f'[INFO]: Training data loader called')
    dataset = H36M_MPII(config.train_subjects, config.annotation_file,
                        config.image_path, config.ignore_images, config.device, config.annotation_path, train=True)
    # dataset = MPIINF(train=True)

    # alterantive for debug dataset
    # sampler = SubsetRandomSampler(
    #     range(2*config.batch_size)) if config.fast_dev_run else None

    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        # pin_memory=config.pin_memory,
        pin_memory=False,
        sampler=sampler,
        shuffle=True,
        # collate_fn=h36m_inf_collate
    )
    # if enabling the fastdev method len(dataset) doesnt reflect actual data !ignore
    print("samples -", len(loader.dataset))
    return loader


def val_dataloader(config):
    print(f'[INFO]: Validation data loader called')
    # dataset = H36M(config.val_subjects, config.annotation_file,
    #                config.image_path, config.ignore_images, config.device, config.annotation_path)
    dataset = MPIINF(train=True)
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        # pin_memory=config.pin_memory,
        pin_memory=False,
        sampler=sampler,
        shuffle=True
    )
    print("samples -", len(loader.dataset))

    return loader


'''
test function for time p
'''


def test_dataloader(config):
    print(f'[INFO]: Test data loader called')
    dataset = H36M(config.subjects, config.annotation_file,
                   config.image_path, config.ignore_images, config.device, config.annotation_path)
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=sampler,
        shuffle=True
    )
    print("samples -", len(loader.dataset))

    return loader


def test():
    from input_reader import Namespace
    config = Namespace()
    config.train_subjects = [1, 5, 6, 7, 8]
    config.annotation_path = f"{os.getenv('HOME')}/lab/HPE3D/src/data"
    config.annotation_file = "h36m17"
    config.image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m/"
    config.batch_size = 4
    config.num_workers = 4
    config.pin_memory = False
    config.ignore_images = True
    config.device = "cpu"
    train_loader = train_dataloader(config)

    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx, len(batch))

        pass


if __name__ == "__main__":
    test()
