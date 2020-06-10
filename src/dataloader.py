import os

import torch
from torch.utils.data import SubsetRandomSampler

from dataset import H36M

'''
Since the data is same across all subjects, 
the dataloader is same for test/train/val
'''


def train_dataloader(config):
    print(f'[INFO]: Training data loader called')
    dataset = H36M(config.train_subjects, config.annotation_file,
                   config.image_path, config.ignore_images, config.device, config.annotation_path)
    # alterantive for debug dataset
    # sampler = SubsetRandomSampler(
    #     range(2*config.batch_size)) if config.fast_dev_run else None

    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=sampler,
        shuffle=True
    )
    # if enabling the fastdev method len(dataset) doesnt reflect actual data !ignore
    print("samples -", len(loader.dataset))
    return loader


def val_dataloader(config):
    print(f'[INFO]: Validation data loader called')
    dataset = H36M(config.val_subjects, config.annotation_file,
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
    config.train_subjects = [1]
    config.annotation_file = "data/debug_h36m17.h5"
    config.image_path = '../../HPE_datasets/h36m/'
    config.batch_size = 4
    config.num_workers = 0
    config.pin_memory = False

    train_loader = train_dataloader(config)
    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx, len(batch))


if __name__ == "__main__":
    test()
