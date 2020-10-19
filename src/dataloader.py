import os

import torch
from torch.utils.data import SubsetRandomSampler

from src.dataset import H36M


def train_dataloader(config):
    print(f'[INFO]: Training data loader called')
    dataset = H36M(config.train_subjects, config.annotation_file,
                   config.image_path, config.ignore_images, config.device, config.annotation_path, train=True, projection=config.self_supervised)
    sampler = SubsetRandomSampler(range(0, 5)) if config.fast_dev_run else None
    # shuffle = False if sampler is not None else True

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


def val_dataloader(config, shuffle=True):
    print(f'[INFO]: Validation data loader called')
    dataset = H36M(config.val_subjects, config.annotation_file,
                   config.image_path, config.ignore_images, config.device, config.annotation_path, projection=config.self_supervised)
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        sampler=sampler,
        shuffle=shuffle
    )
    print("samples -", len(loader.dataset))

    return loader


def test_dataloader(config):
    print(f'[INFO]: Test data loader called')
    dataset = H36M(config.subjects, config.annotation_file,
                   config.image_path, config.ignore_images, config.device, config.annotation_path, projection=config.self_supervised)
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


'''
test function
'''


def test():
    from input_reader import Namespace
    config = Namespace()
    config.train_subjects = [1, 5, 6, 7, 8]
    config.annotation_path = f"{os.getenv('HOME')}/lab/HPE3D/src/data"
    config.annotation_file = "h36m17"
    config.image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m/"
    config.batch_size = 4
    config.num_workers = 0
    config.pin_memory = False
    config.ignore_images = False
    config.device = "cpu"
    train_loader = train_dataloader(config)
    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx, len(batch))
        break
        # pass


if __name__ == "__main__":
    test()

