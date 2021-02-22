import os
import torch
from src.dataset import H36M
from torch.utils.data import SubsetRandomSampler


def train_dataloader(config):
    print(f'[INFO]: Training data loader called')
    dataset = H36M(config.train_file, config.image_path,
                   train=True, projection=config.self_supervised)

    sampler = SubsetRandomSampler(
        range(0, 10)) if config.fast_dev_run else None
    shuffle = False if sampler else True
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         pin_memory=config.pin_memory,
                                         sampler=sampler,
                                         shuffle=shuffle
                                         )
    # if enabling the fastdev method len(dataset) doesnt reflect actual data !ignore
    print(f"[INFO]: Samples in loader: {len(loader)*loader.batch_size}")
    return loader


def val_dataloader(config, shuffle=True):
    print(f'[INFO]: Validation data loader called')
    dataset = H36M(config.test_file, config.image_path,
                   train=True, projection=config.self_supervised)

    # TODO use test loader for test.py
    sampler = SubsetRandomSampler(
        range(0, 10)) if config.fast_dev_run else None

    if sampler:
        shuffle = False

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         pin_memory=config.pin_memory,
                                         sampler=sampler,
                                         shuffle=shuffle
                                         )
    print(f"[INFO]: Samples in loader: {len(loader)*loader.batch_size}")

    return loader


def test_dataloader(config):
    print(f'[INFO]: Test data loader called')
    dataset = H36M(config.test_file, config.image_path,
                   train=False, projection=config.self_supervised)

    sampler = SubsetRandomSampler(
        range(0, 10)) if config.fast_dev_run else None
    shuffle = False if sampler else True

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         pin_memory=config.pin_memory,
                                         sampler=sampler,
                                         shuffle=shuffle
                                         )
    print(f"[INFO]: Samples in loader: {len(loader)*loader.batch_size}")

    return loader


'''
test function
'''


def test():
    from input_reader import Namespace
    config = Namespace()
    config.test_file = f"{os.getenv('HOME')}/lab/HPE3D/src/data/h36m_test_gt_2d.h5"
    config.batch_size = 4
    config.num_workers = 0
    config.pin_memory = False
    config.image_path = ""
    config.self_supervised = True
    config.fast_dev_run = True

    train_loader = val_dataloader(config)
    print()
    for batch_idx, batch in enumerate(train_loader):
        print(f"batch: {batch_idx}, batch_size: {len(batch)}")
        break
        # pass


if __name__ == "__main__":
    test()
