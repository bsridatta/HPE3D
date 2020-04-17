import logging

import torch
from torch.utils.data import SubsetRandomSampler

from dataset import H36M

'''
Since the data is same across all subjects, 
the dataloader is same for test/train/val
'''


def train_dataloader(params):
    dataset = H36M(params.subjects, params.annotation_file, params.image_path)
    # alterantive for debug dataset
    # sampler = SubsetRandomSampler(
    #     range(2*params.batch_size)) if params.fast_dev_run else None
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        sampler=sampler
    )
    # if enabling the fastdev method len(dataset) doesnt reflect actual data
    logging.info(
        f'Training data loader called. len - {len(loader)*params.batch_size}')

    return loader


def val_dataloader(params):
    dataset = H36M(params.subjects, params.annotation_file, params.image_path)
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        sampler=sampler
    )

    logging.info(
        f'Validation data loader called. len - {len(loader)*params.batch_size}')

    return loader


def test_dataloader(params):
    dataset = H36M(params.subjects, params.annotation_file, params.image_path)
    sampler = None
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        sampler=sampler
    )

    logging.info(
        f'Test data loader called. len - {len(loader)*params.batch_size}')

    return loader
