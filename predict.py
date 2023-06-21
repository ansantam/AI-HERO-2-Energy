#!/usr/bin/env python

import argparse
import random

import numpy as np
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from collections import OrderedDict

from dataset import DroneImages
from metric import to_mask, IntersectionOverUnion
from model import MaskRCNN
from tqdm import tqdm
import os


def ddp_setup_torchrun():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def predict(hyperparameters: argparse.Namespace):
    ddp_setup_torchrun()
    device = int(os.environ["LOCAL_RANK"])

    # if in grayscale mode
    if hyperparameters.grayscale:
        n_channels = 3
    else:
        n_channels = 5
    
    # set up the dataset
    drone_images = DroneImages(hyperparameters.root, grayscale=hyperparameters.grayscale, normalize=hyperparameters.normalize)
    test_data = drone_images

    test_sampler = DistributedSampler(test_data, shuffle=False)

    # initialize the Mask-RCNN model
    model = MaskRCNN(in_channels=n_channels, num_classes=2, backbone=hyperparameters.backbone)
    # use all gpus 
    model.to(device)

    if hyperparameters.model:
        print(f'Restoring model checkpoint from {hyperparameters.model}')
        state_dict = torch.load(hyperparameters.model)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    # model = DistributedDataParallel(model, device_ids=[device])
    # set the model in evaluation mode
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyperparameters.batch, pin_memory=True, sampler=test_sampler, collate_fn=collate_fn)

    # test procedure
    test_metric = IntersectionOverUnion(task='multiclass', num_classes=2)
    test_metric = test_metric.to(device)
    
    for _, batch in enumerate(tqdm(test_loader, desc='test')):
        x_test, test_label = batch
        x_test = list(image.to(device, non_blocking=True) for image in x_test)
        test_label = [{k: v.to(device, non_blocking=True) for k, v in l.items()} for l in test_label]

        # score_threshold = 0.7
        with torch.no_grad():
            test_predictions = model(x_test)
            test_metric(*to_mask(test_predictions, test_label))

    print(f'Test IoU: {test_metric.compute()}')

    destroy_process_group()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=1, help="batch size", type=int)
    parser.add_argument('-m', '--model', default='checkpoint.pt', help='model checkpoint', type=str)
    parser.add_argument(
        "root",
        default="/hkfs/work/workspace/scratch/ih5525-E4/AI-HERO-2-Energy/energy-train-data/",
        help="path to the data root",
        type=str,
    )
    parser.add_argument(
        "--normalize", default=False, help="normalize the images", type=bool
    )
    parser.add_argument(
        "--optimizer", default="adam", help="choose optimizer", type=str
    )
    parser.add_argument("--autocast", default=False, help="use autocast", type=bool)
    parser.add_argument(
        "--backbone", default="resnet50", help="choose backbone", type=str
    )
    # option to use grayscale
    parser.add_argument("--grayscale", default=False, help="use grayscale", type=bool)
    # parser.add_argument("--n_gpus", default=4, help="number of gpus", type=int)
    arguments = parser.parse_args()
    predict(arguments)
