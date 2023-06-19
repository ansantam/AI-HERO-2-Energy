#!/usr/bin/env python

import argparse
import random

import numpy as np
import torch
import torch.optim
import torch.utils.data

from dataset import DroneImages
from metric import to_mask, IntersectionOverUnion
from model import MaskRCNN
from tqdm import tqdm


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    # determines the execution device, i.e. CPU or GPU
    device = get_device()
    print(f'Training on {device}')

    # set up the dataset
    drone_images = DroneImages(hyperparameters.root)
    train_data, test_data = torch.utils.data.random_split(drone_images, [0.8, 0.2])

    # initialize MaskRCNN model
    model = MaskRCNN()
    model.to(device)

    # set up optimization procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.lr)
    best_iou = 0.

    # start the actual training procedure
    for epoch in range(hyperparameters.epochs):
        # set the model into training mode
        model.train()
        train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=hyperparameters.batch,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn)

        # training procedure
        train_loss = 0.0
        train_metric = IntersectionOverUnion(task='multiclass', num_classes=2)
        train_metric = train_metric.to(device)
        
        for i, batch in enumerate(tqdm(train_loader, desc='train')):
            x, label = batch
            x = list(image.to(device) for image in x)
            label = [{k: v.to(device) for k, v in l.items()} for l in label]
            model.zero_grad()
            losses = model(x, label)
            loss = sum(l for l in losses.values())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # compute metric
            with torch.no_grad():
                model.eval()
                train_predictions = model(x)
                train_metric(*to_mask(train_predictions, label))
                model.train()
                
        train_loss /= len(train_loader)

        # set the model in evaluation mode
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyperparameters.batch, collate_fn=collate_fn)

        # test procedure
        test_metric = IntersectionOverUnion(task='multiclass', num_classes=2)
        test_metric = test_metric.to(device)
        
        for i, batch in enumerate(tqdm(test_loader, desc='test ')):
            x_test, test_label = batch
            x_test = list(image.to(device) for image in x_test)
            test_label = [{k: v.to(device) for k, v in l.items()} for l in test_label]

            # score_threshold = 0.7
            with torch.no_grad():
                test_predictions = model(x_test)
                test_metric(*to_mask(test_predictions, test_label))

        # output the losses
        print(f'Epoch {epoch}')
        print(f'\tTrain loss: {train_loss}')
        print(f'\tTrain IoU:  {train_metric.compute()}')
        print(f'\tTest IoU:   {test_metric.compute()}')

        # save the best performing model on disk
        if test_metric.compute() > best_iou:
            best_iou = test_metric.compute()
            print('\tSaving better model\n')
            torch.save(model.state_dict(), 'checkpoint.pt')
        else:
            print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=1, help='batch size', type=int)
    # parser.add_argument('-e', '--epochs', default=100, help='number of training epochs', type=int)
    parser.add_argument('-e', '--epochs', default=10, help='number of training epochs', type=int)
    parser.add_argument('-l', '--lr', default=1e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('-s', '--seed', default=42, help='constant random seed for reproduction', type=int)
    # parser.add_argument('root', help='path to the data root', type=str)
    parser.add_argument('--root', default='/hkfs/work/workspace/scratch/ih5525-E4/AI-HERO-2-Energy/energy-train-data/', help='path to the data root', type=str)

    arguments = parser.parse_args()
    train(arguments)
