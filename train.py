#!/usr/bin/env python

import argparse
import random
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
# import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from dataset import DroneImages
from metric import to_mask, IntersectionOverUnion
from model import MaskRCNN
# from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


# def get_device() -> torch.device:
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(hyperparameters: argparse.Namespace):
    ddp_setup_torchrun()
    device = int(os.environ["LOCAL_RANK"])
    # ddp_setup(rank, world_size)
    # initialize wandb
    hyperparameters = wandb.config

    # create a folder to save the model
    save_folder = f"./models/{wandb.run.name}"
    os.makedirs(save_folder, exist_ok=True)



    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    # determines the execution device, i.e. CPU or GPU
    # device = get_device()
    print(f"Training on gpu:{device}")

    # if in grayscale mode
    if hyperparameters.grayscale:
        n_channels = 3
    else:
        n_channels = 5

    # set up the dataset
    drone_images = DroneImages(hyperparameters.root, grayscale=hyperparameters.grayscale, normalize=hyperparameters.normalize)
    train_data, test_data = torch.utils.data.random_split(
        drone_images, [hyperparameters.split, 1 - hyperparameters.split]
    )

    train_sampler = DistributedSampler(train_data)
    test_sampler = DistributedSampler(test_data, shuffle=False)

    # initialize MaskRCNN model
    model = MaskRCNN(
        trainable_backbone_layers=hyperparameters.n_trainablebackbone,
        weights=hyperparameters.weight,
        in_channels=n_channels,
        backbone=hyperparameters.backbone,
    )
    # load model checkpoint if available
    if hyperparameters.load:
        print(f"Restoring model checkpoint from {hyperparameters.load}")
        model.load_state_dict(torch.load(hyperparameters.load))
    
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device])

    # set up optimization procedure
    if hyperparameters.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.lr)
    elif hyperparameters.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyperparameters.lr, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer {hyperparameters.optimizer}")
    best_iou = 0.0

    # set up mixed precision training
    if hyperparameters.autocast:
        scaler = GradScaler()

    # set up data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=hyperparameters.batch,
        # num_workers=1,
        pin_memory=True,
        # shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hyperparameters.batch,
        # num_workers=2,
        pin_memory=True,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    # start the actual training procedure
    for epoch in range(hyperparameters.epochs):
        # set the model into training mode
        model.train()

        # training procedure
        train_loss = 0.0
        train_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
        train_metric = train_metric.to(device)

        # set epoch for sampler
        train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            x, label = batch
            x = list(image.to(device, non_blocking=True) for image in x)
            label = [{k: v.to(device, non_blocking=True) for k, v in l.items()} for l in label]
            model.zero_grad()
            with autocast(enabled=hyperparameters.autocast, dtype=torch.float16):
                losses = model(x, label)
                loss = sum(l for l in losses.values())

            if hyperparameters.autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # loss.backward()
            # optimizer.step()
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

        # test procedure
        test_metric = IntersectionOverUnion(task="multiclass", num_classes=2)
        test_metric = test_metric.to(device)

        for i, batch in enumerate(test_loader):
            x_test, test_label = batch
            x_test = list(image.to(device) for image in x_test)
            test_label = [{k: v.to(device) for k, v in l.items()} for l in test_label]

            # score_threshold = 0.7
            with torch.no_grad():
                test_predictions = model(x_test)
                test_metric(*to_mask(test_predictions, test_label))

        train_iou = train_metric.compute()
        test_iou = test_metric.compute()
        # output the losses
        if device == 0:
            print(f"Epoch {epoch}")
            print(f"\tTrain IoU:  {train_iou}")
            print(f"\tTest IoU:   {test_iou}")
        print(f"\t GPU:{device} \tTrain loss: {train_loss}")

        # log the metrics to wandb
        wandb.log(
            {"train_loss": train_loss, "train_iou": train_iou, "test_iou": test_iou}
        )

        # save the model
        if device == 0:
            torch.save(model.state_dict(), f"{save_folder}/checkpoint_{epoch}.pt")
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_folder}/training_ckp_{epoch}.pt")
            # torch.save(model.module.state_dict(), f"{save_folder}/checkpoint_{epoch}.pt")
            # save the best performing model on disk
            if test_iou > best_iou:
                best_iou = test_iou
                print("\tSaving better model\n")
                torch.save(model.state_dict(), f"{save_folder}/checkpoint_best.pt")
            else:
                print("\n")
    destroy_process_group()


# def ddp_setup(rank, world_size):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)

def ddp_setup_torchrun():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", default=1, help="batch size", type=int)
    parser.add_argument(
        "-n",
        "--n_trainablebackbone",
        default=5,
        help="number of trainable backbone layers",
        type=int,
    )
    parser.add_argument(
        "-p", "--split", default=0.9, help="train evaluation split", type=float
    )
    # parser.add_argument('-e', '--epochs', default=100, help='number of training epochs', type=int)
    parser.add_argument(
        "-e", "--epochs", default=10, help="number of training epochs", type=int
    )
    parser.add_argument(
        "-l", "--lr", default=1e-4, help="learning rate of the optimizer", type=float
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=42,
        help="constant random seed for reproduction",
        type=int,
    )
    # parser.add_argument('root', help='path to the data root', type=str)
    parser.add_argument(
        "--root",
        default="/hkfs/work/workspace/scratch/ih5525-E4/AI-HERO-2-Energy/energy-train-data/",
        help="path to the data root",
        type=str,
    )
    parser.add_argument(
        "--weight", default="DEFAULT", help="use pretrained weights", type=str
    )
    parser.add_argument(
        "--normalize", default=False, help="normalize the images", type=bool
    )
    parser.add_argument(
        "--load", default=None, help="load a model from a checkpoint", type=str
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

    # truly random seed
    if arguments.seed == -1:
        arguments.seed = random.randint(0, 100000)
    # set up wandb
    wandb.init(entity="ibpt-ml", project="aihero-energy", config=arguments)
    # wandb.config["world_size"] = world_size
    arguments = wandb.config
    
    # log the hyperparameters
    config_dict = dict(wandb.config)
    for key in config_dict.keys():
        print(f"{key}: {config_dict[key]}")
    # set up multiprocessing
    # mp.spawn(train, nprocs=4, args=(arguments,))
    train(arguments)
    # train(arguments)
