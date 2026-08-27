import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import normalize, randomFlip, ScanDataset
from loops import train, test

from config import train_scans_directory, train_segments_directory
from config import valid_scans_directory, valid_segments_directory
from config import segments_suffix
from config import batch_size, base_learning_rate, halving_patience, model, epochs
from config import checkpoint_directory, checkpoint_file, checkpoint_interval

import os
from pathlib import Path


transformations = [ normalize(), randomFlip([1], 0.5) ] # Acts on single instance of [ Channel, D1, D2, D3 ]
valid_transformations = [ normalize() ] # Acts on single instance of [ Channel, D1, D2, D3 ]

train_dataset = ScanDataset(train_scans_directory, train_segments_directory, segments_suffix, transformations=transformations)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)

valid_dataset = ScanDataset(valid_scans_directory, valid_segments_directory, segments_suffix, transformations=valid_transformations)
valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, drop_last=True)


loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate, eps=1e-4, betas=(0.9, 0.9))
reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=halving_patience)

current_epoch = 1

os.makedirs(checkpoint_directory, exist_ok=True)

if checkpoint_file:
    state = torch.load(Path.joinpath(checkpoint_directory, checkpoint_file))
    current_epoch = state["epoch"]
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

while current_epoch < epochs:
    print(f"Epoch {current_epoch}")
    train(train_dataloader, model, loss_function, optimizer)
    test(valid_dataloader, model, loss_function, reduce_lr)
    if current_epoch % checkpoint_interval == 0 or current_epoch == epochs - 1:
        torch.save({
            "epoch": current_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, Path.joinpath(checkpoint_directory, "model_" + str(current_epoch) + ".pth"))
    current_epoch += 1