import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import normalize, randomFlip, ScanDataset
from loops import train, test

from config import device
from config import train_scans_directory, train_segments_directory
from config import valid_scans_directory, valid_segments_directory
from config import segments_suffix, edge_boost_config
from config import batch_size, model, epochs
from config import checkpoint_directory, checkpoint_file
from config import checkpoint_interval


transformations = [
    normalize(),
    randomFlip([1], 0.5)
]

valid_transformations = [
    normalize()
]


train_dataset = ScanDataset(
    train_scans_directory,
    train_segments_directory,
    segments_suffix,
    edge_boost_config,
    transformations=transformations
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)


valid_dataset = ScanDataset(
    valid_scans_directory,
    valid_segments_directory,
    segments_suffix,
    edge_boost_config,
    transformations=valid_transformations
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False
)


loss_function = nn.BCEWithLogitsLoss(reduction="none")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    eps=1e-4,
    betas=(0.9, 0.9)
)

reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.5,
    patience=4
)

scaler = torch.cuda.amp.GradScaler(
    enabled=device.type == "cuda"
)

current_epoch = 1

os.makedirs(checkpoint_directory, exist_ok=True)


if checkpoint_file:
    checkpoint_path = Path(checkpoint_directory) / checkpoint_file

    state = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])

    if "scheduler" in state:
        reduce_lr.load_state_dict(state["scheduler"])

    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])

    current_epoch = state["epoch"] + 1

    print(
        f"Resuming from checkpoint {checkpoint_file}, "
        f"starting at epoch {current_epoch}"
    )


while current_epoch <= epochs:
    print(f"\nEpoch {current_epoch}/{epochs}")

    training_loss = train(
        train_dataloader,
        model,
        loss_function,
        optimizer,
        scaler
    )

    validation_loss = test(
        valid_dataloader,
        model,
        loss_function,
        reduce_lr
    )

    current_learning_rate = optimizer.param_groups[0]["lr"]

    print(f"Training loss:   {training_loss:.6f}")
    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Learning rate:   {current_learning_rate:.8f}")

    if (
        current_epoch % checkpoint_interval == 0
        or current_epoch == epochs
    ):
        checkpoint_path = (
            Path(checkpoint_directory)
            / f"model_{current_epoch}.pth"
        )

        torch.save(
            {
                "epoch": current_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": reduce_lr.state_dict(),
                "scaler": scaler.state_dict(),
                "training_loss": training_loss,
                "validation_loss": validation_loss
            },
            checkpoint_path
        )

        print(f"Saved checkpoint: {checkpoint_path}")

    current_epoch += 1