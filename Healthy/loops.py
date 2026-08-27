import torch
from config import device


def train(
    dataloader,
    model,
    unreduced_loss_function,
    optimizer,
    scaler
):
    model.train()
    total_loss = 0.0

    if len(dataloader) == 0:
        raise ValueError("Training dataloader is empty.")

    for x, y, loss_weights in dataloader:
        x = x.to(device)
        y = y.to(device)
        loss_weights = loss_weights.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            prediction = model(x)

            voxel_loss = unreduced_loss_function(prediction, y)
            weighted_loss = voxel_loss * loss_weights
            loss = weighted_loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test(
    dataloader,
    model,
    unreduced_loss_function,
    scheduler
):
    model.eval()
    total_loss = 0.0

    if len(dataloader) == 0:
        raise ValueError("Validation dataloader is empty.")

    with torch.no_grad():
        for x, y, loss_weights in dataloader:
            x = x.to(device)
            y = y.to(device)
            loss_weights = loss_weights.to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                prediction = model(x)

                voxel_loss = unreduced_loss_function(prediction, y)
                weighted_loss = voxel_loss * loss_weights
                loss = weighted_loss.mean()

            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    scheduler.step(average_loss)

    return average_loss