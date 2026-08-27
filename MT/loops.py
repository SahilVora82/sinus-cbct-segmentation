import torch
from config import device

def train(dataloader, model, loss_function, optimizer):
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        loss = loss_function(prediction, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

def test(dataloader, model, loss_function, scheduler):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            weighted_loss = loss_function(prediction, y)
            total_loss += weighted_loss.mean().item()
    scheduler.step(total_loss / len(dataloader)) # "Useless" division for correct thresholding
    print(f"Average loss: {total_loss / len(dataloader)}")