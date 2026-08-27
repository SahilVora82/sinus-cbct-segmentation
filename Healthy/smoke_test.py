import torch
import torch.nn as nn

from dataset import normalize, ScanDataset
from config import (
    device,
    model,
    train_scans_directory,
    train_segments_directory,
    segments_suffix,
    edge_boost_config,
)


print("Creating dataset...")

dataset = ScanDataset(
    train_scans_directory,
    train_segments_directory,
    segments_suffix,
    edge_boost_config,
    transformations=[normalize()],
)

print(f"Training samples found: {len(dataset)}")

x, y, loss_weights = dataset[0]

x = x.unsqueeze(0).to(device)
y = y.unsqueeze(0).to(device)
loss_weights = loss_weights.unsqueeze(0).to(device)

print("Input shape:", tuple(x.shape))
print("Target shape:", tuple(y.shape))
print("Weight shape:", tuple(loss_weights.shape))
print("Target values:", torch.unique(y).tolist())

model.train()
model.zero_grad(set_to_none=True)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

prediction = model(x)

loss_function = nn.BCEWithLogitsLoss(reduction="none")
loss = (
    loss_function(prediction, y) * loss_weights
).mean()

print("Prediction shape:", tuple(prediction.shape))
print("Loss:", loss.item())
print("Loss is finite:", torch.isfinite(loss).item())

loss.backward()

gradients = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.grad is not None
]

backward_worked = (
    len(gradients) > 0
    and all(torch.isfinite(gradient).all().item() for gradient in gradients)
)

print("Backward pass worked:", backward_worked)

if device.type == "cuda":
    memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Maximum GPU memory used: {memory_gb:.2f} GB")

print("SMOKE TEST COMPLETE")