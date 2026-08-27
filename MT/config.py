from pathlib import Path

import torch

from unet import UNet


# ---------------------------------------------------------
# Base directory
# ---------------------------------------------------------

# Always points to the MT folder containing this config.py.
# This prevents paths from accidentally pointing into Healthy.
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# Device selection
# ---------------------------------------------------------

# SeaWulf will normally use CUDA.
# Your MacBook will normally use MPS.
if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ---------------------------------------------------------
# Original source data
# ---------------------------------------------------------

source_data_directory = (
    BASE_DIR / "MT_input_use_allSeg_padded_updated"
)


# ---------------------------------------------------------
# Training data
# ---------------------------------------------------------

train_scans_directory = (
    BASE_DIR / "data" / "scan"
)

train_segments_directory = (
    BASE_DIR / "data" / "segment"
)


# ---------------------------------------------------------
# Validation data
# ---------------------------------------------------------

valid_scans_directory = (
    BASE_DIR / "data" / "scan_valid"
)

valid_segments_directory = (
    BASE_DIR / "data" / "segment_valid"
)


# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------

test_scans_directory = (
    BASE_DIR / "data" / "scan_test"
)

test_segments_directory = (
    BASE_DIR / "data" / "segment_test"
)


# Ground-truth files are named like:
# FileB1_MT_L_GT.nrrd
segments_suffix = "_GT"


# ---------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------

checkpoint_directory = (
    BASE_DIR / "checkpoints"
)

checkpoint_directory.mkdir(
    parents=True,
    exist_ok=True,
)

# None means begin training from the start.
checkpoint_file = None

# Save every five epochs to avoid creating too many large files.
checkpoint_interval = 5


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

batch_size = 1

base_learning_rate = 0.004

# Reduce the learning rate when validation loss stops improving.
halving_patience = 10

epochs = 50


# ---------------------------------------------------------
# 3D U-Net model
# ---------------------------------------------------------

# The current unet.py produces three output channels,
# corresponding to the three mask values: 0, 1, and 2.
#
# We still need to confirm visually which of labels 1 and 2
# represents air and which represents mucosal thickening.
model = UNet(
    [48, 64, 128, 196]
).to(device)


# ---------------------------------------------------------
# Inference data
# ---------------------------------------------------------

inference_scans_directory = (
    BASE_DIR / "data" / "inference"
)

inference_output_directory = (
    BASE_DIR / "data" / "inference_output"
)

inference_output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

# Predictions should be marked as predictions, not ground truth.
inference_segments_suffix = "_PRED"


# ---------------------------------------------------------
# Inference checkpoint
# ---------------------------------------------------------

inference_checkpoint_directory = (
    BASE_DIR / "checkpoints"
)

# Change this after training, for example:
# inference_checkpoint_file = "model_50.pth"
inference_checkpoint_file = None