from pathlib import Path

import torch

from dataset import EdgeBoostConfig
from unet import UNet


# ---------------------------------------------------------
# Base project directory
# ---------------------------------------------------------

# This always points to the folder containing this config.py file.
# Therefore, all paths below will remain inside the Healthy folder,
# regardless of where the program is launched from.
BASE_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------
# Device selection
# ---------------------------------------------------------

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


# Ground-truth segmentation filename suffix.
segments_suffix = "_GT"


# ---------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------

checkpoint_directory = (
    BASE_DIR
    / "checkpoints_fixed_training_normalization"
)

# Create the checkpoint folder if it does not exist.
checkpoint_directory.mkdir(
    parents=True,
    exist_ok=True
)

# Start training from the beginning.
checkpoint_file = None

# Save a checkpoint every five epochs.
checkpoint_interval = 5


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

batch_size = 1
epochs = 50


# ---------------------------------------------------------
# Fixed training-set normalization
# ---------------------------------------------------------

# These values were calculated using only the 29 training scans.
training_mean = -34.69477476229485
training_std = 244.6038824012994


# ---------------------------------------------------------
# 3D U-Net model
# ---------------------------------------------------------

# Binary healthy-sinus segmentation:
# 0 = background
# 1 = healthy maxillary sinus air
model = UNet(
    [32, 64, 128, 256]
).to(device)


# ---------------------------------------------------------
# Boundary-weighted loss settings
# ---------------------------------------------------------

# Increase the importance of errors near segmentation boundaries.
edge_boost_config = EdgeBoostConfig(
    region_size=5,
    boost_factor=4
)


# ---------------------------------------------------------
# Inference folders
# ---------------------------------------------------------

inference_scans_directory = (
    BASE_DIR / "data" / "inference"
)

inference_output_directory = (
    BASE_DIR / "data" / "inference_output"
)

# Create the output folder if it does not exist.
inference_output_directory.mkdir(
    parents=True,
    exist_ok=True
)

inference_segments_suffix = "_PRED"


# ---------------------------------------------------------
# Inference checkpoint
# ---------------------------------------------------------

inference_checkpoint_directory = (
    BASE_DIR
    / "checkpoints_fixed_training_normalization"
)

inference_checkpoint_file = "model_50.pth"