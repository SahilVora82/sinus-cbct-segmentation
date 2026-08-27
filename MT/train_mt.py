from pathlib import Path
import csv
import time

import numpy as np
import nrrd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet import UNet


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_SCAN_DIR = BASE_DIR / "data" / "scan"
TRAIN_MASK_DIR = BASE_DIR / "data" / "segment"

VAL_SCAN_DIR = BASE_DIR / "data" / "scan_valid"
VAL_MASK_DIR = BASE_DIR / "data" / "segment_valid"

# IMPORTANT:
# We DO NOT load data/scan_test or data/segment_test anywhere
# in this script. The test set stays untouched.

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_mt_model.pt"
HISTORY_PATH = RESULTS_DIR / "mt_training_history.csv"

# Computed ONLY from the 48 training scans
TRAIN_MEAN = -46.1730322190273
TRAIN_STD = 293.1394271328278

NUM_CLASSES = 3

# Label mapping:
# 0 = background
# 1 = mucosal thickening
# 2 = sinus air

EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.001


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple MPS"

else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

print("\n============================================================")
print("MT SEGMENTATION TRAINING")
print("============================================================")
print(f"Device: {DEVICE} ({DEVICE_NAME})")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print()
print("Labels:")
print("  0 = background")
print("  1 = mucosal thickening (MT)")
print("  2 = sinus air")
print()
print(f"Training normalization mean: {TRAIN_MEAN}")
print(f"Training normalization std:  {TRAIN_STD}")
print("============================================================\n")


# ============================================================
# DATASET
# ============================================================

class SinusMTDataset(Dataset):

    def __init__(self, scan_dir, mask_dir):

        self.scan_dir = Path(scan_dir)
        self.mask_dir = Path(mask_dir)

        self.scan_files = sorted(self.scan_dir.glob("*.nrrd"))

        if len(self.scan_files) == 0:
            raise RuntimeError(
                f"No NRRD scans found in:\n{self.scan_dir}"
            )

        # Verify every scan has a mask
        for scan_path in self.scan_files:

            mask_path = (
                self.mask_dir /
                f"{scan_path.stem}_GT.nrrd"
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for {scan_path.name}:\n"
                    f"{mask_path}"
                )

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):

        scan_path = self.scan_files[index]

        mask_path = (
            self.mask_dir /
            f"{scan_path.stem}_GT.nrrd"
        )

        scan, _ = nrrd.read(str(scan_path))
        mask, _ = nrrd.read(str(mask_path))

        scan = scan.astype(np.float32)
        mask = mask.astype(np.int64)

        # ----------------------------------------------------
        # FIXED TRAINING-SET NORMALIZATION
        # ----------------------------------------------------
        #
        # We use the SAME normalization for train/validation.
        # We do NOT calculate separate statistics per scan.
        #

        scan = (
            scan - TRAIN_MEAN
        ) / TRAIN_STD

        # [D,H,W] -> [1,D,H,W]
        scan = torch.from_numpy(scan).float().unsqueeze(0)

        # CrossEntropy expects integer class IDs:
        # [D,H,W], NOT one-hot encoded masks
        mask = torch.from_numpy(mask).long()

        return scan, mask


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = SinusMTDataset(
    TRAIN_SCAN_DIR,
    TRAIN_MASK_DIR
)

val_dataset = SinusMTDataset(
    VAL_SCAN_DIR,
    VAL_MASK_DIR
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print("Test samples used during training: 0\n")


# ============================================================
# TRAINING CLASS DISTRIBUTION
# ============================================================

def calculate_class_weights(mask_directory):

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    mask_files = sorted(
        Path(mask_directory).glob("*_GT.nrrd")
    )

    print("Calculating class frequencies from TRAINING masks only...")

    for path in mask_files:

        mask, _ = nrrd.read(str(path))
        mask = mask.astype(np.int64)

        current = np.bincount(
            mask.reshape(-1),
            minlength=NUM_CLASSES
        )

        counts += current[:NUM_CLASSES]

    total = counts.sum()
    frequencies = counts / total

    # Inverse square-root weighting.
    #
    # MT is much smaller than air/background.
    # Straight inverse frequency would give MT an enormous
    # weight, so sqrt gives us a more stable compromise.

    weights = 1.0 / np.sqrt(frequencies + 1e-8)

    # Normalize so average weight is approximately 1
    weights = weights / weights.mean()

    print("\nTRAIN CLASS DISTRIBUTION")
    print("--------------------------------")

    names = [
        "Background",
        "MT",
        "Air"
    ]

    for i in range(NUM_CLASSES):

        print(
            f"{names[i]:10s}: "
            f"{int(counts[i]):12,d} voxels | "
            f"{frequencies[i] * 100:7.3f}% | "
            f"weight {weights[i]:.4f}"
        )

    print()

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


class_weights = calculate_class_weights(
    TRAIN_MASK_DIR
).to(DEVICE)


# ============================================================
# MODEL
# ============================================================

# Uses the same existing MT project's 3D U-Net implementation.
model = UNet(
    [16, 32, 64, 96]
).to(DEVICE)


# ============================================================
# LOSSES
# ============================================================

cross_entropy = nn.CrossEntropyLoss(
    weight=class_weights
)


def foreground_dice_loss(logits, target):
    """
    Soft Dice loss for:
        class 1 = MT
        class 2 = sinus air

    Background is deliberately excluded from Dice loss
    because it occupies most of the volume.
    """

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    target_one_hot = torch.nn.functional.one_hot(
        target,
        num_classes=NUM_CLASSES
    )

    # [B,D,H,W,C] -> [B,C,D,H,W]
    target_one_hot = target_one_hot.permute(
        0, 4, 1, 2, 3
    ).float()

    dice_losses = []

    for class_id in [1, 2]:

        pred_class = probabilities[:, class_id]
        target_class = target_one_hot[:, class_id]

        intersection = torch.sum(
            pred_class * target_class
        )

        denominator = (
            torch.sum(pred_class)
            + torch.sum(target_class)
        )

        dice = (
            (2.0 * intersection + 1e-6)
            /
            (denominator + 1e-6)
        )

        dice_losses.append(
            1.0 - dice
        )

    return torch.stack(
        dice_losses
    ).mean()


def combined_loss(logits, target):

    ce = cross_entropy(
        logits,
        target
    )

    dice = foreground_dice_loss(
        logits,
        target
    )

    # Balanced combination
    total = (
        0.5 * ce
        +
        0.5 * dice
    )

    return total, ce, dice


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, target, class_id):

    pred = prediction == class_id
    true = target == class_id

    tp = torch.sum(
        pred & true
    ).item()

    fp = torch.sum(
        pred & (~true)
    ).item()

    fn = torch.sum(
        (~pred) & true
    ).item()

    denominator_dice = (
        2 * tp + fp + fn
    )

    denominator_iou = (
        tp + fp + fn
    )

    dice = (
        2 * tp / denominator_dice
        if denominator_dice > 0
        else 1.0
    )

    iou = (
        tp / denominator_iou
        if denominator_iou > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 1.0
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }


# ============================================================
# MODEL SHAPE CHECK
# ============================================================

print("Checking model input/output dimensions...")

model.eval()

with torch.no_grad():

    sample_scan, sample_mask = train_dataset[0]

    sample_scan = (
        sample_scan
        .unsqueeze(0)
        .to(DEVICE)
    )

    sample_output = model(
        sample_scan
    )

print(
    "Input shape: ",
    tuple(sample_scan.shape)
)

print(
    "Output shape:",
    tuple(sample_output.shape)
)

print(
    "Mask shape:  ",
    tuple(sample_mask.shape)
)


if sample_output.ndim != 5:
    raise RuntimeError(
        "U-Net output should have shape "
        "[B, C, D, H, W]."
    )

if sample_output.shape[1] != NUM_CLASSES:
    raise RuntimeError(
        "\nYour U-Net is not currently outputting "
        "3 channels.\n"
        f"Current output shape: {tuple(sample_output.shape)}\n"
        "We need exactly:\n"
        "[batch, 3, depth, height, width]"
    )


print("\n3-class model output: PASS\n")


# ============================================================
# CSV HISTORY
# ============================================================

history_fields = [
    "epoch",
    "learning_rate",
    "train_loss",
    "val_loss",
    "mt_dice",
    "mt_iou",
    "mt_precision",
    "mt_recall",
    "air_dice",
    "air_iou",
    "air_precision",
    "air_recall",
    "foreground_mean_dice",
    "seconds"
]


with open(
    HISTORY_PATH,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history_fields
    )

    writer.writeheader()


# ============================================================
# TRAINING
# ============================================================

best_score = -1.0
best_epoch = -1

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(
    "\nIMPORTANT: test data are NOT being evaluated during training."
)

print(
    "Best checkpoint will be selected using VALIDATION performance only.\n"
)


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    training_loss_total = 0.0

    for batch_number, (scans, masks) in enumerate(
        train_loader,
        start=1
    ):

        scans = scans.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            scans
        )

        loss, ce_loss, dice_loss = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        training_loss_total += loss.item()

        print(
            f"\rEpoch {epoch:02d}/{EPOCHS} | "
            f"Training batch "
            f"{batch_number:02d}/{len(train_loader)} | "
            f"Loss {loss.item():.4f}",
            end="",
            flush=True
        )

    train_loss = (
        training_loss_total
        /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0

    mt_metrics_all = []
    air_metrics_all = []

    with torch.no_grad():

        for scans, masks in val_loader:

            scans = scans.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(
                scans
            )

            loss, _, _ = combined_loss(
                logits,
                masks
            )

            val_loss_total += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            mt_metrics_all.append(
                calculate_metrics(
                    predictions,
                    masks,
                    class_id=1
                )
            )

            air_metrics_all.append(
                calculate_metrics(
                    predictions,
                    masks,
                    class_id=2
                )
            )


    val_loss = (
        val_loss_total
        /
        len(val_loader)
    )


    def average_metric(metric_list, name):

        return float(
            np.mean(
                [
                    entry[name]
                    for entry in metric_list
                ]
            )
        )


    mt_dice = average_metric(
        mt_metrics_all,
        "dice"
    )

    mt_iou = average_metric(
        mt_metrics_all,
        "iou"
    )

    mt_precision = average_metric(
        mt_metrics_all,
        "precision"
    )

    mt_recall = average_metric(
        mt_metrics_all,
        "recall"
    )


    air_dice = average_metric(
        air_metrics_all,
        "dice"
    )

    air_iou = average_metric(
        air_metrics_all,
        "iou"
    )

    air_precision = average_metric(
        air_metrics_all,
        "precision"
    )

    air_recall = average_metric(
        air_metrics_all,
        "recall"
    )


    # Model-selection score:
    #
    # Mean of MT Dice and air Dice.
    #
    # Background Dice is intentionally excluded because
    # background dominates the volume.

    foreground_mean_dice = (
        mt_dice + air_dice
    ) / 2.0


    scheduler.step(
        foreground_mean_dice
    )


    elapsed = (
        time.time()
        - epoch_start
    )

    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")

    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss:   {val_loss:.4f}"
    )

    print()

    print("MT:")
    print(
        f"  Dice      = {mt_dice:.4f}"
    )
    print(
        f"  IoU       = {mt_iou:.4f}"
    )
    print(
        f"  Precision = {mt_precision:.4f}"
    )
    print(
        f"  Recall    = {mt_recall:.4f}"
    )

    print()

    print("Sinus Air:")
    print(
        f"  Dice      = {air_dice:.4f}"
    )
    print(
        f"  IoU       = {air_iou:.4f}"
    )
    print(
        f"  Precision = {air_precision:.4f}"
    )
    print(
        f"  Recall    = {air_recall:.4f}"
    )

    print()

    print(
        f"Mean foreground Dice = "
        f"{foreground_mean_dice:.4f}"
    )

    print(
        f"LR = {current_lr:.6f}"
    )

    print(
        f"Epoch time = {elapsed:.1f} sec"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if foreground_mean_dice > best_score:

        best_score = foreground_mean_dice
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "validation_foreground_mean_dice":
                    foreground_mean_dice,

                "validation_mt_dice":
                    mt_dice,

                "validation_mt_iou":
                    mt_iou,

                "validation_mt_precision":
                    mt_precision,

                "validation_mt_recall":
                    mt_recall,

                "validation_air_dice":
                    air_dice,

                "train_mean":
                    TRAIN_MEAN,

                "train_std":
                    TRAIN_STD,

                "labels": {
                    0: "background",
                    1: "mucosal_thickening",
                    2: "sinus_air"
                }
            },

            BEST_MODEL_PATH
        )

        print()
        print(
            "***** NEW BEST VALIDATION MODEL SAVED *****"
        )

        print(
            f"Checkpoint: {BEST_MODEL_PATH}"
        )


    # ========================================================
    # WRITE HISTORY
    # ========================================================

    with open(
        HISTORY_PATH,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history_fields
        )

        writer.writerow(
            {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,

                "mt_dice": mt_dice,
                "mt_iou": mt_iou,
                "mt_precision": mt_precision,
                "mt_recall": mt_recall,

                "air_dice": air_dice,
                "air_iou": air_iou,
                "air_precision": air_precision,
                "air_recall": air_recall,

                "foreground_mean_dice":
                    foreground_mean_dice,

                "seconds": elapsed
            }
        )


    print(
        f"\nBest so far: epoch {best_epoch} | "
        f"foreground Dice {best_score:.4f}"
    )

    print("=" * 70)


# ============================================================
# FINISHED
# ============================================================

print("\nTRAINING FINISHED")
print("=" * 70)

print(
    f"Best validation epoch: {best_epoch}"
)

print(
    f"Best validation foreground Dice: "
    f"{best_score:.4f}"
)

print(
    f"Best model:\n{BEST_MODEL_PATH}"
)

print(
    f"Training history:\n{HISTORY_PATH}"
)

print()
print(
    "TEST SET HAS NOT BEEN USED."
)

print(
    "Do NOT evaluate the test set until the model "
    "and training procedure are locked."
)