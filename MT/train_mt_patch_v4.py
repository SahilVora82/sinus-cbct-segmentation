from pathlib import Path
import random
import time
import csv

import numpy as np
import nrrd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import binary_dilation

from unet import UNet


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_SCAN_DIR = BASE_DIR / "data" / "scan"
TRAIN_MASK_DIR = BASE_DIR / "data" / "segment"

VAL_SCAN_DIR = BASE_DIR / "data" / "scan_valid"
VAL_MASK_DIR = BASE_DIR / "data" / "segment_valid"

SOURCE_CHECKPOINT = (
    BASE_DIR
    / "results_hybrid"
    / "best_mt_model.pt"
)

OUT_DIR = BASE_DIR / "results_patch_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = [32, 64, 128, 256]

BG = 0
MT = 1
AIR = 2

NORM_MEAN = -46.1730322190273
NORM_STD = 293.1394271328278

SEED = 42

PATCH_SIZE = 64
PATCHES_PER_CASE = 2

# 75% of training samples are centered on MT.
POSITIVE_PATCH_PROB = 0.75

# Hard-negative air is sampled near the MT boundary.
HARD_NEGATIVE_DILATION = 8

BATCH_SIZE = 1

MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 5

# Preserve pretrained features.
BACKBONE_LR = 5e-6

# Let the new binary head learn much faster.
HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-5

# Dice + focal
DICE_WEIGHT = 0.60
FOCAL_WEIGHT = 0.40

FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

MT_THRESHOLD = 0.50

# Frozen hybrid model provides a VERY permissive sinus ROI.
# We intentionally do NOT use argmax because old error analysis showed
# many GT-MT voxels were being pushed toward background.
ROI_THRESHOLD = 0.05

BASELINE_MT_DICE = 0.6868


# =============================================================================
# REPRODUCIBILITY + DEVICE
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint_state(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model" in checkpoint:
            return checkpoint["model"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        f"Could not find model state in {path}"
    )


# =============================================================================
# HELPERS
# =============================================================================

def load_scan_and_mask(
    scan_path,
    mask_path,
):

    scan, _ = nrrd.read(
        str(scan_path)
    )

    mask, _ = nrrd.read(
        str(mask_path)
    )

    scan = scan.astype(
        np.float32
    )

    mask = mask.astype(
        np.uint8
    )

    if scan.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{scan.shape} vs {mask.shape}"
        )

    scan = (
        scan - NORM_MEAN
    ) / NORM_STD

    return scan, mask


def crop_around_center(
    array,
    center,
    patch_size,
):

    starts = []

    for axis in range(3):

        dimension = array.shape[axis]

        start = (
            int(center[axis])
            - patch_size // 2
        )

        start = max(
            0,
            min(
                start,
                dimension - patch_size,
            ),
        )

        starts.append(start)

    z, y, x = starts

    return array[
        z:z + patch_size,
        y:y + patch_size,
        x:x + patch_size,
    ]


def binary_stats(
    prediction,
    target,
):

    prediction = prediction.bool()
    target = target.bool()

    tp = (
        prediction
        & target
    ).sum().item()

    fp = (
        prediction
        & (~target)
    ).sum().item()

    fn = (
        (~prediction)
        & target
    ).sum().item()

    pred_sum = (
        prediction
        .sum()
        .item()
    )

    target_sum = (
        target
        .sum()
        .item()
    )

    dice_denominator = (
        pred_sum
        + target_sum
    )

    dice = (
        2.0 * tp
        / dice_denominator
        if dice_denominator > 0
        else 1.0
    )

    union = (
        tp + fp + fn
    )

    iou = (
        tp / union
        if union > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return (
        float(dice),
        float(iou),
        float(precision),
        float(recall),
    )


# =============================================================================
# PATCH TRAINING DATASET
# =============================================================================

class MTPatchDataset(Dataset):

    def __init__(
        self,
        scan_dir,
        mask_dir,
    ):

        self.scan_dir = Path(scan_dir)
        self.mask_dir = Path(mask_dir)

        self.ids = sorted(
            path.stem
            for path
            in self.scan_dir.glob("*.nrrd")
        )

        if not self.ids:
            raise RuntimeError(
                f"No scans in {self.scan_dir}"
            )

        self.cases = []

        print(
            "Preloading training data..."
        )

        for case_id in self.ids:

            scan_path = (
                self.scan_dir
                / f"{case_id}.nrrd"
            )

            mask_path = (
                self.mask_dir
                / f"{case_id}_GT.nrrd"
            )

            scan, mask = (
                load_scan_and_mask(
                    scan_path,
                    mask_path,
                )
            )

            mt_mask = (
                mask == MT
            )

            air_mask = (
                mask == AIR
            )

            mt_coordinates = (
                np.argwhere(
                    mt_mask
                )
            )

            if len(mt_coordinates) == 0:
                raise RuntimeError(
                    f"{case_id} has no MT voxels."
                )

            # Air immediately surrounding / near MT.
            dilated_mt = binary_dilation(
                mt_mask,
                iterations=(
                    HARD_NEGATIVE_DILATION
                ),
            )

            hard_negative_mask = (
                dilated_mt
                & air_mask
            )

            hard_negative_coordinates = (
                np.argwhere(
                    hard_negative_mask
                )
            )

            # Fallback if no near-lesion air.
            if (
                len(
                    hard_negative_coordinates
                )
                == 0
            ):
                hard_negative_coordinates = (
                    np.argwhere(
                        air_mask
                    )
                )

            self.cases.append(
                {
                    "id":
                        case_id,

                    "scan":
                        scan,

                    "mask":
                        mask,

                    "mt_coordinates":
                        mt_coordinates,

                    "hard_negative_coordinates":
                        hard_negative_coordinates,
                }
            )

        print(
            f"Loaded {len(self.cases)} "
            f"training volumes."
        )

    def __len__(self):

        return (
            len(self.cases)
            * PATCHES_PER_CASE
        )

    def __getitem__(
        self,
        index,
    ):

        case = self.cases[
            index
            % len(self.cases)
        ]

        use_positive = (
            random.random()
            < POSITIVE_PATCH_PROB
        )

        if use_positive:

            coordinates = (
                case[
                    "mt_coordinates"
                ]
            )

        else:

            coordinates = (
                case[
                    "hard_negative_coordinates"
                ]
            )

        random_index = (
            random.randrange(
                len(coordinates)
            )
        )

        center = coordinates[
            random_index
        ]

        scan_patch = crop_around_center(
            case["scan"],
            center,
            PATCH_SIZE,
        )

        mask_patch = crop_around_center(
            case["mask"],
            center,
            PATCH_SIZE,
        )

        # Random spatial flips.
        for axis in range(3):

            if random.random() < 0.5:

                scan_patch = np.flip(
                    scan_patch,
                    axis=axis,
                )

                mask_patch = np.flip(
                    mask_patch,
                    axis=axis,
                )

        # np.flip produces negative strides.
        scan_patch = (
            np.ascontiguousarray(
                scan_patch
            )
        )

        mask_patch = (
            np.ascontiguousarray(
                mask_patch
            )
        )

        scan_tensor = (
            torch
            .from_numpy(
                scan_patch
            )
            .unsqueeze(0)
            .float()
        )

        mask_tensor = (
            torch
            .from_numpy(
                mask_patch.astype(
                    np.int64
                )
            )
            .long()
        )

        return (
            scan_tensor,
            mask_tensor,
            case["id"],
            int(use_positive),
        )


# =============================================================================
# FULL VALIDATION DATASET
# =============================================================================

class FullVolumeDataset(Dataset):

    def __init__(
        self,
        scan_dir,
        mask_dir,
    ):

        self.scan_dir = Path(scan_dir)
        self.mask_dir = Path(mask_dir)

        self.ids = sorted(
            path.stem
            for path
            in self.scan_dir.glob("*.nrrd")
        )

        self.cases = []

        print(
            "Preloading validation data..."
        )

        for case_id in self.ids:

            scan, mask = (
                load_scan_and_mask(
                    self.scan_dir
                    / f"{case_id}.nrrd",

                    self.mask_dir
                    / f"{case_id}_GT.nrrd",
                )
            )

            self.cases.append(
                (
                    case_id,
                    scan,
                    mask,
                )
            )

        print(
            f"Loaded {len(self.cases)} "
            f"validation volumes."
        )

    def __len__(self):
        return len(self.cases)

    def __getitem__(
        self,
        index,
    ):

        case_id, scan, mask = (
            self.cases[index]
        )

        scan_tensor = (
            torch
            .from_numpy(scan)
            .unsqueeze(0)
            .float()
        )

        mask_tensor = (
            torch
            .from_numpy(
                mask.astype(
                    np.int64
                )
            )
            .long()
        )

        return (
            scan_tensor,
            mask_tensor,
            case_id,
        )


# =============================================================================
# STAGE 1: FROZEN SINUS GATE
# =============================================================================

def build_source_model():

    model = UNet(
        CHANNELS
    )

    state = load_checkpoint_state(
        SOURCE_CHECKPOINT
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    return model


# =============================================================================
# STAGE 2: BINARY MT MODEL
# =============================================================================

class BinaryMTUNet(nn.Module):

    def __init__(
        self,
        checkpoint_path,
    ):

        super().__init__()

        base = UNet(
            CHANNELS
        )

        source_state = (
            load_checkpoint_state(
                checkpoint_path
            )
        )

        # First load the exact trained architecture.
        base.load_state_dict(
            source_state,
            strict=True,
        )

        old_head = (
            base.conv_out
        )

        if not isinstance(
            old_head,
            nn.Conv3d,
        ):
            raise TypeError(
                "Expected conv_out "
                "to be Conv3d."
            )

        if (
            old_head.out_channels
            != 3
        ):
            raise RuntimeError(
                "Source checkpoint must "
                "have 3 classes."
            )

        new_head = nn.Conv3d(
            old_head.in_channels,
            1,
            kernel_size=(
                old_head.kernel_size
            ),
            stride=(
                old_head.stride
            ),
            padding=(
                old_head.padding
            ),
            dilation=(
                old_head.dilation
            ),
            groups=(
                old_head.groups
            ),
            bias=(
                old_head.bias
                is not None
            ),
        )

        # Initialize binary MT-vs-rest logit
        # directly from the learned 3-class head.
        with torch.no_grad():

            w = (
                old_head
                .weight
                .detach()
                .clone()
            )

            if (
                old_head.bias
                is not None
            ):

                b = (
                    old_head
                    .bias
                    .detach()
                    .clone()
                )

            else:

                b = torch.zeros(
                    3,
                    dtype=w.dtype,
                )

            non_mt_weight = (
                0.5
                * (
                    w[BG]
                    + w[AIR]
                )
            )

            binary_weight = (
                w[MT]
                - non_mt_weight
            )

            new_head.weight.copy_(
                binary_weight.unsqueeze(0)
            )

            if (
                new_head.bias
                is not None
            ):

                non_mt_bias = (
                    0.5
                    * (
                        b[BG]
                        + b[AIR]
                    )
                )

                binary_bias = (
                    b[MT]
                    - non_mt_bias
                )

                new_head.bias.copy_(
                    binary_bias.view(1)
                )

        base.conv_out = new_head

        self.backbone = base

    @property
    def head(self):
        return self.backbone.conv_out

    def forward(
        self,
        x,
    ):
        return self.backbone(x)


# =============================================================================
# LOSS: MASKED DICE + MASKED FOCAL
# =============================================================================

def masked_dice_loss(
    logits,
    target,
    sinus_mask,
    eps=1e-5,
):

    probability = torch.sigmoid(
        logits
    )

    target = target.float()
    sinus_mask = sinus_mask.float()

    probability = (
        probability
        * sinus_mask
    )

    target = (
        target
        * sinus_mask
    )

    dimensions = tuple(
        range(
            1,
            probability.ndim,
        )
    )

    intersection = torch.sum(
        probability
        * target,
        dim=dimensions,
    )

    denominator = (
        torch.sum(
            probability,
            dim=dimensions,
        )
        +
        torch.sum(
            target,
            dim=dimensions,
        )
    )

    dice = (
        2.0
        * intersection
        + eps
    ) / (
        denominator
        + eps
    )

    return (
        1.0
        - dice.mean()
    )


def masked_focal_loss(
    logits,
    target,
    sinus_mask,
):

    valid = (
        sinus_mask
        > 0.5
    )

    if not torch.any(valid):

        return (
            logits.sum()
            * 0.0
        )

    selected_logits = (
        logits[valid]
    )

    selected_target = (
        target[valid]
        .float()
    )

    bce = (
        F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_target,
            reduction="none",
        )
    )

    probability = torch.sigmoid(
        selected_logits
    )

    p_t = torch.where(
        selected_target > 0.5,
        probability,
        1.0 - probability,
    )

    alpha_t = torch.where(
        selected_target > 0.5,
        torch.full_like(
            selected_target,
            FOCAL_ALPHA,
        ),
        torch.full_like(
            selected_target,
            1.0 - FOCAL_ALPHA,
        ),
    )

    focal = (
        alpha_t
        * (
            1.0 - p_t
        ).pow(
            FOCAL_GAMMA
        )
        * bce
    )

    return focal.mean()


def compute_loss(
    logits,
    target,
):

    mt_target = (
        target == MT
    ).unsqueeze(1)

    # IMPORTANT:
    # Stage 2 learns ONLY inside
    # the GT sinus (MT + AIR).
    sinus_mask = (
        target != BG
    ).unsqueeze(1)

    dice = masked_dice_loss(
        logits,
        mt_target,
        sinus_mask,
    )

    focal = masked_focal_loss(
        logits,
        mt_target,
        sinus_mask,
    )

    total = (
        DICE_WEIGHT
        * dice
        +
        FOCAL_WEIGHT
        * focal
    )

    return (
        total,
        dice,
        focal,
    )


# =============================================================================
# DATA
# =============================================================================

train_dataset = MTPatchDataset(
    TRAIN_SCAN_DIR,
    TRAIN_MASK_DIR,
)

validation_dataset = FullVolumeDataset(
    VAL_SCAN_DIR,
    VAL_MASK_DIR,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        device.type == "cuda"
    ),
)


# =============================================================================
# PRECOMPUTE FROZEN STAGE-1 ROI
# =============================================================================

print()
print("=" * 78)
print("BUILDING FROZEN SINUS ROI CACHE")
print("=" * 78)

roi_model = (
    build_source_model()
    .to(device)
)

roi_model.eval()

for parameter in (
    roi_model.parameters()
):
    parameter.requires_grad = False


validation_roi_cache = {}

roi_mt_recalls = []

with torch.no_grad():

    for index in range(
        len(validation_dataset)
    ):

        (
            scan,
            target,
            case_id,
        ) = validation_dataset[
            index
        ]

        scan_gpu = (
            scan
            .unsqueeze(0)
            .to(
                device,
                dtype=torch.float32,
            )
        )

        logits = roi_model(
            scan_gpu
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

        # Sinus probability =
        # P(MT) + P(AIR)
        sinus_probability = (
            probabilities[:, MT]
            +
            probabilities[:, AIR]
        )

        roi = (
            sinus_probability
            > ROI_THRESHOLD
        ).squeeze(0)

        validation_roi_cache[
            case_id
        ] = roi.cpu()

        gt_mt = (
            target == MT
        )

        gt_mt_count = (
            gt_mt
            .sum()
            .item()
        )

        if gt_mt_count > 0:

            roi_recall = (
                (
                    roi.cpu()
                    & gt_mt
                )
                .sum()
                .item()
                / gt_mt_count
            )

        else:
            roi_recall = 1.0

        roi_mt_recalls.append(
            roi_recall
        )

        print(
            f"{case_id:<20} "
            f"GT-MT inside ROI="
            f"{roi_recall:.4f}"
        )


print()
print(
    "Mean GT-MT ROI recall:",
    f"{np.mean(roi_mt_recalls):.4f}",
)

# Free the frozen source network.
del roi_model

if device.type == "cuda":
    torch.cuda.empty_cache()


# =============================================================================
# BINARY MODEL
# =============================================================================

model = BinaryMTUNet(
    SOURCE_CHECKPOINT
).to(device)


# =============================================================================
# SHAPE CHECK
# =============================================================================

sample_scan, _, _, _ = (
    train_dataset[0]
)

with torch.no_grad():

    output = model(
        sample_scan
        .unsqueeze(0)
        .to(device)
    )

print()
print("=" * 78)
print("PATCH V4 MODEL SHAPE CHECK")
print("=" * 78)

print(
    "Input:",
    tuple(
        sample_scan
        .unsqueeze(0)
        .shape
    ),
)

print(
    "Output:",
    tuple(
        output.shape
    ),
)

expected_shape = (
    1,
    1,
    PATCH_SIZE,
    PATCH_SIZE,
    PATCH_SIZE,
)

if (
    tuple(output.shape)
    != expected_shape
):
    raise RuntimeError(
        f"Unexpected output shape: "
        f"{output.shape}"
    )

print(
    "BINARY PATCH OUTPUT CHECK: PASS"
)


# =============================================================================
# OPTIMIZER
# =============================================================================

head_parameter_ids = {
    id(parameter)
    for parameter
    in model.head.parameters()
}

backbone_parameters = [
    parameter
    for parameter
    in model.parameters()
    if id(parameter)
    not in head_parameter_ids
]

head_parameters = list(
    model.head.parameters()
)

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                backbone_parameters,
            "lr":
                BACKBONE_LR,
        },
        {
            "params":
                head_parameters,
            "lr":
                HEAD_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model():

    model.eval()

    raw_rows = []
    cascade_rows = []

    with torch.no_grad():

        for index in range(
            len(validation_dataset)
        ):

            (
                scan,
                target,
                case_id,
            ) = validation_dataset[
                index
            ]

            scan_gpu = (
                scan
                .unsqueeze(0)
                .to(
                    device,
                    dtype=torch.float32,
                )
            )

            logits = model(
                scan_gpu
            )

            probability = (
                torch.sigmoid(
                    logits
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
            )

            raw_prediction = (
                probability
                > MT_THRESHOLD
            )

            roi = (
                validation_roi_cache[
                    case_id
                ]
            )

            cascade_prediction = (
                raw_prediction
                & roi
            )

            gt_mt = (
                target == MT
            )

            raw_rows.append(
                binary_stats(
                    raw_prediction,
                    gt_mt,
                )
            )

            cascade_rows.append(
                binary_stats(
                    cascade_prediction,
                    gt_mt,
                )
            )

    raw_rows = np.asarray(
        raw_rows,
        dtype=np.float64,
    )

    cascade_rows = np.asarray(
        cascade_rows,
        dtype=np.float64,
    )

    return {

        "raw_dice":
            float(
                raw_rows[:, 0].mean()
            ),

        "cascade_dice":
            float(
                cascade_rows[:, 0].mean()
            ),

        "cascade_iou":
            float(
                cascade_rows[:, 1].mean()
            ),

        "precision":
            float(
                cascade_rows[:, 2].mean()
            ),

        "recall":
            float(
                cascade_rows[:, 3].mean()
            ),
    }


# =============================================================================
# HISTORY
# =============================================================================

history_path = (
    OUT_DIR
    / "training_history.csv"
)

history_fields = [
    "epoch",
    "train_loss",
    "train_dice_loss",
    "train_focal_loss",
    "raw_mt_dice",
    "cascade_mt_dice",
    "cascade_iou",
    "precision",
    "recall",
    "backbone_lr",
    "head_lr",
]

with open(
    history_path,
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history_fields,
    )

    writer.writeheader()


# =============================================================================
# START INFO
# =============================================================================

print()
print("=" * 78)
print("PATCH-BASED BINARY MT TRAINING V4")
print("=" * 78)

print(
    f"Device: {device}"
)

print(
    f"Training volumes: "
    f"{len(train_dataset.cases)}"
)

print(
    f"Training patches/epoch: "
    f"{len(train_dataset)}"
)

print(
    f"Validation volumes: "
    f"{len(validation_dataset)}"
)

print(
    "Test volumes loaded: 0"
)

print(
    "TEST DATA LOADED: NO"
)

print()
print(
    f"Patch size: "
    f"{PATCH_SIZE}^3"
)

print(
    f"Positive patch probability: "
    f"{POSITIVE_PATCH_PROB}"
)

print(
    f"Backbone LR: "
    f"{BACKBONE_LR}"
)

print(
    f"Head LR: "
    f"{HEAD_LR}"
)

print(
    f"Loss: "
    f"{DICE_WEIGHT:.2f} Dice + "
    f"{FOCAL_WEIGHT:.2f} Focal"
)

print(
    f"ROI threshold: "
    f"{ROI_THRESHOLD}"
)

print(
    f"MT threshold: "
    f"{MT_THRESHOLD}"
)

print(
    f"Hybrid benchmark: "
    f"{BASELINE_MT_DICE:.4f}"
)

print("=" * 78)


# =============================================================================
# TRAIN
# =============================================================================

best_dice = -1.0
best_epoch = -1

epochs_without_improvement = 0

try:

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.time()

        model.train()

        running_total = 0.0
        running_dice = 0.0
        running_focal = 0.0

        positive_count = 0
        negative_count = 0

        for (
            batch_index,
            (
                scan,
                target,
                _,
                positive_flag,
            ),
        ) in enumerate(
            train_loader,
            start=1,
        ):

            scan = scan.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            target = target.to(
                device,
                dtype=torch.long,
                non_blocking=True,
            )

            positive_count += int(
                positive_flag
                .sum()
                .item()
            )

            negative_count += (
                len(positive_flag)
                -
                int(
                    positive_flag
                    .sum()
                    .item()
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                scan
            )

            (
                total_loss,
                dice_loss,
                focal_loss,
            ) = compute_loss(
                logits,
                target,
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            running_total += (
                total_loss.item()
            )

            running_dice += (
                dice_loss.item()
            )

            running_focal += (
                focal_loss.item()
            )

            if (
                batch_index % 24 == 0
                or
                batch_index
                == len(train_loader)
            ):

                print(
                    f"Epoch "
                    f"{epoch:02d}/"
                    f"{MAX_EPOCHS} | "
                    f"batch "
                    f"{batch_index:03d}/"
                    f"{len(train_loader)} | "
                    f"loss="
                    f"{total_loss.item():.4f}"
                )

        train_loss = (
            running_total
            / len(train_loader)
        )

        train_dice_loss = (
            running_dice
            / len(train_loader)
        )

        train_focal_loss = (
            running_focal
            / len(train_loader)
        )

        # ---------------------------------------------------------------------
        # FULL-VOLUME VALIDATION
        # ---------------------------------------------------------------------

        metrics = validate_model()

        backbone_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        head_lr = (
            optimizer
            .param_groups[1]["lr"]
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Epoch {epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        print(
            f"TRAIN loss="
            f"{train_loss:.4f}"
        )

        print(
            f"TRAIN DiceLoss="
            f"{train_dice_loss:.4f}"
        )

        print(
            f"TRAIN FocalLoss="
            f"{train_focal_loss:.4f}"
        )

        print(
            f"Patch sampling: "
            f"positive={positive_count}, "
            f"hard-negative={negative_count}"
        )

        print()
        print(
            "FULL-VOLUME VALIDATION"
        )

        print(
            f"Raw Stage-2 MT Dice: "
            f"{metrics['raw_dice']:.4f}"
        )

        print(
            f"Cascade MT Dice:     "
            f"{metrics['cascade_dice']:.4f}"
        )

        print(
            f"Cascade IoU:         "
            f"{metrics['cascade_iou']:.4f}"
        )

        print(
            f"Precision:           "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall:              "
            f"{metrics['recall']:.4f}"
        )

        print()
        print(
            f"Vs hybrid benchmark: "
            f"{metrics['cascade_dice'] - BASELINE_MT_DICE:+.4f}"
        )

        print(
            f"Epoch time: "
            f"{elapsed:.1f} sec"
        )

        # ---------------------------------------------------------------------
        # SAVE HISTORY
        # ---------------------------------------------------------------------

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_dice_loss":
                train_dice_loss,

            "train_focal_loss":
                train_focal_loss,

            "raw_mt_dice":
                metrics[
                    "raw_dice"
                ],

            "cascade_mt_dice":
                metrics[
                    "cascade_dice"
                ],

            "cascade_iou":
                metrics[
                    "cascade_iou"
                ],

            "precision":
                metrics[
                    "precision"
                ],

            "recall":
                metrics[
                    "recall"
                ],

            "backbone_lr":
                backbone_lr,

            "head_lr":
                head_lr,
        }

        with open(
            history_path,
            "a",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    history_fields
                ),
            )

            writer.writerow(
                row
            )

        checkpoint = {
            "epoch":
                epoch,

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "metrics":
                metrics,

            "channels":
                CHANNELS,

            "normalization_mean":
                NORM_MEAN,

            "normalization_std":
                NORM_STD,

            "patch_size":
                PATCH_SIZE,

            "roi_threshold":
                ROI_THRESHOLD,

            "mt_threshold":
                MT_THRESHOLD,

            "source_checkpoint":
                str(
                    SOURCE_CHECKPOINT
                ),

            "version":
                "patch_v4",
        }

        torch.save(
            checkpoint,
            OUT_DIR
            / "latest.pt",
        )

        # ---------------------------------------------------------------------
        # BEST
        # ---------------------------------------------------------------------

        current_dice = (
            metrics[
                "cascade_dice"
            ]
        )

        if current_dice > best_dice:

            best_dice = (
                current_dice
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                checkpoint,
                OUT_DIR
                / "best_mt_model.pt",
            )

            print()
            print(
                "*** NEW BEST PATCH V4 MODEL ***"
            )

            print(
                f"MT Dice = "
                f"{best_dice:.4f}"
            )

            if (
                best_dice
                > BASELINE_MT_DICE
            ):

                print(
                    "!!! BEATS HYBRID BASELINE !!!"
                )

        else:

            epochs_without_improvement += 1

        print()
        print(
            f"Best MT Dice: "
            f"{best_dice:.4f} "
            f"(epoch {best_epoch})"
        )

        print(
            "Epochs without improvement: "
            f"{epochs_without_improvement}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            print(
                "EARLY STOPPING"
            )

            break

        print(
            "=" * 78
        )


except KeyboardInterrupt:

    print()
    print(
        "Training interrupted."
    )


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 78)
print("PATCH V4 FINISHED")
print("=" * 78)

print(
    f"Best validation MT Dice: "
    f"{best_dice:.4f}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Hybrid benchmark: "
    f"{BASELINE_MT_DICE:.4f}"
)

print(
    f"Difference: "
    f"{best_dice - BASELINE_MT_DICE:+.4f}"
)

print(
    "Best checkpoint:",
    OUT_DIR
    / "best_mt_model.pt",
)

print(
    "TEST DATA WERE NEVER LOADED."
)