from pathlib import Path
import csv
import random
import time

import numpy as np
import nrrd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

OUT_DIR = (
    BASE_DIR
    / "results_hierarchical_v3"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# MODEL / DATA CONSTANTS
# =============================================================================

CHANNELS = [32, 64, 128, 256]

NORM_MEAN = -46.1730322190273
NORM_STD = 293.1394271328278

BG = 0
MT = 1
AIR = 2


# =============================================================================
# V3 TRAINING SETTINGS
#
# Main changes from V2:
#
# 1. Sinus stage stays recall-sensitive.
# 2. MT stage now penalizes false positives more strongly.
# 3. MT BCE positive weight reduced from 4 -> 2.
# 4. Separate Tversky settings for sinus and MT.
# 5. Two warmup epochs instead of one.
# 6. Lower full-finetune LR to avoid destroying useful pretrained features.
# =============================================================================

BATCH_SIZE = 1

MAX_EPOCHS = 25
HEAD_WARMUP_EPOCHS = 2

HEAD_LR = 1e-4
FINETUNE_LR = 2e-5

EARLY_STOP_PATIENCE = 8

SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6


# Stage 1: sinus localization
SINUS_POS_WEIGHT = 4.0

SINUS_TVERSKY_ALPHA = 0.30
SINUS_TVERSKY_BETA = 0.70


# Stage 2: MT vs air
#
# V2 used pos_weight=4 and alpha/beta=0.3/0.7.
# That drove MT recall very high but precision too low.
#
# V3 reduces positive pressure and penalizes false positives more.
MT_POS_WEIGHT = 2.0

MT_TVERSKY_ALPHA = 0.60
MT_TVERSKY_BETA = 0.40


SINUS_LOSS_WEIGHT = 0.40
MT_LOSS_WEIGHT = 0.60

SEED = 42

BASELINE_MT_DICE = 0.6868


# =============================================================================
# REPRODUCIBILITY / DEVICE
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    device = torch.device("cuda")
    print("Using CUDA")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    device = torch.device("mps")
    print("Using Apple MPS")

else:

    device = torch.device("cpu")
    print("Using CPU")


# =============================================================================
# DATASET
# =============================================================================

class MTDataset(Dataset):

    def __init__(
        self,
        scan_dir,
        mask_dir,
        augment=False,
    ):

        self.scan_dir = Path(scan_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment

        self.ids = sorted(
            p.stem
            for p
            in self.scan_dir.glob("*.nrrd")
        )

        if not self.ids:

            raise RuntimeError(
                f"No NRRD scans found in "
                f"{self.scan_dir}"
            )

        for case_id in self.ids:

            target_path = (
                self.mask_dir
                / f"{case_id}_GT.nrrd"
            )

            if not target_path.exists():

                raise FileNotFoundError(
                    target_path
                )

    def __len__(self):

        return len(self.ids)

    def __getitem__(
        self,
        index,
    ):

        case_id = self.ids[index]

        scan_path = (
            self.scan_dir
            / f"{case_id}.nrrd"
        )

        target_path = (
            self.mask_dir
            / f"{case_id}_GT.nrrd"
        )

        scan, _ = nrrd.read(
            str(scan_path)
        )

        target, _ = nrrd.read(
            str(target_path)
        )

        scan = scan.astype(
            np.float32
        )

        target = target.astype(
            np.int64
        )

        if scan.shape != target.shape:

            raise ValueError(
                f"{case_id}: "
                f"scan shape {scan.shape} "
                f"!= target shape "
                f"{target.shape}"
            )

        scan = (
            scan - NORM_MEAN
        ) / NORM_STD

        scan = (
            torch
            .from_numpy(scan)
            .unsqueeze(0)
            .float()
        )

        target = (
            torch
            .from_numpy(target)
            .long()
        )

        # Keep same augmentation used in V2.
        if (
            self.augment
            and random.random() < 0.5
        ):

            scan = torch.flip(
                scan,
                dims=[1],
            )

            target = torch.flip(
                target,
                dims=[0],
            )

        return (
            scan,
            target,
            case_id,
        )


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint_state(
    path,
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(
        checkpoint,
        dict,
    ):

        if "model" in checkpoint:

            return checkpoint["model"]

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if "state_dict" in checkpoint:

            return checkpoint[
                "state_dict"
            ]

        if all(
            torch.is_tensor(v)
            for v
            in checkpoint.values()
        ):

            return checkpoint

    raise RuntimeError(
        f"Could not find model state "
        f"inside {path}"
    )


# =============================================================================
# HIERARCHICAL MODEL
# =============================================================================

class HierarchicalMTUNet(nn.Module):

    """
    Shared pretrained 3D U-Net with two heads.

    Stage 1:
        background vs whole sinus region
        (MT + AIR)

    Stage 2:
        inside sinus:
        MT vs AIR

    Final probabilities:

        P(BG)
            = 1 - P(sinus)

        P(MT)
            = P(sinus)
            * P(MT | sinus)

        P(AIR)
            = P(sinus)
            * (1 - P(MT | sinus))
    """

    def __init__(
        self,
        source_checkpoint,
    ):

        super().__init__()

        base = UNet(
            CHANNELS
        )

        source_state = (
            load_checkpoint_state(
                source_checkpoint
            )
        )

        # IMPORTANT:
        # exact architecture match required.
        base.load_state_dict(
            source_state,
            strict=True,
        )

        if not hasattr(
            base,
            "conv_out",
        ):

            raise AttributeError(
                "UNet must expose final "
                "layer as model.conv_out"
            )

        old_head = (
            base.conv_out
        )

        if not isinstance(
            old_head,
            nn.Conv3d,
        ):

            raise TypeError(
                f"Expected conv_out "
                f"to be Conv3d, "
                f"got {type(old_head)}"
            )

        if old_head.out_channels != 3:

            raise ValueError(
                "Expected source model "
                "to have 3 output channels, "
                f"got {old_head.out_channels}"
            )

        feature_channels = (
            old_head.in_channels
        )

        # Remove original 3-class head.
        # The U-Net now returns final feature maps.
        base.conv_out = nn.Identity()

        self.backbone = base

        self.sinus_head = nn.Conv3d(
            feature_channels,
            1,
            kernel_size=old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
            dilation=old_head.dilation,
            groups=old_head.groups,
            bias=(
                old_head.bias
                is not None
            ),
        )

        self.mt_head = nn.Conv3d(
            feature_channels,
            1,
            kernel_size=old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
            dilation=old_head.dilation,
            groups=old_head.groups,
            bias=(
                old_head.bias
                is not None
            ),
        )

        # -------------------------------------------------------------
        # Initialize hierarchical heads from the already-trained
        # 3-class hybrid model rather than random initialization.
        # -------------------------------------------------------------

        with torch.no_grad():

            w = (
                old_head
                .weight
                .detach()
                .clone()
            )

            if old_head.bias is not None:

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

            # Whole-sinus head:
            #
            # sinus approximately =
            # average(MT, AIR) - BG
            self.sinus_head.weight.copy_(
                (
                    0.5
                    * (
                        w[MT]
                        + w[AIR]
                    )
                    - w[BG]
                ).unsqueeze(0)
            )

            # MT conditional head:
            #
            # MT vs AIR
            self.mt_head.weight.copy_(
                (
                    w[MT]
                    - w[AIR]
                ).unsqueeze(0)
            )

            if (
                self.sinus_head.bias
                is not None
            ):

                self.sinus_head.bias.copy_(
                    (
                        0.5
                        * (
                            b[MT]
                            + b[AIR]
                        )
                        - b[BG]
                    ).view(1)
                )

            if (
                self.mt_head.bias
                is not None
            ):

                self.mt_head.bias.copy_(
                    (
                        b[MT]
                        - b[AIR]
                    ).view(1)
                )

    def forward(
        self,
        x,
    ):

        features = (
            self.backbone(x)
        )

        sinus_logit = (
            self.sinus_head(
                features
            )
        )

        mt_logit = (
            self.mt_head(
                features
            )
        )

        return (
            sinus_logit,
            mt_logit,
        )

    @staticmethod
    def probabilities(
        sinus_logit,
        mt_logit,
    ):

        p_sinus = torch.sigmoid(
            sinus_logit
        )

        p_mt_given_sinus = (
            torch.sigmoid(
                mt_logit
            )
        )

        p_bg = (
            1.0
            - p_sinus
        )

        p_mt = (
            p_sinus
            * p_mt_given_sinus
        )

        p_air = (
            p_sinus
            * (
                1.0
                - p_mt_given_sinus
            )
        )

        return torch.cat(
            [
                p_bg,
                p_mt,
                p_air,
            ],
            dim=1,
        )

    @staticmethod
    def labels(
        sinus_logit,
        mt_logit,
    ):

        probabilities = (
            HierarchicalMTUNet
            .probabilities(
                sinus_logit,
                mt_logit,
            )
        )

        return torch.argmax(
            probabilities,
            dim=1,
        )


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def binary_tversky_loss(
    logits,
    target,
    alpha,
    beta,
    eps=1e-5,
):

    probabilities = (
        torch.sigmoid(
            logits
        )
    )

    target = (
        target.float()
    )

    dimensions = tuple(
        range(
            1,
            probabilities.ndim,
        )
    )

    true_positive = torch.sum(
        probabilities
        * target,
        dim=dimensions,
    )

    false_positive = torch.sum(
        probabilities
        * (
            1.0
            - target
        ),
        dim=dimensions,
    )

    false_negative = torch.sum(
        (
            1.0
            - probabilities
        )
        * target,
        dim=dimensions,
    )

    tversky = (
        true_positive
        + eps
    ) / (
        true_positive
        + alpha
        * false_positive
        + beta
        * false_negative
        + eps
    )

    return (
        1.0
        - tversky.mean()
    )


def masked_mt_bce_loss(
    mt_logit,
    mt_target,
    sinus_target,
):

    # Only train MT-vs-air classification
    # inside the ground-truth sinus region.
    mask = (
        sinus_target
        > 0.5
    )

    if not torch.any(
        mask
    ):

        return (
            mt_logit.sum()
            * 0.0
        )

    selected_logits = (
        mt_logit[mask]
    )

    selected_target = (
        mt_target[mask]
    )

    pos_weight = torch.tensor(
        MT_POS_WEIGHT,
        dtype=selected_logits.dtype,
        device=selected_logits.device,
    )

    return (
        F.binary_cross_entropy_with_logits(
            selected_logits,
            selected_target,
            pos_weight=pos_weight,
        )
    )


def masked_mt_tversky_loss(
    mt_logit,
    mt_target,
    sinus_target,
    alpha,
    beta,
    eps=1e-5,
):

    probabilities = (
        torch.sigmoid(
            mt_logit
        )
    )

    target = (
        mt_target.float()
    )

    mask = (
        sinus_target.float()
    )

    probabilities = (
        probabilities
        * mask
    )

    target = (
        target
        * mask
    )

    dimensions = tuple(
        range(
            1,
            probabilities.ndim,
        )
    )

    true_positive = torch.sum(
        probabilities
        * target,
        dim=dimensions,
    )

    false_positive = torch.sum(
        probabilities
        * (
            1.0
            - target
        )
        * mask,
        dim=dimensions,
    )

    false_negative = torch.sum(
        (
            1.0
            - probabilities
        )
        * target,
        dim=dimensions,
    )

    tversky = (
        true_positive
        + eps
    ) / (
        true_positive
        + alpha
        * false_positive
        + beta
        * false_negative
        + eps
    )

    return (
        1.0
        - tversky.mean()
    )


def compute_loss(
    sinus_logit,
    mt_logit,
    target,
):

    # Anything that is MT or AIR
    # belongs to the whole sinus region.
    sinus_target = (
        target != BG
    ).float().unsqueeze(1)

    mt_target = (
        target == MT
    ).float().unsqueeze(1)

    # -----------------------------------------------------------------
    # STAGE 1 LOSS
    #
    # Keep V2's recall-sensitive sinus loss.
    # -----------------------------------------------------------------

    sinus_pos_weight = torch.tensor(
        SINUS_POS_WEIGHT,
        dtype=sinus_logit.dtype,
        device=sinus_logit.device,
    )

    sinus_bce = (
        F.binary_cross_entropy_with_logits(
            sinus_logit,
            sinus_target,
            pos_weight=sinus_pos_weight,
        )
    )

    sinus_tversky = (
        binary_tversky_loss(
            sinus_logit,
            sinus_target,
            alpha=(
                SINUS_TVERSKY_ALPHA
            ),
            beta=(
                SINUS_TVERSKY_BETA
            ),
        )
    )

    sinus_loss = (
        0.40
        * sinus_bce
        + 0.60
        * sinus_tversky
    )

    # -----------------------------------------------------------------
    # STAGE 2 LOSS
    #
    # V3:
    # lower positive weighting +
    # FP-heavy Tversky.
    # -----------------------------------------------------------------

    mt_bce = (
        masked_mt_bce_loss(
            mt_logit,
            mt_target,
            sinus_target,
        )
    )

    mt_tversky = (
        masked_mt_tversky_loss(
            mt_logit,
            mt_target,
            sinus_target,
            alpha=(
                MT_TVERSKY_ALPHA
            ),
            beta=(
                MT_TVERSKY_BETA
            ),
        )
    )

    mt_loss = (
        0.40
        * mt_bce
        + 0.60
        * mt_tversky
    )

    total = (
        SINUS_LOSS_WEIGHT
        * sinus_loss
        + MT_LOSS_WEIGHT
        * mt_loss
    )

    return {
        "total": total,
        "sinus": sinus_loss,
        "mt": mt_loss,
        "sinus_bce": sinus_bce,
        "sinus_tversky": (
            sinus_tversky
        ),
        "mt_bce": mt_bce,
        "mt_tversky": (
            mt_tversky
        ),
    }


# =============================================================================
# METRICS
# =============================================================================

def binary_stats(
    prediction,
    target,
):

    prediction = (
        prediction.bool()
    )

    target = (
        target.bool()
    )

    true_positive = (
        torch.logical_and(
            prediction,
            target,
        )
        .sum()
        .item()
    )

    false_positive = (
        torch.logical_and(
            prediction,
            ~target,
        )
        .sum()
        .item()
    )

    false_negative = (
        torch.logical_and(
            ~prediction,
            target,
        )
        .sum()
        .item()
    )

    predicted_voxels = (
        prediction
        .sum()
        .item()
    )

    target_voxels = (
        target
        .sum()
        .item()
    )

    denominator = (
        predicted_voxels
        + target_voxels
    )

    dice = (
        2.0
        * true_positive
        / denominator
        if denominator
        else 1.0
    )

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    iou = (
        true_positive
        / union
        if union
        else 1.0
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        )
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        )
        else 0.0
    )

    return (
        float(dice),
        float(iou),
        float(precision),
        float(recall),
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate(
    model,
    loader,
):

    model.eval()

    total_loss = 0.0

    mt_rows = []
    air_rows = []
    sinus_rows = []

    with torch.no_grad():

        for (
            scan,
            target,
            _,
        ) in loader:

            scan = scan.to(
                device,
                dtype=torch.float32,
            )

            target = target.to(
                device,
                dtype=torch.long,
            )

            (
                sinus_logit,
                mt_logit,
            ) = model(scan)

            losses = compute_loss(
                sinus_logit,
                mt_logit,
                target,
            )

            total_loss += (
                losses[
                    "total"
                ].item()
            )

            prediction = (
                model.labels(
                    sinus_logit,
                    mt_logit,
                )
            )

            mt_rows.append(
                binary_stats(
                    prediction == MT,
                    target == MT,
                )
            )

            air_rows.append(
                binary_stats(
                    prediction == AIR,
                    target == AIR,
                )
            )

            sinus_rows.append(
                binary_stats(
                    prediction != BG,
                    target != BG,
                )
            )

    def mean_column(
        rows,
        column,
    ):

        return float(
            np.mean(
                [
                    row[column]
                    for row
                    in rows
                ]
            )
        )

    return {

        "loss":
            total_loss
            / len(loader),

        "mt_dice":
            mean_column(
                mt_rows,
                0,
            ),

        "mt_iou":
            mean_column(
                mt_rows,
                1,
            ),

        "mt_precision":
            mean_column(
                mt_rows,
                2,
            ),

        "mt_recall":
            mean_column(
                mt_rows,
                3,
            ),

        "air_dice":
            mean_column(
                air_rows,
                0,
            ),

        "sinus_dice":
            mean_column(
                sinus_rows,
                0,
            ),

        "sinus_recall":
            mean_column(
                sinus_rows,
                3,
            ),
    }


# =============================================================================
# DATA LOADERS
# =============================================================================

train_dataset = MTDataset(
    TRAIN_SCAN_DIR,
    TRAIN_MASK_DIR,
    augment=True,
)

validation_dataset = MTDataset(
    VAL_SCAN_DIR,
    VAL_MASK_DIR,
    augment=False,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


# =============================================================================
# STARTUP INFORMATION
# =============================================================================

print()
print("=" * 78)
print("HIERARCHICAL MT TRAINING V3")
print("=" * 78)

print(
    f"Training samples:   "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(validation_dataset)}"
)

print(
    "Test samples used:  0"
)

print(
    "TEST DATA LOADED: NO"
)

print(
    f"Initialization: "
    f"{SOURCE_CHECKPOINT}"
)

print()
print(
    "Stage 1: "
    "background vs whole sinus "
    "(MT + AIR)"
)

print(
    "Stage 2: "
    "inside sinus, MT vs AIR"
)

print()
print(
    f"Loss weights: "
    f"sinus={SINUS_LOSS_WEIGHT}, "
    f"MT={MT_LOSS_WEIGHT}"
)

print()
print(
    f"Sinus BCE pos_weight: "
    f"{SINUS_POS_WEIGHT}"
)

print(
    f"Sinus Tversky alpha/beta: "
    f"{SINUS_TVERSKY_ALPHA}/"
    f"{SINUS_TVERSKY_BETA}"
)

print()
print(
    f"MT BCE pos_weight: "
    f"{MT_POS_WEIGHT}"
)

print(
    f"MT Tversky alpha/beta: "
    f"{MT_TVERSKY_ALPHA}/"
    f"{MT_TVERSKY_BETA}"
)

print()
print(
    f"Warmup epochs: "
    f"{HEAD_WARMUP_EPOCHS}"
)

print(
    f"Head LR: "
    f"{HEAD_LR}"
)

print(
    f"Fine-tune LR: "
    f"{FINETUNE_LR}"
)

print(
    f"Max epochs: "
    f"{MAX_EPOCHS}"
)

print()
print(
    f"Hybrid benchmark to beat: "
    f"MT Dice {BASELINE_MT_DICE:.4f}"
)

print("=" * 78)


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

model = HierarchicalMTUNet(
    SOURCE_CHECKPOINT
).to(device)


# =============================================================================
# MODEL SHAPE CHECK
# =============================================================================

sample_scan, _, _ = (
    train_dataset[0]
)

with torch.no_grad():

    (
        sinus_logit,
        mt_logit,
    ) = model(
        sample_scan
        .unsqueeze(0)
        .to(device)
    )

    final_probabilities = (
        model.probabilities(
            sinus_logit,
            mt_logit,
        )
    )


print()
print("MODEL SHAPE CHECK")
print("-" * 78)

print(
    "Input:",
    tuple(
        sample_scan
        .unsqueeze(0)
        .shape
    ),
)

print(
    "Sinus head:",
    tuple(
        sinus_logit.shape
    ),
)

print(
    "MT head:",
    tuple(
        mt_logit.shape
    ),
)

print(
    "Final probabilities:",
    tuple(
        final_probabilities.shape
    ),
)

if (
    final_probabilities.shape[1]
    != 3
):

    raise RuntimeError(
        "Expected final "
        "3-class probabilities."
    )

print(
    "3-CLASS HIERARCHICAL "
    "OUTPUT CHECK: PASS"
)


# =============================================================================
# TRAINING HISTORY
# =============================================================================

history_path = (
    OUT_DIR
    / "training_history.csv"
)

history_fields = [

    "epoch",
    "phase",

    "train_loss",
    "train_sinus_loss",
    "train_mt_loss",

    "val_loss",

    "mt_dice",
    "mt_iou",
    "mt_precision",
    "mt_recall",

    "air_dice",

    "sinus_dice",
    "sinus_recall",

    "learning_rate",
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
# FREEZE / UNFREEZE HELPERS
# =============================================================================

def freeze_backbone():

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = (
            False
        )

    for parameter in (
        model
        .sinus_head
        .parameters()
    ):

        parameter.requires_grad = (
            True
        )

    for parameter in (
        model
        .mt_head
        .parameters()
    ):

        parameter.requires_grad = (
            True
        )


def unfreeze_all():

    for parameter in (
        model.parameters()
    ):

        parameter.requires_grad = (
            True
        )


# =============================================================================
# TRAINING STATE
# =============================================================================

best_mt_dice = -1.0
best_epoch = -1

epochs_without_improvement = 0

optimizer = None
scheduler = None


# =============================================================================
# TRAINING LOOP
# =============================================================================

try:

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.time()
        )

        # ---------------------------------------------------------------------
        # HEAD WARMUP
        # ---------------------------------------------------------------------

        if (
            epoch
            <= HEAD_WARMUP_EPOCHS
        ):

            phase = (
                "head_warmup"
            )

            if epoch == 1:

                freeze_backbone()

                optimizer = (
                    torch.optim.Adam(
                        [
                            parameter
                            for parameter
                            in model.parameters()
                            if parameter.requires_grad
                        ],
                        lr=HEAD_LR,
                    )
                )

                print()
                print("=" * 78)

                print(
                    "TRAINING PHASE: "
                    "head_warmup"
                )

                print("=" * 78)

                print(
                    "Backbone frozen; "
                    "hierarchical heads "
                    "trainable."
                )

        # ---------------------------------------------------------------------
        # FULL FINE-TUNING
        # ---------------------------------------------------------------------

        else:

            phase = (
                "full_finetune"
            )

            if (
                epoch
                == HEAD_WARMUP_EPOCHS
                + 1
            ):

                unfreeze_all()

                optimizer = (
                    torch.optim.Adam(
                        model.parameters(),
                        lr=FINETUNE_LR,
                    )
                )

                scheduler = (
                    torch.optim
                    .lr_scheduler
                    .ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=(
                            SCHEDULER_FACTOR
                        ),
                        patience=(
                            SCHEDULER_PATIENCE
                        ),
                        min_lr=(
                            MIN_LR
                        ),
                    )
                )

                print()
                print("=" * 78)

                print(
                    "TRAINING PHASE: "
                    "full_finetune"
                )

                print("=" * 78)

                print(
                    "Entire shared "
                    "pretrained U-Net "
                    "is now trainable."
                )

        # ---------------------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------------------

        model.train()

        running_total = 0.0
        running_sinus = 0.0
        running_mt = 0.0

        for (
            batch_index,
            (
                scan,
                target,
                _,
            ),
        ) in enumerate(
            train_loader,
            start=1,
        ):

            scan = scan.to(
                device,
                dtype=torch.float32,
            )

            target = target.to(
                device,
                dtype=torch.long,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            (
                sinus_logit,
                mt_logit,
            ) = model(
                scan
            )

            losses = compute_loss(
                sinus_logit,
                mt_logit,
                target,
            )

            losses[
                "total"
            ].backward()

            optimizer.step()

            running_total += (
                losses[
                    "total"
                ].item()
            )

            running_sinus += (
                losses[
                    "sinus"
                ].item()
            )

            running_mt += (
                losses[
                    "mt"
                ].item()
            )

            if (
                batch_index % 12 == 0
                or batch_index
                == len(train_loader)
            ):

                print(
                    f"Epoch "
                    f"{epoch:02d}/"
                    f"{MAX_EPOCHS} | "
                    f"{phase} | "
                    f"batch "
                    f"{batch_index:02d}/"
                    f"{len(train_loader)} | "
                    f"loss "
                    f"{losses['total'].item():.4f}"
                )

        train_loss = (
            running_total
            / len(train_loader)
        )

        train_sinus_loss = (
            running_sinus
            / len(train_loader)
        )

        train_mt_loss = (
            running_mt
            / len(train_loader)
        )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        validation = validate(
            model,
            validation_loader,
        )

        if scheduler is not None:

            scheduler.step(
                validation[
                    "mt_dice"
                ]
            )

        learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        # ---------------------------------------------------------------------
        # PRINT EPOCH RESULTS
        # ---------------------------------------------------------------------

        print()
        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        print(
            f"Phase: "
            f"{phase}"
        )

        print(
            f"TRAIN total loss="
            f"{train_loss:.4f}"
        )

        print(
            f"TRAIN sinus loss="
            f"{train_sinus_loss:.4f}"
        )

        print(
            f"TRAIN MT loss="
            f"{train_mt_loss:.4f}"
        )

        print(
            f"VAL total loss="
            f"{validation['loss']:.4f}"
        )

        print()
        print(
            "MUCOSAL THICKENING"
        )

        print(
            f"  Dice:      "
            f"{validation['mt_dice']:.4f}"
        )

        print(
            f"  IoU:       "
            f"{validation['mt_iou']:.4f}"
        )

        print(
            f"  Precision: "
            f"{validation['mt_precision']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{validation['mt_recall']:.4f}"
        )

        print()
        print(
            "SINUS AIR"
        )

        print(
            f"  Dice:      "
            f"{validation['air_dice']:.4f}"
        )

        print()
        print(
            "WHOLE SINUS REGION"
        )

        print(
            f"  Dice:      "
            f"{validation['sinus_dice']:.4f}"
        )

        print(
            f"  Recall:    "
            f"{validation['sinus_recall']:.4f}"
        )

        print()
        print(
            f"Learning rate="
            f"{learning_rate:.7f}"
        )

        print(
            f"Epoch time="
            f"{time.time() - epoch_start:.1f} sec"
        )

        improvement_vs_hybrid = (
            validation[
                "mt_dice"
            ]
            - BASELINE_MT_DICE
        )

        print(
            f"Vs hybrid benchmark: "
            f"{improvement_vs_hybrid:+.4f}"
        )

        # ---------------------------------------------------------------------
        # WRITE HISTORY
        # ---------------------------------------------------------------------

        row = {

            "epoch":
                epoch,

            "phase":
                phase,

            "train_loss":
                train_loss,

            "train_sinus_loss":
                train_sinus_loss,

            "train_mt_loss":
                train_mt_loss,

            "val_loss":
                validation["loss"],

            "mt_dice":
                validation["mt_dice"],

            "mt_iou":
                validation["mt_iou"],

            "mt_precision":
                validation[
                    "mt_precision"
                ],

            "mt_recall":
                validation[
                    "mt_recall"
                ],

            "air_dice":
                validation[
                    "air_dice"
                ],

            "sinus_dice":
                validation[
                    "sinus_dice"
                ],

            "sinus_recall":
                validation[
                    "sinus_recall"
                ],

            "learning_rate":
                learning_rate,
        }

        with open(
            history_path,
            "a",
            newline="",
        ) as file:

            writer = (
                csv.DictWriter(
                    file,
                    fieldnames=(
                        history_fields
                    ),
                )
            )

            writer.writerow(
                row
            )

        # ---------------------------------------------------------------------
        # CHECKPOINT
        # ---------------------------------------------------------------------

        checkpoint = {

            "epoch":
                epoch,

            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "scheduler":
                (
                    scheduler.state_dict()
                    if scheduler
                    is not None
                    else None
                ),

            "validation":
                validation,

            "channels":
                CHANNELS,

            "normalization_mean":
                NORM_MEAN,

            "normalization_std":
                NORM_STD,

            "hierarchical":
                True,

            "hierarchical_version":
                3,

            "source_checkpoint":
                str(
                    SOURCE_CHECKPOINT
                ),

            "sinus_pos_weight":
                SINUS_POS_WEIGHT,

            "mt_pos_weight":
                MT_POS_WEIGHT,

            "sinus_tversky_alpha":
                SINUS_TVERSKY_ALPHA,

            "sinus_tversky_beta":
                SINUS_TVERSKY_BETA,

            "mt_tversky_alpha":
                MT_TVERSKY_ALPHA,

            "mt_tversky_beta":
                MT_TVERSKY_BETA,

            "sinus_loss_weight":
                SINUS_LOSS_WEIGHT,

            "mt_loss_weight":
                MT_LOSS_WEIGHT,
        }

        torch.save(
            checkpoint,
            OUT_DIR
            / "latest.pt",
        )

        # ---------------------------------------------------------------------
        # BEST MODEL
        # ---------------------------------------------------------------------

        if (
            validation[
                "mt_dice"
            ]
            > best_mt_dice
        ):

            best_mt_dice = (
                validation[
                    "mt_dice"
                ]
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            torch.save(
                checkpoint,
                OUT_DIR
                / "best_mt_model.pt",
            )

            print()
            print(
                "*** NEW BEST "
                "HIERARCHICAL V3 "
                "MT MODEL ***"
            )

            print(
                f"MT Dice = "
                f"{best_mt_dice:.4f}"
            )

            print(
                "Saved:",
                OUT_DIR
                / "best_mt_model.pt",
            )

        elif (
            phase
            == "full_finetune"
        ):

            epochs_without_improvement += 1

        print()
        print(
            f"Best validation "
            f"MT Dice: "
            f"{best_mt_dice:.4f} "
            f"(epoch {best_epoch})"
        )

        if (
            best_mt_dice
            > BASELINE_MT_DICE
        ):

            print(
                f"CURRENT BEST BEATS "
                f"HYBRID BY "
                f"{best_mt_dice - BASELINE_MT_DICE:+.4f}"
            )

        else:

            print(
                f"Current best still "
                f"{BASELINE_MT_DICE - best_mt_dice:.4f} "
                f"below hybrid benchmark."
            )

        # ---------------------------------------------------------------------
        # EARLY STOPPING
        # ---------------------------------------------------------------------

        if (
            phase
            == "full_finetune"
        ):

            print(
                "Full-finetune epochs "
                "without MT improvement: "
                f"{epochs_without_improvement}"
            )

            if (
                epochs_without_improvement
                >= EARLY_STOP_PATIENCE
            ):

                print()
                print(
                    "EARLY STOPPING"
                )

                break

        print(
            "=" * 78
        )


# =============================================================================
# CTRL+C HANDLING
# =============================================================================

except KeyboardInterrupt:

    print()
    print(
        "Training interrupted "
        "by user."
    )

    interrupted_path = (
        OUT_DIR
        / "interrupted.pt"
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "best_mt_dice":
                best_mt_dice,

            "best_epoch":
                best_epoch,

            "hierarchical":
                True,

            "hierarchical_version":
                3,
        },
        interrupted_path,
    )

    print(
        "Saved interrupted "
        "checkpoint:",
        interrupted_path,
    )


# =============================================================================
# FINISH
# =============================================================================

print()
print("=" * 78)
print(
    "TRAINING FINISHED"
)
print("=" * 78)

print(
    f"Best validation MT Dice: "
    f"{best_mt_dice:.4f}"
)

print(
    f"Best epoch: "
    f"{best_epoch}"
)

print(
    f"Hybrid benchmark: "
    f"{BASELINE_MT_DICE:.4f}"
)

if (
    best_mt_dice
    > BASELINE_MT_DICE
):

    print(
        f"Improvement over hybrid: "
        f"{best_mt_dice - BASELINE_MT_DICE:+.4f}"
    )

else:

    print(
        f"Difference from hybrid: "
        f"{best_mt_dice - BASELINE_MT_DICE:+.4f}"
    )

print(
    "Best checkpoint:",
    OUT_DIR
    / "best_mt_model.pt",
)

print(
    "Test data were NEVER loaded."
)