from pathlib import Path
import argparse
import csv
import time

import numpy as np
import nrrd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from unet import UNet


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

TRAIN_SCAN_DIR = BASE_DIR / "data" / "scan"
TRAIN_MASK_DIR = BASE_DIR / "data" / "segment"

VAL_SCAN_DIR = BASE_DIR / "data" / "scan_valid"
VAL_MASK_DIR = BASE_DIR / "data" / "segment_valid"


# ------------------------------------------------------------
# IMPORTANT:
#
# There is intentionally NO reference anywhere in this script
# to scan_test or segment_test.
#
# MT test data remain completely untouched.
# ------------------------------------------------------------


NUM_CLASSES = 3

LABEL_NAMES = {
    0: "background",
    1: "mucosal_thickening",
    2: "sinus_air",
}


# SAME architecture as healthy model
CHANNELS = [32, 64, 128, 256]


# MT TRAINING-ONLY normalization
TRAIN_MEAN = -46.1730322190273
TRAIN_STD = 293.1394271328278


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train MT 3D U-Net with optional healthy-sinus transfer learning."
    )

    parser.add_argument(
        "--init",
        choices=["transfer", "scratch"],
        default="transfer",
        help="Initialize from healthy model or from random weights."
    )

    parser.add_argument(
        "--healthy-checkpoint",
        type=Path,
        default=(
            BASE_DIR.parent
            / "Healthy"
            / "checkpoints_fixed_training_normalization"
            / "model_50.pth"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Train only new 3-class output head for this many epochs."
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--scratch-lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None
    )

    return parser.parse_args()


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            "Using CUDA:",
            torch.cuda.get_device_name(0)
        )

        return device

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):

        print("Using Apple MPS")
        return torch.device("mps")

    print("Using CPU")
    return torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class MTDataset(Dataset):

    def __init__(
        self,
        scan_dir: Path,
        mask_dir: Path
    ):

        self.scan_dir = Path(scan_dir)
        self.mask_dir = Path(mask_dir)

        self.scan_files = sorted(
            self.scan_dir.glob("*.nrrd")
        )

        if not self.scan_files:

            raise RuntimeError(
                f"No scans found in {self.scan_dir}"
            )

        for scan_path in self.scan_files:

            mask_path = (
                self.mask_dir
                / f"{scan_path.stem}_GT.nrrd"
            )

            if not mask_path.exists():

                raise FileNotFoundError(
                    f"Missing mask for {scan_path.name}"
                )

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, index):

        scan_path = self.scan_files[index]

        mask_path = (
            self.mask_dir
            / f"{scan_path.stem}_GT.nrrd"
        )

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
            np.int64
        )

        # --------------------------------------------
        # SAME normalization applied to every
        # training and validation scan.
        # Statistics came from MT TRAINING ONLY.
        # --------------------------------------------

        scan = (
            scan - TRAIN_MEAN
        ) / TRAIN_STD

        scan = torch.from_numpy(
            scan
        ).float().unsqueeze(0)

        mask = torch.from_numpy(
            mask
        ).long()

        return scan, mask


# ============================================================
# MODEL
# ============================================================

def build_three_class_model():

    # Original healthy architecture
    model = UNet(
        CHANNELS
    )

    if not hasattr(
        model,
        "conv_out"
    ):
        raise RuntimeError(
            "Expected U-Net to contain model.conv_out."
        )

    old_head = model.conv_out

    print()
    print("Original output layer:")
    print(old_head)

    # --------------------------------------------------------
    # HEALTHY:
    #   1 output channel
    #
    # MT:
    #   3 output channels
    #
    # 0 background
    # 1 MT
    # 2 air
    # --------------------------------------------------------

    model.conv_out = nn.Conv3d(
        in_channels=old_head.in_channels,
        out_channels=NUM_CLASSES,
        kernel_size=old_head.kernel_size,
        stride=old_head.stride,
        padding=old_head.padding,
        dilation=old_head.dilation,
        groups=old_head.groups,
        bias=(old_head.bias is not None),
        padding_mode=old_head.padding_mode
    )

    print()
    print("NEW MT output layer:")
    print(model.conv_out)

    return model


# ============================================================
# CHECKPOINT UTILITIES
# ============================================================

def safe_torch_load(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        return torch.load(
            path,
            map_location="cpu"
        )


def extract_state_dict(checkpoint):

    if not isinstance(
        checkpoint,
        dict
    ):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    if "model_state_dict" in checkpoint:

        state = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state = checkpoint[
            "state_dict"
        ]

    elif "model" in checkpoint:

        state = checkpoint[
            "model"
        ]

    else:

        # Some checkpoints ARE the state dict
        tensor_values = [
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ]

        if tensor_values and all(
            tensor_values
        ):

            state = checkpoint

        else:

            raise RuntimeError(
                "Could not find model weights "
                "inside healthy checkpoint."
            )

    cleaned = {}

    for key, value in state.items():

        # Handle DataParallel checkpoints
        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


# ============================================================
# TRANSFER HEALTHY WEIGHTS
# ============================================================

def transfer_healthy_weights(
    model,
    checkpoint_path
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "\nHealthy checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    print()
    print("=" * 70)
    print("LOADING HEALTHY PRETRAINED MODEL")
    print("=" * 70)

    print(
        "Checkpoint:",
        checkpoint_path
    )

    checkpoint = safe_torch_load(
        checkpoint_path
    )

    healthy_state = extract_state_dict(
        checkpoint
    )

    mt_state = model.state_dict()

    transferred = []
    skipped = []

    for key, value in healthy_state.items():

        # --------------------------------------------
        # DO NOT transfer the healthy output layer.
        #
        # Healthy output = 1 channel
        # MT output      = 3 channels
        # --------------------------------------------

        if key.startswith("conv_out."):

            skipped.append(
                (
                    key,
                    "healthy output head"
                )
            )

            continue

        if key not in mt_state:

            skipped.append(
                (
                    key,
                    "key not present"
                )
            )

            continue

        if (
            mt_state[key].shape
            != value.shape
        ):

            skipped.append(
                (
                    key,
                    "shape mismatch"
                )
            )

            continue

        mt_state[key] = value

        transferred.append(
            key
        )

    # --------------------------------------------------------
    # Load COMPLETE resulting MT state strictly.
    #
    # We manually merged compatible weights above, therefore
    # we do not need strict=False.
    # --------------------------------------------------------

    model.load_state_dict(
        mt_state,
        strict=True
    )

    transferable_target_keys = [
        key
        for key in mt_state
        if not key.startswith("conv_out.")
    ]

    coverage = (
        len(transferred)
        /
        max(
            1,
            len(transferable_target_keys)
        )
    )

    print()
    print(
        f"Transferred parameters/buffers: "
        f"{len(transferred)}"
    )

    print(
        f"Expected transferable keys: "
        f"{len(transferable_target_keys)}"
    )

    print(
        f"Transfer coverage: "
        f"{coverage * 100:.1f}%"
    )

    print()
    print(
        "Healthy conv_out was NOT transferred."
    )

    print(
        "MT conv_out remains freshly initialized."
    )

    # If architectures unexpectedly do not match,
    # stop instead of silently pretending transfer worked.

    if coverage < 0.90:

        print()
        print("Skipped keys:")

        for item in skipped:

            print(
                " ",
                item
            )

        raise RuntimeError(
            "Less than 90% of the non-output "
            "U-Net parameters matched. "
            "The architectures may differ."
        )

    print()
    print(
        "HEALTHY -> MT TRANSFER SUCCESSFUL"
    )

    print("=" * 70)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    mask_directory
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    mask_files = sorted(
        Path(mask_directory).glob(
            "*_GT.nrrd"
        )
    )

    print()
    print(
        "Calculating class weights from "
        "TRAINING MASKS ONLY..."
    )

    for path in mask_files:

        mask, _ = nrrd.read(
            str(path)
        )

        mask = mask.astype(
            np.int64
        )

        counts += np.bincount(
            mask.reshape(-1),
            minlength=NUM_CLASSES
        )[:NUM_CLASSES]

    frequencies = (
        counts / counts.sum()
    )

    # MT is tiny compared with background.
    #
    # Inverse square-root weighting gives MT
    # additional importance without using
    # extremely aggressive raw inverse frequency.

    weights = (
        1.0
        /
        np.sqrt(
            frequencies + 1e-8
        )
    )

    weights /= weights.mean()

    print()
    print("TRAINING CLASS DISTRIBUTION")
    print("-" * 60)

    for class_id in range(
        NUM_CLASSES
    ):

        print(
            f"{class_id} "
            f"{LABEL_NAMES[class_id]:22s} "
            f"{int(counts[class_id]):12,d} voxels | "
            f"{frequencies[class_id] * 100:7.3f}% | "
            f"weight={weights[class_id]:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ============================================================
# LOSS
# ============================================================

def foreground_dice_loss(
    logits,
    target
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    target_one_hot = F.one_hot(
        target,
        num_classes=NUM_CLASSES
    )

    target_one_hot = (
        target_one_hot
        .permute(
            0,
            4,
            1,
            2,
            3
        )
        .float()
    )

    losses = []

    # Only clinically relevant foreground classes
    # participate in Dice loss.
    #
    # 1 = MT
    # 2 = sinus air

    for class_id in [1, 2]:

        prediction = probabilities[
            :,
            class_id
        ]

        truth = target_one_hot[
            :,
            class_id
        ]

        intersection = (
            prediction
            * truth
        ).sum()

        denominator = (
            prediction.sum()
            + truth.sum()
        )

        dice = (
            2.0 * intersection
            + 1e-6
        ) / (
            denominator
            + 1e-6
        )

        losses.append(
            1.0 - dice
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# METRICS
# ============================================================

def class_metrics(
    prediction,
    target,
    class_id
):

    predicted_class = (
        prediction == class_id
    )

    true_class = (
        target == class_id
    )

    tp = (
        predicted_class
        & true_class
    ).sum().item()

    fp = (
        predicted_class
        & (~true_class)
    ).sum().item()

    fn = (
        (~predicted_class)
        & true_class
    ).sum().item()

    dice_denom = (
        2 * tp
        + fp
        + fn
    )

    iou_denom = (
        tp
        + fp
        + fn
    )

    dice = (
        2 * tp / dice_denom
        if dice_denom > 0
        else 1.0
    )

    iou = (
        tp / iou_denom
        if iou_denom > 0
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
        else 1.0
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def average_metric(
    results,
    key
):

    return float(
        np.mean(
            [
                result[key]
                for result in results
            ]
        )
    )


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

def train_only_new_head(
    model
):

    for parameter in model.parameters():

        parameter.requires_grad = False

    for parameter in model.conv_out.parameters():

        parameter.requires_grad = True


def train_entire_model(
    model
):

    for parameter in model.parameters():

        parameter.requires_grad = True


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    device = get_device()

    if args.output_dir is None:

        if args.init == "transfer":

            output_dir = (
                BASE_DIR
                / "results_transfer"
            )

        else:

            output_dir = (
                BASE_DIR
                / "results_scratch"
            )

    else:

        output_dir = (
            args.output_dir
            .expanduser()
            .resolve()
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_foreground_path = (
        output_dir
        / "best_foreground.pt"
    )

    best_mt_path = (
        output_dir
        / "best_mt.pt"
    )

    latest_path = (
        output_dir
        / "latest.pt"
    )

    history_path = (
        output_dir
        / "training_history.csv"
    )


    print()
    print("=" * 70)
    print("MT 3D U-NET")
    print("=" * 70)

    print(
        f"Initialization: {args.init}"
    )

    print(
        f"Architecture: {CHANNELS}"
    )

    print(
        f"MT normalization mean: {TRAIN_MEAN}"
    )

    print(
        f"MT normalization std:  {TRAIN_STD}"
    )

    print()
    print(
        "TEST DATA LOADED: NO"
    )

    print("=" * 70)


    # ========================================================
    # DATA
    # ========================================================

    train_dataset = MTDataset(
        TRAIN_SCAN_DIR,
        TRAIN_MASK_DIR
    )

    val_dataset = MTDataset(
        VAL_SCAN_DIR,
        VAL_MASK_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print()
    print(
        f"Training samples:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        "Test samples used:  0"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = build_three_class_model()

    start_epoch = 1

    best_foreground = -1.0
    best_mt = -1.0


    if args.resume is not None:

        resume_checkpoint = safe_torch_load(
            args.resume
        )

        model.load_state_dict(
            resume_checkpoint[
                "model_state_dict"
            ],
            strict=True
        )

        start_epoch = (
            int(
                resume_checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_foreground = float(
            resume_checkpoint.get(
                "best_foreground",
                -1.0
            )
        )

        best_mt = float(
            resume_checkpoint.get(
                "best_mt",
                -1.0
            )
        )

        print()
        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )

    elif args.init == "transfer":

        transfer_healthy_weights(
            model,
            args.healthy_checkpoint
        )

    else:

        print()
        print(
            "Using RANDOM INITIALIZATION."
        )


    model = model.to(
        device
    )


    # ========================================================
    # MODEL SHAPE CHECK
    # ========================================================

    sample_scan, sample_mask = (
        train_dataset[0]
    )

    sample_input = (
        sample_scan
        .unsqueeze(0)
        .to(device)
    )

    model.eval()

    with torch.no_grad():

        sample_output = model(
            sample_input
        )

    print()
    print("MODEL SHAPE CHECK")
    print("-" * 60)

    print(
        "Input:",
        tuple(
            sample_input.shape
        )
    )

    print(
        "Output:",
        tuple(
            sample_output.shape
        )
    )

    print(
        "Mask:",
        tuple(
            sample_mask.shape
        )
    )

    expected_shape = (
        1,
        NUM_CLASSES,
        96,
        96,
        96
    )

    if tuple(
        sample_output.shape
    ) != expected_shape:

        raise RuntimeError(
            f"Expected output "
            f"{expected_shape}, got "
            f"{tuple(sample_output.shape)}"
        )

    print()
    print(
        "3-CLASS OUTPUT CHECK: PASS"
    )


    # ========================================================
    # LOSS
    # ========================================================

    class_weights = (
        calculate_class_weights(
            TRAIN_MASK_DIR
        )
        .to(device)
    )

    cross_entropy = (
        nn.CrossEntropyLoss(
            weight=class_weights
        )
    )


    def calculate_loss(
        logits,
        masks
    ):

        ce = cross_entropy(
            logits,
            masks
        )

        dice = foreground_dice_loss(
            logits,
            masks
        )

        total = (
            0.5 * ce
            + 0.5 * dice
        )

        return total


    # ========================================================
    # CSV
    # ========================================================

    fields = [
        "epoch",
        "phase",
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
        "seconds",
    ]

    if start_epoch == 1:

        with open(
            history_path,
            "w",
            newline=""
        ) as file:

            csv.DictWriter(
                file,
                fieldnames=fields
            ).writeheader()


    # ========================================================
    # TRAINING
    # ========================================================

    current_phase = None
    optimizer = None
    scheduler = None


    for epoch in range(
        start_epoch,
        args.epochs + 1
    ):

        epoch_start = time.time()


        # ----------------------------------------------------
        # TRAINING PHASE
        # ----------------------------------------------------

        if (
            args.init == "transfer"
            and epoch <= args.warmup_epochs
        ):

            desired_phase = "head_warmup"

        else:

            desired_phase = "full_finetune"


        if desired_phase != current_phase:

            current_phase = (
                desired_phase
            )

            print()
            print("=" * 70)
            print(
                "TRAINING PHASE:",
                current_phase
            )
            print("=" * 70)


            if (
                current_phase
                == "head_warmup"
            ):

                train_only_new_head(
                    model
                )

                optimizer = torch.optim.AdamW(
                    filter(
                        lambda p:
                        p.requires_grad,
                        model.parameters()
                    ),
                    lr=args.head_lr
                )

                scheduler = None

                print(
                    "Frozen: pretrained U-Net"
                )

                print(
                    "Training: new 3-class head only"
                )


            else:

                train_entire_model(
                    model
                )

                learning_rate = (
                    args.finetune_lr
                    if args.init == "transfer"
                    else args.scratch_lr
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate
                )

                scheduler = (
                    torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=0.5,
                        patience=4
                    )
                )

                print(
                    "Entire U-Net is trainable."
                )


        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        total_train_loss = 0.0

        for batch_index, (
            scans,
            masks
        ) in enumerate(
            train_loader,
            start=1
        ):

            scans = scans.to(
                device
            )

            masks = masks.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                scans
            )

            loss = calculate_loss(
                logits,
                masks
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
            )

            print(
                f"\r"
                f"Epoch "
                f"{epoch:02d}/"
                f"{args.epochs} | "
                f"{current_phase} | "
                f"Batch "
                f"{batch_index:02d}/"
                f"{len(train_loader)} | "
                f"Loss "
                f"{loss.item():.4f}",
                end="",
                flush=True
            )


        train_loss = (
            total_train_loss
            /
            len(train_loader)
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        total_val_loss = 0.0

        mt_results = []
        air_results = []


        with torch.no_grad():

            for scans, masks in val_loader:

                scans = scans.to(
                    device
                )

                masks = masks.to(
                    device
                )

                logits = model(
                    scans
                )

                loss = calculate_loss(
                    logits,
                    masks
                )

                total_val_loss += (
                    loss.item()
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                for batch_item in range(
                    predictions.shape[0]
                ):

                    mt_results.append(
                        class_metrics(
                            predictions[
                                batch_item
                            ],
                            masks[
                                batch_item
                            ],
                            class_id=1
                        )
                    )

                    air_results.append(
                        class_metrics(
                            predictions[
                                batch_item
                            ],
                            masks[
                                batch_item
                            ],
                            class_id=2
                        )
                    )


        val_loss = (
            total_val_loss
            /
            len(val_loader)
        )


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        mt_dice = average_metric(
            mt_results,
            "dice"
        )

        mt_iou = average_metric(
            mt_results,
            "iou"
        )

        mt_precision = average_metric(
            mt_results,
            "precision"
        )

        mt_recall = average_metric(
            mt_results,
            "recall"
        )


        air_dice = average_metric(
            air_results,
            "dice"
        )

        air_iou = average_metric(
            air_results,
            "iou"
        )

        air_precision = average_metric(
            air_results,
            "precision"
        )

        air_recall = average_metric(
            air_results,
            "recall"
        )


        foreground_mean_dice = (
            mt_dice
            + air_dice
        ) / 2.0


        if scheduler is not None:

            scheduler.step(
                foreground_mean_dice
            )


        learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )

        seconds = (
            time.time()
            - epoch_start
        )


        # ====================================================
        # PRINT
        # ====================================================

        print("\n")

        print(
            f"Epoch {epoch:02d}"
        )

        print(
            f"Phase: {current_phase}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val loss:   "
            f"{val_loss:.4f}"
        )

        print()

        print("MUCOSAL THICKENING")

        print(
            f"  Dice:      "
            f"{mt_dice:.4f}"
        )

        print(
            f"  IoU:       "
            f"{mt_iou:.4f}"
        )

        print(
            f"  Precision: "
            f"{mt_precision:.4f}"
        )

        print(
            f"  Recall:    "
            f"{mt_recall:.4f}"
        )

        print()

        print("SINUS AIR")

        print(
            f"  Dice:      "
            f"{air_dice:.4f}"
        )

        print(
            f"  IoU:       "
            f"{air_iou:.4f}"
        )

        print(
            f"  Precision: "
            f"{air_precision:.4f}"
        )

        print(
            f"  Recall:    "
            f"{air_recall:.4f}"
        )

        print()

        print(
            "Mean foreground Dice:",
            f"{foreground_mean_dice:.4f}"
        )

        print(
            "Learning rate:",
            f"{learning_rate:.7f}"
        )

        print(
            "Epoch time:",
            f"{seconds:.1f} seconds"
        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        checkpoint = {

            "epoch":
                epoch,

            "phase":
                current_phase,

            "initialization":
                args.init,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "mt_dice":
                mt_dice,

            "mt_iou":
                mt_iou,

            "mt_precision":
                mt_precision,

            "mt_recall":
                mt_recall,

            "air_dice":
                air_dice,

            "air_iou":
                air_iou,

            "air_precision":
                air_precision,

            "air_recall":
                air_recall,

            "foreground_mean_dice":
                foreground_mean_dice,

            "best_foreground":
                max(
                    best_foreground,
                    foreground_mean_dice
                ),

            "best_mt":
                max(
                    best_mt,
                    mt_dice
                ),

            "channels":
                CHANNELS,

            "labels":
                LABEL_NAMES,

            "normalization_mean":
                TRAIN_MEAN,

            "normalization_std":
                TRAIN_STD,

            "healthy_checkpoint":
                (
                    str(
                        args.healthy_checkpoint
                    )
                    if args.init
                    == "transfer"
                    else None
                ),
        }


        # Always save most recent epoch

        torch.save(
            checkpoint,
            latest_path
        )


        # Best balanced foreground model

        if (
            foreground_mean_dice
            > best_foreground
        ):

            best_foreground = (
                foreground_mean_dice
            )

            torch.save(
                checkpoint,
                best_foreground_path
            )

            print()
            print(
                "*** NEW BEST FOREGROUND MODEL ***"
            )

            print(
                f"Mean Dice = "
                f"{best_foreground:.4f}"
            )


        # Best MT-specific model

        if mt_dice > best_mt:

            best_mt = mt_dice

            torch.save(
                checkpoint,
                best_mt_path
            )

            print()
            print(
                "*** NEW BEST MT MODEL ***"
            )

            print(
                f"MT Dice = "
                f"{best_mt:.4f}"
            )


        # ====================================================
        # HISTORY CSV
        # ====================================================

        row = {

            "epoch":
                epoch,

            "phase":
                current_phase,

            "learning_rate":
                learning_rate,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "mt_dice":
                mt_dice,

            "mt_iou":
                mt_iou,

            "mt_precision":
                mt_precision,

            "mt_recall":
                mt_recall,

            "air_dice":
                air_dice,

            "air_iou":
                air_iou,

            "air_precision":
                air_precision,

            "air_recall":
                air_recall,

            "foreground_mean_dice":
                foreground_mean_dice,

            "seconds":
                seconds,
        }


        with open(
            history_path,
            "a",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields
            )

            writer.writerow(
                row
            )


        print()

        print(
            f"Best foreground Dice: "
            f"{best_foreground:.4f}"
        )

        print(
            f"Best MT Dice: "
            f"{best_mt:.4f}"
        )

        print("=" * 70)


        if (
            device.type == "cuda"
        ):

            torch.cuda.empty_cache()

        elif (
            device.type == "mps"
            and hasattr(
                torch.mps,
                "empty_cache"
            )
        ):

            torch.mps.empty_cache()


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best foreground checkpoint:\n"
        f"{best_foreground_path}"
    )

    print()

    print(
        f"Best MT checkpoint:\n"
        f"{best_mt_path}"
    )

    print()

    print(
        f"Latest checkpoint:\n"
        f"{latest_path}"
    )

    print()

    print(
        f"Training history:\n"
        f"{history_path}"
    )

    print()
    print(
        "MT TEST SET WAS NEVER LOADED."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()