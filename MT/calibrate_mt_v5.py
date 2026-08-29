from pathlib import Path
import csv
import json

import numpy as np
import nrrd
import torch

from scipy.ndimage import (
    binary_fill_holes,
    label,
)

from unet import UNet


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

VAL_SCAN_DIR = BASE_DIR / "data" / "scan_valid"
VAL_MASK_DIR = BASE_DIR / "data" / "segment_valid"

CHECKPOINT_PATH = (
    BASE_DIR
    / "results_hybrid"
    / "best_mt_model.pt"
)

OUT_DIR = (
    BASE_DIR
    / "results_calibrated_v5"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================

CHANNELS = [32, 64, 128, 256]

BG = 0
MT = 1
AIR = 2

NORM_MEAN = -46.1730322190273
NORM_STD = 293.1394271328278

HYBRID_BENCHMARK = 0.6868
POSTPROCESS_BENCHMARK = 0.6937


# =============================================================================
# DEVICE
# =============================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
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
# MODEL
# =============================================================================

model = UNet(
    CHANNELS
)

state = load_checkpoint_state(
    CHECKPOINT_PATH
)

model.load_state_dict(
    state,
    strict=True,
)

model = model.to(device)
model.eval()


# =============================================================================
# DATA
# =============================================================================

case_ids = sorted(
    path.stem
    for path
    in VAL_SCAN_DIR.glob("*.nrrd")
)

if not case_ids:
    raise RuntimeError(
        "No validation scans found."
    )

print()
print(
    f"Validation cases: "
    f"{len(case_ids)}"
)

print(
    "TEST DATA LOADED: NO"
)


# =============================================================================
# CACHE LOGITS ONCE
# =============================================================================

cases = []

print()
print("=" * 78)
print("CACHING VALIDATION LOGITS")
print("=" * 78)

with torch.no_grad():

    for case_id in case_ids:

        scan_path = (
            VAL_SCAN_DIR
            / f"{case_id}.nrrd"
        )

        mask_path = (
            VAL_MASK_DIR
            / f"{case_id}_GT.nrrd"
        )

        scan, _ = nrrd.read(
            str(scan_path)
        )

        target, _ = nrrd.read(
            str(mask_path)
        )

        scan = scan.astype(
            np.float32
        )

        target = target.astype(
            np.int64
        )

        scan = (
            scan - NORM_MEAN
        ) / NORM_STD

        scan_tensor = (
            torch
            .from_numpy(scan)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        logits = model(
            scan_tensor
        )

        logits = (
            logits
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        cases.append(
            {
                "id": case_id,
                "logits": logits,
                "target": target,
            }
        )

        print(
            f"Cached {case_id}"
        )


# =============================================================================
# METRICS
# =============================================================================

def binary_metrics(
    prediction,
    target,
):

    prediction = (
        prediction.astype(bool)
    )

    target = (
        target.astype(bool)
    )

    tp = np.logical_and(
        prediction,
        target,
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(target),
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        target,
    ).sum()

    pred_count = (
        prediction.sum()
    )

    target_count = (
        target.sum()
    )

    denominator = (
        pred_count
        + target_count
    )

    dice = (
        2.0 * tp / denominator
        if denominator > 0
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

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# =============================================================================
# POSTPROCESSING
# =============================================================================

CONNECTIVITY_26 = np.ones(
    (3, 3, 3),
    dtype=np.uint8,
)


def remove_small_components(
    binary_mask,
    minimum_size,
):

    if minimum_size <= 0:
        return binary_mask

    labeled, number = label(
        binary_mask,
        structure=CONNECTIVITY_26,
    )

    if number == 0:
        return binary_mask

    counts = np.bincount(
        labeled.ravel()
    )

    keep = np.zeros(
        len(counts),
        dtype=bool,
    )

    keep[
        counts >= minimum_size
    ] = True

    # Never keep background label 0.
    keep[0] = False

    return keep[
        labeled
    ]


def postprocess_mt(
    mt_mask,
    fill_holes,
    minimum_size,
):

    result = (
        mt_mask.copy()
    )

    if fill_holes:
        result = binary_fill_holes(
            result
        )

    if minimum_size > 0:
        result = remove_small_components(
            result,
            minimum_size,
        )

    return result.astype(bool)


# =============================================================================
# PREDICTION
#
# V5 idea:
#
# Add a scalar bias to MT logit:
#
#     adjusted_MT_logit = MT_logit + bias
#
# Negative bias:
#     harder to call MT
#     => precision up, recall down
#
# Positive bias:
#     easier to call MT
#     => recall up, precision down
#
# =============================================================================

def predict_mt(
    logits,
    mt_bias,
):

    adjusted = (
        logits.copy()
    )

    adjusted[MT] += (
        mt_bias
    )

    prediction = np.argmax(
        adjusted,
        axis=0,
    )

    return (
        prediction == MT
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_configuration(
    mt_bias,
    fill_holes=False,
    minimum_size=0,
):

    rows = []

    for case in cases:

        mt_prediction = predict_mt(
            case["logits"],
            mt_bias,
        )

        if (
            fill_holes
            or minimum_size > 0
        ):

            mt_prediction = postprocess_mt(
                mt_prediction,
                fill_holes,
                minimum_size,
            )

        gt_mt = (
            case["target"] == MT
        )

        metrics = binary_metrics(
            mt_prediction,
            gt_mt,
        )

        rows.append(
            {
                "case":
                    case["id"],

                **metrics,
            }
        )

    mean_dice = float(
        np.mean(
            [
                row["dice"]
                for row in rows
            ]
        )
    )

    mean_iou = float(
        np.mean(
            [
                row["iou"]
                for row in rows
            ]
        )
    )

    mean_precision = float(
        np.mean(
            [
                row["precision"]
                for row in rows
            ]
        )
    )

    mean_recall = float(
        np.mean(
            [
                row["recall"]
                for row in rows
            ]
        )
    )

    return {
        "mt_bias":
            float(mt_bias),

        "fill_holes":
            bool(fill_holes),

        "minimum_size":
            int(minimum_size),

        "mean_dice":
            mean_dice,

        "mean_iou":
            mean_iou,

        "mean_precision":
            mean_precision,

        "mean_recall":
            mean_recall,

        "rows":
            rows,
    }


# =============================================================================
# STEP 1 — RAW BIAS SWEEP
# =============================================================================

print()
print("=" * 78)
print("STEP 1: MT LOGIT-BIAS SWEEP")
print("=" * 78)

bias_values = np.arange(
    -1.00,
    1.0001,
    0.025,
)

raw_results = []

for bias in bias_values:

    result = evaluate_configuration(
        mt_bias=float(bias),
        fill_holes=False,
        minimum_size=0,
    )

    raw_results.append(
        result
    )


raw_results.sort(
    key=lambda item:
        item["mean_dice"],
    reverse=True,
)


print()
print("TOP RAW CALIBRATIONS")
print("-" * 78)

for result in raw_results[:10]:

    print(
        f"bias={result['mt_bias']:+.3f}  "
        f"Dice={result['mean_dice']:.4f}  "
        f"P={result['mean_precision']:.4f}  "
        f"R={result['mean_recall']:.4f}"
    )


# =============================================================================
# STEP 2 — POSTPROCESS TOP BIASES
# =============================================================================

print()
print("=" * 78)
print("STEP 2: POSTPROCESS TOP CALIBRATIONS")
print("=" * 78)

top_biases = [
    result["mt_bias"]
    for result
    in raw_results[:8]
]

minimum_sizes = [
    0,
    10,
    25,
    50,
    100,
    200,
]

fill_options = [
    False,
    True,
]

all_results = []

for bias in top_biases:

    for fill_holes in fill_options:

        for minimum_size in minimum_sizes:

            result = evaluate_configuration(
                mt_bias=bias,
                fill_holes=fill_holes,
                minimum_size=minimum_size,
            )

            all_results.append(
                result
            )


all_results.sort(
    key=lambda item:
        item["mean_dice"],
    reverse=True,
)

best = all_results[0]


# =============================================================================
# RESULTS
# =============================================================================

print()
print("=" * 78)
print("BEST V5 CALIBRATED CONFIGURATION")
print("=" * 78)

print(
    f"MT logit bias:     "
    f"{best['mt_bias']:+.3f}"
)

print(
    f"Fill holes:        "
    f"{best['fill_holes']}"
)

print(
    f"Min component:     "
    f"{best['minimum_size']}"
)

print()
print(
    f"Mean MT Dice:      "
    f"{best['mean_dice']:.4f}"
)

print(
    f"Mean MT IoU:       "
    f"{best['mean_iou']:.4f}"
)

print(
    f"Mean precision:    "
    f"{best['mean_precision']:.4f}"
)

print(
    f"Mean recall:       "
    f"{best['mean_recall']:.4f}"
)

print()
print(
    f"Vs raw hybrid:     "
    f"{best['mean_dice'] - HYBRID_BENCHMARK:+.4f}"
)

print(
    f"Vs old postproc:   "
    f"{best['mean_dice'] - POSTPROCESS_BENCHMARK:+.4f}"
)


# =============================================================================
# PER-CASE RESULTS
# =============================================================================

print()
print("PER-CASE")
print("-" * 78)

for row in sorted(
    best["rows"],
    key=lambda item:
        item["dice"],
):

    print(
        f"{row['case']:<20} "
        f"Dice={row['dice']:.4f}  "
        f"P={row['precision']:.4f}  "
        f"R={row['recall']:.4f}"
    )


# =============================================================================
# SAVE
# =============================================================================

summary = {
    key: value
    for key, value
    in best.items()
    if key != "rows"
}

summary["hybrid_benchmark"] = (
    HYBRID_BENCHMARK
)

summary[
    "old_postprocess_benchmark"
] = (
    POSTPROCESS_BENCHMARK
)


with open(
    OUT_DIR
    / "best_configuration.json",
    "w",
) as file:

    json.dump(
        summary,
        file,
        indent=2,
    )


with open(
    OUT_DIR
    / "best_per_case.csv",
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "case",
            "dice",
            "iou",
            "precision",
            "recall",
        ],
    )

    writer.writeheader()

    writer.writerows(
        best["rows"]
    )


with open(
    OUT_DIR
    / "raw_bias_sweep.csv",
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "mt_bias",
            "mean_dice",
            "mean_iou",
            "mean_precision",
            "mean_recall",
        ],
    )

    writer.writeheader()

    for result in raw_results:

        writer.writerow(
            {
                "mt_bias":
                    result["mt_bias"],

                "mean_dice":
                    result["mean_dice"],

                "mean_iou":
                    result["mean_iou"],

                "mean_precision":
                    result["mean_precision"],

                "mean_recall":
                    result["mean_recall"],
            }
        )


print()
print(
    "Saved:",
    OUT_DIR
)

print(
    "TEST DATA WERE NEVER LOADED."
)