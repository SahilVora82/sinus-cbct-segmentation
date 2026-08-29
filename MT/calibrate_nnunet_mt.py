from pathlib import Path
import csv

import numpy as np
import nrrd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

GT_DIR = BASE / "data" / "segment_valid"

VAL_DIR = (
    ROOT
    / "nnUNet_work"
    / "results"
    / "Dataset501_SinusMT"
    / "nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)

OUT_DIR = ROOT / "nnUNet_probability_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = 0
MT = 1
AIR = 2


def dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0

    return float(
        2 * np.logical_and(pred, gt).sum()
        / denom
    )


def metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    d = dice(pred, gt)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else 0.0
    )

    return (
        d,
        float(precision),
        float(recall)
    )


def load_probabilities(case_id):
    path = VAL_DIR / f"{case_id}.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"No probability file: {path}"
        )

    data = np.load(path)

    print(
        f"{case_id}: npz keys = "
        f"{list(data.keys())}"
    )

    if "probabilities" in data:
        probs = data["probabilities"]

    elif "softmax" in data:
        probs = data["softmax"]

    else:
        # fallback: find first 4D array
        probs = None

        for key in data.keys():
            candidate = data[key]

            if candidate.ndim == 4:
                probs = candidate
                print(
                    f"Using '{key}' as probabilities"
                )
                break

        if probs is None:
            raise RuntimeError(
                f"Could not find probability array "
                f"in {path}"
            )

    probs = probs.astype(np.float32)
    # nnU-Net probability arrays use reversed spatial axis order
    # relative to pynrrd-loaded NRRD volumes.
    probs = np.transpose(
        probs,
        (0, 3, 2, 1)
    )
    # Expected shape: classes x X x Y x Z
    if probs.shape[0] == 3:
        return probs

    # Handle channels-last if necessary
    if probs.shape[-1] == 3:
        return np.moveaxis(
            probs,
            -1,
            0
        )

    raise RuntimeError(
        f"Unexpected probability shape: "
        f"{probs.shape}"
    )


# ============================================================================
# LOAD ALL 9 VALIDATION CASES
# ============================================================================

case_ids = sorted(
    p.stem
    for p in VAL_DIR.glob("*.npz")
)

print()
print("=" * 80)
print("NNUNET MT PROBABILITY CALIBRATION")
print("=" * 80)

print(
    f"Probability files found: "
    f"{len(case_ids)}"
)

print("TEST DATA LOADED: NO")

cases = []

for case_id in case_ids:

    probs = load_probabilities(
        case_id
    )

    gt, _ = nrrd.read(
        str(
            GT_DIR
            / f"{case_id}_GT.nrrd"
        )
    )

    gt_mt = (
        gt == MT
    )

    if probs.shape[1:] != gt.shape:
        raise RuntimeError(
            f"{case_id}: probability/GT mismatch "
            f"{probs.shape} vs {gt.shape}"
        )

    cases.append(
        {
            "case": case_id,
            "probs": probs,
            "gt": gt_mt
        }
    )


# ============================================================================
# VERIFY ORIGINAL ARGMAX
# ============================================================================

original_rows = []

for case in cases:

    pred = (
        np.argmax(
            case["probs"],
            axis=0
        )
        == MT
    )

    d, p, r = metrics(
        pred,
        case["gt"]
    )

    original_rows.append(
        (case["case"], d, p, r)
    )


original_mean = np.mean(
    [x[1] for x in original_rows]
)


print()
print("=" * 80)
print("ORIGINAL ARGMAX")
print("=" * 80)

for case_id, d, p, r in original_rows:

    print(
        f"{case_id:<20} "
        f"Dice={d:.4f} "
        f"P={p:.4f} "
        f"R={r:.4f}"
    )

print()
print(
    f"Original mean Dice: "
    f"{original_mean:.4f}"
)


# ============================================================================
# CALIBRATION SWEEP
#
# Two requirements for MT:
#
# 1. P(MT) must exceed a minimum probability
# 2. P(MT) must beat the next-best class by a minimum margin
#
# This specifically attacks low-confidence false-positive MT.
# ============================================================================

thresholds = np.arange(
    0.34,
    0.801,
    0.02
)

margins = np.arange(
    0.00,
    0.301,
    0.02
)

results = []


for threshold in thresholds:

    for margin in margins:

        rows = []

        for case in cases:

            probs = case["probs"]

            p_bg = probs[BG]
            p_mt = probs[MT]
            p_air = probs[AIR]

            strongest_non_mt = np.maximum(
                p_bg,
                p_air
            )

            pred_mt = (
                (p_mt >= threshold)
                &
                (
                    p_mt
                    - strongest_non_mt
                    >= margin
                )
            )

            d, p, r = metrics(
                pred_mt,
                case["gt"]
            )

            rows.append(
                {
                    "case": case["case"],
                    "dice": d,
                    "precision": p,
                    "recall": r
                }
            )


        mean_dice = np.mean(
            [x["dice"] for x in rows]
        )

        mean_precision = np.mean(
            [x["precision"] for x in rows]
        )

        mean_recall = np.mean(
            [x["recall"] for x in rows]
        )


        results.append(
            {
                "threshold":
                    float(threshold),

                "margin":
                    float(margin),

                "mean_dice":
                    float(mean_dice),

                "precision":
                    float(mean_precision),

                "recall":
                    float(mean_recall),

                "rows":
                    rows
            }
        )


results.sort(
    key=lambda x:
        x["mean_dice"],
    reverse=True
)

best = results[0]


# ============================================================================
# RESULTS
# ============================================================================

print()
print("=" * 80)
print("BEST CALIBRATION")
print("=" * 80)

print(
    f"Minimum P(MT): "
    f"{best['threshold']:.2f}"
)

print(
    f"Minimum MT margin: "
    f"{best['margin']:.2f}"
)

print()

print(
    f"Original mean Dice: "
    f"{original_mean:.4f}"
)

print(
    f"Calibrated mean Dice: "
    f"{best['mean_dice']:.4f}"
)

print(
    f"Change: "
    f"{best['mean_dice'] - original_mean:+.4f}"
)

print()

print(
    f"Mean precision: "
    f"{best['precision']:.4f}"
)

print(
    f"Mean recall: "
    f"{best['recall']:.4f}"
)


print()
print("=" * 80)
print("PER CASE")
print("=" * 80)


original_lookup = {
    case_id: (d, p, r)
    for case_id, d, p, r
    in original_rows
}


for row in sorted(
    best["rows"],
    key=lambda x:
        x["dice"]
):

    old_dice = (
        original_lookup[
            row["case"]
        ][0]
    )

    print(
        f"{row['case']:<20} "
        f"{old_dice:.4f} "
        f"-> "
        f"{row['dice']:.4f} "
        f"({row['dice'] - old_dice:+.4f}) "
        f"P={row['precision']:.3f} "
        f"R={row['recall']:.3f}"
    )


# ============================================================================
# SAVE SWEEP
# ============================================================================

with open(
    OUT_DIR / "calibration_sweep.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "minimum_mt_probability",
            "minimum_mt_margin",
            "mean_dice",
            "mean_precision",
            "mean_recall"
        ]
    )

    for result in results:

        writer.writerow(
            [
                result["threshold"],
                result["margin"],
                result["mean_dice"],
                result["precision"],
                result["recall"]
            ]
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "Saved results:",
    OUT_DIR
)

print(
    "TEST DATA WERE NEVER LOADED."
)