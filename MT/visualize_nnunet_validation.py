from pathlib import Path

import numpy as np
import nrrd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

SCAN_DIR = BASE / "data" / "scan_valid"
GT_DIR = BASE / "data" / "segment_valid"

NNUNET_RESULTS = ROOT / "nnUNet_work" / "results"

OUT_DIR = ROOT / "nnUNet_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MT_LABEL = 1


# -------------------------------------------------------------------------
# Find nnU-Net validation prediction
# -------------------------------------------------------------------------

def find_prediction(case_id):

    candidates = list(
        NNUNET_RESULTS.rglob(
            f"{case_id}.nrrd"
        )
    )

    # Prefer files inside a validation folder
    validation_candidates = [
        p for p in candidates
        if "validation" in str(p).lower()
    ]

    if validation_candidates:
        return validation_candidates[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not find prediction for {case_id}"
    )


# -------------------------------------------------------------------------
# Dice
# -------------------------------------------------------------------------

def dice_score(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0

    return (
        2.0
        * np.logical_and(pred, gt).sum()
        / denom
    )


# -------------------------------------------------------------------------
# Nice CBCT display
# -------------------------------------------------------------------------

def normalize_for_display(image):

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    image = np.clip(
        image,
        low,
        high,
    )

    image = (
        image - low
    ) / (
        high - low + 1e-8
    )

    return image


# -------------------------------------------------------------------------
# Cases
# -------------------------------------------------------------------------

case_ids = sorted(
    p.stem
    for p in SCAN_DIR.glob("*.nrrd")
)

print("Cases:", len(case_ids))


overview_data = []


# -------------------------------------------------------------------------
# Make individual figures
# -------------------------------------------------------------------------

for case_id in case_ids:

    scan_path = (
        SCAN_DIR /
        f"{case_id}.nrrd"
    )

    gt_path = (
        GT_DIR /
        f"{case_id}_GT.nrrd"
    )

    pred_path = find_prediction(
        case_id
    )

    print()
    print(case_id)
    print("Prediction:", pred_path)

    scan, _ = nrrd.read(
        str(scan_path)
    )

    gt, _ = nrrd.read(
        str(gt_path)
    )

    pred, _ = nrrd.read(
        str(pred_path)
    )

    if scan.shape != gt.shape:
        raise RuntimeError(
            f"{case_id}: scan/GT mismatch "
            f"{scan.shape} vs {gt.shape}"
        )

    if pred.shape != gt.shape:
        raise RuntimeError(
            f"{case_id}: prediction/GT mismatch "
            f"{pred.shape} vs {gt.shape}"
        )

    gt_mt = (
        gt == MT_LABEL
    )

    pred_mt = (
        pred == MT_LABEL
    )

    dice = dice_score(
        pred_mt,
        gt_mt,
    )

    # -------------------------------------------------------------
    # Pick the slice containing the MOST ground-truth MT.
    # This is much better than blindly showing the center slice.
    # -------------------------------------------------------------

    mt_per_slice = (
        gt_mt.sum(
            axis=(1, 2)
        )
    )

    slice_index = int(
        np.argmax(
            mt_per_slice
        )
    )

    image = normalize_for_display(
        scan[slice_index]
    )

    gt_slice = (
        gt_mt[slice_index]
    )

    pred_slice = (
        pred_mt[slice_index]
    )

    # Flip vertically purely for intuitive display.
    image = np.flipud(image)
    gt_slice = np.flipud(gt_slice)
    pred_slice = np.flipud(pred_slice)

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    # 1. Original CBCT
    axes[0].imshow(
        image,
        cmap="gray"
    )

    axes[0].set_title(
        "CBCT"
    )

    # 2. Ground truth
    axes[1].imshow(
        image,
        cmap="gray"
    )

    axes[1].imshow(
        np.ma.masked_where(
            ~gt_slice,
            gt_slice
        ),
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Ground Truth MT"
    )

    # 3. Prediction
    axes[2].imshow(
        image,
        cmap="gray"
    )

    axes[2].imshow(
        np.ma.masked_where(
            ~pred_slice,
            pred_slice
        ),
        cmap="Blues",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "nnU-Net Prediction"
    )

    # 4. Overlay
    axes[3].imshow(
        image,
        cmap="gray"
    )

    axes[3].imshow(
        np.ma.masked_where(
            ~gt_slice,
            gt_slice
        ),
        cmap="Reds",
        alpha=0.50,
        vmin=0,
        vmax=1
    )

    axes[3].imshow(
        np.ma.masked_where(
            ~pred_slice,
            pred_slice
        ),
        cmap="Blues",
        alpha=0.50,
        vmin=0,
        vmax=1
    )

    axes[3].set_title(
        f"Overlay\nDice = {dice:.3f}"
    )

    for ax in axes:
        ax.axis("off")

    legend_elements = [
        Patch(
            facecolor="red",
            alpha=0.5,
            label="Ground Truth"
        ),
        Patch(
            facecolor="blue",
            alpha=0.5,
            label="Prediction"
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2
    )

    fig.suptitle(
        f"{case_id} | MT Segmentation",
        fontsize=14
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 0.93]
    )

    output_path = (
        OUT_DIR /
        f"{case_id}_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Dice = {dice:.4f}"
    )

    overview_data.append(
        {
            "case": case_id,
            "image": image,
            "gt": gt_slice,
            "pred": pred_slice,
            "dice": dice
        }
    )


# -------------------------------------------------------------------------
# BIG OVERVIEW FIGURE
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    len(overview_data),
    3,
    figsize=(
        10,
        3 * len(overview_data)
    )
)

for row, item in enumerate(
    overview_data
):

    image = item["image"]
    gt_slice = item["gt"]
    pred_slice = item["pred"]

    # CBCT
    axes[row, 0].imshow(
        image,
        cmap="gray"
    )

    # GT
    axes[row, 1].imshow(
        image,
        cmap="gray"
    )

    axes[row, 1].imshow(
        np.ma.masked_where(
            ~gt_slice,
            gt_slice
        ),
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    # Prediction
    axes[row, 2].imshow(
        image,
        cmap="gray"
    )

    axes[row, 2].imshow(
        np.ma.masked_where(
            ~pred_slice,
            pred_slice
        ),
        cmap="Blues",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    axes[row, 0].set_ylabel(
        f"{item['case']}\nDice {item['dice']:.3f}",
        fontsize=9
    )

    for col in range(3):
        axes[row, col].axis("off")


axes[0, 0].set_title(
    "CBCT",
    fontsize=13
)

axes[0, 1].set_title(
    "Ground Truth MT",
    fontsize=13
)

axes[0, 2].set_title(
    "nnU-Net Prediction",
    fontsize=13
)


plt.suptitle(
    "nnU-Net Mucosal Thickening Segmentation — Validation Cases",
    fontsize=16
)

plt.tight_layout(
    rect=[0, 0, 1, 0.98]
)

overview_path = (
    OUT_DIR /
    "ALL_CASES_overview.png"
)

plt.savefig(
    overview_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------------

dice_values = [
    x["dice"]
    for x in overview_data
]

print()
print("=" * 70)

print(
    "Mean per-case MT Dice:",
    f"{np.mean(dice_values):.4f}"
)

print(
    "Median MT Dice:",
    f"{np.median(dice_values):.4f}"
)

print(
    "Min MT Dice:",
    f"{np.min(dice_values):.4f}"
)

print(
    "Max MT Dice:",
    f"{np.max(dice_values):.4f}"
)

print("=" * 70)

print()
print(
    "Figures saved to:"
)

print(
    OUT_DIR
)

print()
print(
    "Main overview:"
)

print(
    overview_path
)