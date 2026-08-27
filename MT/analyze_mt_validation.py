from pathlib import Path

import numpy as np
import nrrd
import torch
import matplotlib.pyplot as plt

from unet import UNet


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SCAN_DIR = BASE_DIR / "data" / "scan_valid"
MASK_DIR = BASE_DIR / "data" / "segment_valid"

CHECKPOINT = (
    BASE_DIR
    / "results_hybrid"
    / "best_mt_model.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "validation_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MEAN = -46.1730322190273
STD = 293.1394271328278

MT_CLASS = 1
AIR_CLASS = 2


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

print("Device:", device)


# ============================================================
# MODEL
# ============================================================

model = UNet(
    [32, 64, 128, 256]
).to(device)


def load_state_dict(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        return checkpoint["model"]

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    if isinstance(checkpoint, dict):
        return checkpoint

    raise RuntimeError(
        "Could not find model weights."
    )


state = load_state_dict(
    CHECKPOINT
)

model.load_state_dict(
    state,
    strict=True,
)

model.eval()

print(
    "Loaded:",
    CHECKPOINT,
)


# ============================================================
# METRICS
# ============================================================

def metrics(
    prediction,
    target,
    class_id,
):
    p = prediction == class_id
    t = target == class_id

    tp = np.logical_and(
        p,
        t,
    ).sum()

    fp = np.logical_and(
        p,
        np.logical_not(t),
    ).sum()

    fn = np.logical_and(
        np.logical_not(p),
        t,
    ).sum()

    pred_n = p.sum()
    target_n = t.sum()

    denom = (
        pred_n
        + target_n
    )

    if denom == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * tp
            / denom
        )

    union = (
        tp
        + fp
        + fn
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
        "gt_voxels": int(target_n),
        "pred_voxels": int(pred_n),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def normalize_for_display(image):
    low = np.percentile(
        image,
        2,
    )

    high = np.percentile(
        image,
        98,
    )

    if high <= low:
        return image

    image = np.clip(
        image,
        low,
        high,
    )

    return (
        image - low
    ) / (
        high - low
    )


def slice_with_most_mt(
    mask,
    axis,
):
    mt = (
        mask == MT_CLASS
    )

    other_axes = tuple(
        x
        for x in range(3)
        if x != axis
    )

    counts = mt.sum(
        axis=other_axes
    )

    return int(
        np.argmax(counts)
    )


def get_slice(
    volume,
    axis,
    index,
):
    if axis == 0:
        result = volume[
            index,
            :,
            :
        ]

    elif axis == 1:
        result = volume[
            :,
            index,
            :
        ]

    else:
        result = volume[
            :,
            :,
            index
        ]

    return np.rot90(
        result
    )


def save_case_figure(
    case_name,
    scan,
    target,
    prediction,
    mt_stats,
):
    orientation_names = [
        "Depth",
        "Height",
        "Width",
    ]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14, 11),
    )

    for axis in range(3):

        index = slice_with_most_mt(
            target,
            axis,
        )

        scan_slice = get_slice(
            scan,
            axis,
            index,
        )

        target_slice = get_slice(
            target,
            axis,
            index,
        )

        pred_slice = get_slice(
            prediction,
            axis,
            index,
        )

        scan_display = normalize_for_display(
            scan_slice
        )

        gt_mt = (
            target_slice == MT_CLASS
        )

        pred_mt = (
            pred_slice == MT_CLASS
        )

        tp = (
            gt_mt
            & pred_mt
        )

        fp = (
            (~gt_mt)
            & pred_mt
        )

        fn = (
            gt_mt
            & (~pred_mt)
        )

        error_map = np.zeros(
            gt_mt.shape,
            dtype=np.uint8,
        )

        error_map[tp] = 1
        error_map[fp] = 2
        error_map[fn] = 3

        # --------------------------------
        # Raw scan
        # --------------------------------

        axes[axis, 0].imshow(
            scan_display,
            cmap="gray",
        )

        axes[axis, 0].set_title(
            f"{orientation_names[axis]} scan"
        )

        # --------------------------------
        # Ground truth
        # --------------------------------

        axes[axis, 1].imshow(
            scan_display,
            cmap="gray",
        )

        overlay = np.ma.masked_where(
            ~gt_mt,
            gt_mt,
        )

        axes[axis, 1].imshow(
            overlay,
            cmap="autumn",
            alpha=0.55,
        )

        axes[axis, 1].set_title(
            "Ground-truth MT"
        )

        # --------------------------------
        # Prediction
        # --------------------------------

        axes[axis, 2].imshow(
            scan_display,
            cmap="gray",
        )

        overlay = np.ma.masked_where(
            ~pred_mt,
            pred_mt,
        )

        axes[axis, 2].imshow(
            overlay,
            cmap="winter",
            alpha=0.55,
        )

        axes[axis, 2].set_title(
            "Predicted MT"
        )

        # --------------------------------
        # Error map
        # --------------------------------

        axes[axis, 3].imshow(
            scan_display,
            cmap="gray",
        )

        error_overlay = np.ma.masked_where(
            error_map == 0,
            error_map,
        )

        axes[axis, 3].imshow(
            error_overlay,
            cmap="viridis",
            alpha=0.65,
            vmin=1,
            vmax=3,
        )

        axes[axis, 3].set_title(
            "MT overlap / errors"
        )

        for col in range(4):
            axes[axis, col].axis(
                "off"
            )

    fig.suptitle(
        (
            f"{case_name}\n"
            f"MT Dice={mt_stats['dice']:.3f}   "
            f"Precision={mt_stats['precision']:.3f}   "
            f"Recall={mt_stats['recall']:.3f}"
        ),
        fontsize=15,
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{case_name}_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# RUN VALIDATION
# ============================================================

scan_paths = sorted(
    SCAN_DIR.glob(
        "*.nrrd"
    )
)

rows = []

print()
print(
    "=" * 90
)
print(
    "PER-CASE VALIDATION"
)
print(
    "=" * 90
)

for scan_path in scan_paths:

    case_name = scan_path.stem

    mask_path = (
        MASK_DIR
        / f"{case_name}_GT.nrrd"
    )

    scan, scan_header = nrrd.read(
        str(scan_path)
    )

    target, target_header = nrrd.read(
        str(mask_path)
    )

    scan = scan.astype(
        np.float32
    )

    target = target.astype(
        np.int64
    )

    normalized = (
        scan - MEAN
    ) / STD

    tensor = torch.from_numpy(
        normalized
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(
            device,
            dtype=torch.float32,
        )
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        prediction = (
            torch.argmax(
                logits,
                dim=1,
            )
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

    mt_stats = metrics(
        prediction,
        target,
        MT_CLASS,
    )

    air_stats = metrics(
        prediction,
        target,
        AIR_CLASS,
    )

    rows.append(
        {
            "case": case_name,

            "mt_dice":
                mt_stats["dice"],

            "mt_iou":
                mt_stats["iou"],

            "mt_precision":
                mt_stats["precision"],

            "mt_recall":
                mt_stats["recall"],

            "gt_mt_voxels":
                mt_stats["gt_voxels"],

            "pred_mt_voxels":
                mt_stats["pred_voxels"],

            "mt_false_positive_voxels":
                mt_stats["fp"],

            "mt_false_negative_voxels":
                mt_stats["fn"],

            "air_dice":
                air_stats["dice"],
        }
    )

    prediction_path = (
        PREDICTION_DIR
        / f"{case_name}_PRED.nrrd"
    )

    nrrd.write(
        str(prediction_path),
        prediction,
        header=target_header,
    )

    save_case_figure(
        case_name,
        scan,
        target,
        prediction,
        mt_stats,
    )

    print(
        f"{case_name:20s} "
        f"MT Dice={mt_stats['dice']:.4f}  "
        f"P={mt_stats['precision']:.4f}  "
        f"R={mt_stats['recall']:.4f}  "
        f"Air={air_stats['dice']:.4f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

import csv

csv_path = (
    OUTPUT_DIR
    / "validation_metrics.csv"
)

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

rows_sorted = sorted(
    rows,
    key=lambda x: x["mt_dice"],
)

mt_dices = [
    row["mt_dice"]
    for row in rows
]

air_dices = [
    row["air_dice"]
    for row in rows
]

print()
print(
    "=" * 90
)
print(
    "SUMMARY"
)
print(
    "=" * 90
)

print(
    f"Mean per-case MT Dice: "
    f"{np.mean(mt_dices):.4f}"
)

print(
    f"Median MT Dice: "
    f"{np.median(mt_dices):.4f}"
)

print(
    f"Minimum MT Dice: "
    f"{np.min(mt_dices):.4f}"
)

print(
    f"Maximum MT Dice: "
    f"{np.max(mt_dices):.4f}"
)

print(
    f"Mean Air Dice: "
    f"{np.mean(air_dices):.4f}"
)

print()
print(
    "WORST CASES:"
)

for row in rows_sorted[:3]:

    print(
        f"  {row['case']}: "
        f"Dice={row['mt_dice']:.4f}, "
        f"P={row['mt_precision']:.4f}, "
        f"R={row['mt_recall']:.4f}"
    )

print()
print(
    "BEST CASES:"
)

for row in rows_sorted[-3:]:

    print(
        f"  {row['case']}: "
        f"Dice={row['mt_dice']:.4f}, "
        f"P={row['mt_precision']:.4f}, "
        f"R={row['mt_recall']:.4f}"
    )


# ============================================================
# BAR CHART
# ============================================================

chart_rows = sorted(
    rows,
    key=lambda x: x["mt_dice"],
)

names = [
    row["case"]
    for row in chart_rows
]

values = [
    row["mt_dice"]
    for row in chart_rows
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.barh(
    names,
    values,
)

ax.axvline(
    np.mean(values),
    linestyle="--",
)

ax.set_xlim(
    0,
    1,
)

ax.set_xlabel(
    "MT Dice"
)

ax.set_title(
    "Validation MT Dice by Scan"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "mt_dice_by_case.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# MEETING MONTAGE
# ============================================================

selected = [
    chart_rows[0],
    chart_rows[len(chart_rows) // 2],
    chart_rows[-1],
]

labels = [
    "Worst validation case",
    "Median validation case",
    "Best validation case",
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(11, 11),
)

for row_index, (
    row,
    label,
) in enumerate(
    zip(
        selected,
        labels,
    )
):

    case_name = row[
        "case"
    ]

    scan, _ = nrrd.read(
        str(
            SCAN_DIR
            / f"{case_name}.nrrd"
        )
    )

    target, _ = nrrd.read(
        str(
            MASK_DIR
            / f"{case_name}_GT.nrrd"
        )
    )

    prediction, _ = nrrd.read(
        str(
            PREDICTION_DIR
            / f"{case_name}_PRED.nrrd"
        )
    )

    z = slice_with_most_mt(
        target,
        axis=0,
    )

    scan_slice = normalize_for_display(
        get_slice(
            scan,
            0,
            z,
        )
    )

    gt_mt = (
        get_slice(
            target,
            0,
            z,
        )
        == MT_CLASS
    )

    pred_mt = (
        get_slice(
            prediction,
            0,
            z,
        )
        == MT_CLASS
    )

    axes[row_index, 0].imshow(
        scan_slice,
        cmap="gray",
    )

    axes[row_index, 0].set_title(
        f"{label}\n{case_name}"
    )

    axes[row_index, 1].imshow(
        scan_slice,
        cmap="gray",
    )

    axes[row_index, 1].imshow(
        np.ma.masked_where(
            ~gt_mt,
            gt_mt,
        ),
        cmap="autumn",
        alpha=0.55,
    )

    axes[row_index, 1].set_title(
        "Ground truth"
    )

    axes[row_index, 2].imshow(
        scan_slice,
        cmap="gray",
    )

    axes[row_index, 2].imshow(
        np.ma.masked_where(
            ~pred_mt,
            pred_mt,
        ),
        cmap="winter",
        alpha=0.55,
    )

    axes[row_index, 2].set_title(
        (
            f"Prediction\n"
            f"Dice={row['mt_dice']:.3f}"
        )
    )

    for col in range(3):
        axes[row_index, col].axis(
            "off"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "meeting_montage.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


print()
print(
    "DONE"
)

print(
    "CSV:",
    csv_path,
)

print(
    "Figures:",
    FIGURE_DIR,
)

print(
    "Meeting montage:",
    OUTPUT_DIR
    / "meeting_montage.png",
)

print(
    "Dice chart:",
    OUTPUT_DIR
    / "mt_dice_by_case.png",
)