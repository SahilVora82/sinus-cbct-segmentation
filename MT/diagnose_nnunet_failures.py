from pathlib import Path
import csv

import numpy as np
import nrrd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

SCAN_DIR = BASE / "data" / "scan_valid"
GT_DIR = BASE / "data" / "segment_valid"

NNUNET_RESULTS = ROOT / "nnUNet_work" / "results"

OUT_DIR = ROOT / "nnUNet_failure_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SETTINGS
# =============================================================================

MT_LABEL = 1

# Analyze this many worst validation cases in detail.
N_WORST_CASES = 5

# Show this many slices per bad case.
N_SLICES = 10


# =============================================================================
# HELPERS
# =============================================================================

def find_prediction(case_id):

    candidates = list(
        NNUNET_RESULTS.rglob(
            f"{case_id}.nrrd"
        )
    )

    validation_candidates = [
        p for p in candidates
        if "validation" in str(p).lower()
    ]

    if validation_candidates:
        return validation_candidates[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Prediction not found for {case_id}"
    )


def normalize_for_display(image):

    low = np.percentile(image, 1)
    high = np.percentile(image, 99)

    image = np.clip(
        image,
        low,
        high,
    )

    return (
        image - low
    ) / (
        high - low + 1e-8
    )


def metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(
        pred,
        gt
    ).sum()

    fp = np.logical_and(
        pred,
        ~gt
    ).sum()

    fn = np.logical_and(
        ~pred,
        gt
    ).sum()

    pred_count = pred.sum()
    gt_count = gt.sum()

    denom = (
        pred_count
        + gt_count
    )

    dice = (
        2 * tp / denom
        if denom > 0
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
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "gt_voxels": int(gt_count),
        "pred_voxels": int(pred_count),
        "volume_ratio": (
            float(pred_count / gt_count)
            if gt_count > 0
            else 0.0
        ),
    }


# =============================================================================
# LOAD ALL CASES
# =============================================================================

case_ids = sorted(
    p.stem
    for p in SCAN_DIR.glob("*.nrrd")
)

cases = []

print()
print("=" * 80)
print("NNUNET VALIDATION FAILURE ANALYSIS")
print("=" * 80)

for case_id in case_ids:

    scan, _ = nrrd.read(
        str(
            SCAN_DIR
            / f"{case_id}.nrrd"
        )
    )

    gt, _ = nrrd.read(
        str(
            GT_DIR
            / f"{case_id}_GT.nrrd"
        )
    )

    pred_path = find_prediction(
        case_id
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
            f"{case_id}: pred/GT mismatch "
            f"{pred.shape} vs {gt.shape}"
        )

    gt_mt = (
        gt == MT_LABEL
    )

    pred_mt = (
        pred == MT_LABEL
    )

    result = metrics(
        pred_mt,
        gt_mt,
    )

    cases.append(
        {
            "case": case_id,
            "scan": scan,
            "gt": gt_mt,
            "pred": pred_mt,
            **result,
        }
    )


# =============================================================================
# RANK WORST -> BEST
# =============================================================================

cases.sort(
    key=lambda x: x["dice"]
)

print()
print("3D MT PERFORMANCE — WORST TO BEST")
print("-" * 80)

for item in cases:

    print(
        f"{item['case']:<20} "
        f"Dice={item['dice']:.4f}  "
        f"P={item['precision']:.4f}  "
        f"R={item['recall']:.4f}  "
        f"FP={item['fp']:>7}  "
        f"FN={item['fn']:>7}  "
        f"Pred/GT={item['volume_ratio']:.2f}x"
    )


# =============================================================================
# SAVE CASE SUMMARY
# =============================================================================

with open(
    OUT_DIR / "case_summary.csv",
    "w",
    newline=""
) as f:

    fields = [
        "case",
        "dice",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "gt_voxels",
        "pred_voxels",
        "volume_ratio",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for item in cases:

        writer.writerow(
            {
                key: item[key]
                for key in fields
            }
        )


# =============================================================================
# ANALYZE WORST CASES
# =============================================================================

worst_cases = cases[
    :min(
        N_WORST_CASES,
        len(cases)
    )
]


for item in worst_cases:

    case_id = item["case"]
    scan = item["scan"]
    gt = item["gt"]
    pred = item["pred"]

    print()
    print("=" * 80)
    print(
        f"DETAILED CASE: {case_id}"
    )
    print(
        f"3D Dice = {item['dice']:.4f}"
    )
    print(
        f"Precision = {item['precision']:.4f}"
    )
    print(
        f"Recall = {item['recall']:.4f}"
    )
    print(
        f"False positives = {item['fp']}"
    )
    print(
        f"False negatives = {item['fn']}"
    )
    print(
        f"Prediction/GT volume = "
        f"{item['volume_ratio']:.2f}x"
    )


    # =========================================================================
    # PER-SLICE METRICS
    # =========================================================================

    slice_rows = []

    for z in range(
        scan.shape[0]
    ):

        gt_slice = gt[z]
        pred_slice = pred[z]

        slice_metric = metrics(
            pred_slice,
            gt_slice
        )

        disagreement = (
            slice_metric["fp"]
            + slice_metric["fn"]
        )

        slice_rows.append(
            {
                "slice": z,
                **slice_metric,
                "disagreement": disagreement,
            }
        )


    # Save every slice's numbers.
    slice_csv = (
        OUT_DIR
        / f"{case_id}_slice_metrics.csv"
    )

    with open(
        slice_csv,
        "w",
        newline=""
    ) as f:

        fields = [
            "slice",
            "dice",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "gt_voxels",
            "pred_voxels",
            "volume_ratio",
            "disagreement",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            slice_rows
        )


    # =========================================================================
    # CHOOSE MOST INFORMATIVE SLICES
    #
    # Rather than picking the prettiest slices, choose slices with the
    # greatest number of incorrect voxels.
    # =========================================================================

    active_rows = [
        row
        for row in slice_rows
        if (
            row["gt_voxels"] > 0
            or row["pred_voxels"] > 0
        )
    ]

    active_rows.sort(
        key=lambda x:
            x["disagreement"],
        reverse=True
    )

    selected = active_rows[
        :min(
            N_SLICES,
            len(active_rows)
        )
    ]

    # Put them back into anatomical order.
    selected.sort(
        key=lambda x:
            x["slice"]
    )


    # =========================================================================
    # FIGURE
    #
    # Column 1: CBCT
    # Column 2: Ground truth
    # Column 3: Prediction
    # Column 4: Error map
    #
    # Error colors:
    # green  = correct MT
    # red    = missed MT
    # yellow = false positive MT
    # =========================================================================

    n_rows = len(selected)

    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(
            12,
            3 * n_rows
        )
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]


    for row_index, row in enumerate(
        selected
    ):

        z = row["slice"]

        image = normalize_for_display(
            scan[z]
        )

        gt_slice = gt[z]
        pred_slice = pred[z]

        correct = np.logical_and(
            gt_slice,
            pred_slice
        )

        missed = np.logical_and(
            gt_slice,
            ~pred_slice
        )

        false_positive = np.logical_and(
            ~gt_slice,
            pred_slice
        )


        # ---------------------------------------------------------
        # Flip for display
        # ---------------------------------------------------------

        image = np.flipud(image)

        gt_slice = np.flipud(
            gt_slice
        )

        pred_slice = np.flipud(
            pred_slice
        )

        correct = np.flipud(
            correct
        )

        missed = np.flipud(
            missed
        )

        false_positive = np.flipud(
            false_positive
        )


        # ---------------------------------------------------------
        # CBCT
        # ---------------------------------------------------------

        axes[row_index, 0].imshow(
            image,
            cmap="gray"
        )


        # ---------------------------------------------------------
        # GT
        # ---------------------------------------------------------

        axes[row_index, 1].imshow(
            image,
            cmap="gray"
        )

        axes[row_index, 1].imshow(
            np.ma.masked_where(
                ~gt_slice,
                gt_slice
            ),
            cmap="Reds",
            alpha=0.65,
            vmin=0,
            vmax=1
        )


        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------

        axes[row_index, 2].imshow(
            image,
            cmap="gray"
        )

        axes[row_index, 2].imshow(
            np.ma.masked_where(
                ~pred_slice,
                pred_slice
            ),
            cmap="Blues",
            alpha=0.65,
            vmin=0,
            vmax=1
        )


        # ---------------------------------------------------------
        # Diagnostic error map
        # ---------------------------------------------------------

        axes[row_index, 3].imshow(
            image,
            cmap="gray"
        )

        # Correct overlap = green
        axes[row_index, 3].imshow(
            np.ma.masked_where(
                ~correct,
                correct
            ),
            cmap="Greens",
            alpha=0.75,
            vmin=0,
            vmax=1
        )

        # Missed GT = red
        axes[row_index, 3].imshow(
            np.ma.masked_where(
                ~missed,
                missed
            ),
            cmap="Reds",
            alpha=0.80,
            vmin=0,
            vmax=1
        )

        # False positive = yellow/orange
        axes[row_index, 3].imshow(
            np.ma.masked_where(
                ~false_positive,
                false_positive
            ),
            cmap="autumn",
            alpha=0.80,
            vmin=0,
            vmax=1
        )


        axes[row_index, 0].set_ylabel(
            f"Slice {z}\n"
            f"2D Dice {row['dice']:.3f}\n"
            f"FP {row['fp']} | FN {row['fn']}",
            fontsize=9
        )

        for column in range(4):

            axes[row_index, column].axis(
                "off"
            )


    axes[0, 0].set_title(
        "CBCT"
    )

    axes[0, 1].set_title(
        "Ground Truth MT"
    )

    axes[0, 2].set_title(
        "nnU-Net Prediction"
    )

    axes[0, 3].set_title(
        "Errors\n"
        "Green=Correct | Red=Missed | Yellow=False +"
    )


    fig.suptitle(
        f"{case_id} — "
        f"3D MT Dice {item['dice']:.3f} | "
        f"Precision {item['precision']:.3f} | "
        f"Recall {item['recall']:.3f}",
        fontsize=14
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97
        ]
    )


    output_path = (
        OUT_DIR
        / f"{case_id}_WORST_SLICES.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

dice_values = [
    item["dice"]
    for item in cases
]

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Mean per-case MT Dice: "
    f"{np.mean(dice_values):.4f}"
)

print(
    f"Median MT Dice: "
    f"{np.median(dice_values):.4f}"
)

print(
    f"Worst case: "
    f"{cases[0]['case']} "
    f"({cases[0]['dice']:.4f})"
)

print(
    f"Best case: "
    f"{cases[-1]['case']} "
    f"({cases[-1]['dice']:.4f})"
)

print()
print(
    "Results saved to:"
)

print(
    OUT_DIR
)

print()
print(
    "TEST DATA WERE NEVER LOADED."
)