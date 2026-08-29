from pathlib import Path
import csv

import numpy as np
import nrrd
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

SCAN_DIR = BASE / "data" / "scan_valid"
GT_DIR = BASE / "data" / "segment_valid"

NNUNET_RESULTS = (
    ROOT
    / "nnUNet_work"
    / "results"
)

OUT_DIR = (
    ROOT
    / "nnUNet_air_distance_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LABELS
# =============================================================================

BG = 0
MT = 1
AIR = 2


# =============================================================================
# DISTANCES TO TEST
#
# These are in mm when NRRD spacing can be read.
# Otherwise they behave as voxels.
# =============================================================================

DISTANCES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    12.0,
]

N_WORST_VISUALS = 5
N_SLICES_PER_CASE = 8


# =============================================================================
# FIND NNUNET PREDICTION
# =============================================================================

def find_prediction(case_id):

    candidates = list(
        NNUNET_RESULTS.rglob(
            f"{case_id}.nrrd"
        )
    )

    validation_candidates = [
        path
        for path in candidates
        if "validation" in str(path).lower()
    ]

    if validation_candidates:
        return validation_candidates[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No prediction found for {case_id}"
    )


# =============================================================================
# SPACING
# =============================================================================

def get_spacing(header):

    try:

        directions = header.get(
            "space directions",
            None
        )

        if directions is not None:

            spacing = []

            for direction in directions:

                direction = np.asarray(
                    direction,
                    dtype=float
                )

                spacing.append(
                    float(
                        np.linalg.norm(
                            direction
                        )
                    )
                )

            if (
                len(spacing) == 3
                and
                all(
                    np.isfinite(spacing)
                )
            ):
                return tuple(spacing)

    except Exception:
        pass

    return (1.0, 1.0, 1.0)


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    prediction,
    target,
):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target
    ).sum()

    pred_count = prediction.sum()
    target_count = target.sum()

    denominator = (
        pred_count
        + target_count
    )

    dice = (
        2.0 * tp / denominator
        if denominator > 0
        else 1.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "pred_voxels": int(pred_count),
        "gt_voxels": int(target_count),
    }


# =============================================================================
# DISPLAY
# =============================================================================

def normalize_image(image):

    low = np.percentile(
        image,
        1
    )

    high = np.percentile(
        image,
        99
    )

    image = np.clip(
        image,
        low,
        high
    )

    return (
        image - low
    ) / (
        high - low + 1e-8
    )


# =============================================================================
# LOAD CASES
# =============================================================================

case_ids = sorted(
    path.stem
    for path
    in SCAN_DIR.glob("*.nrrd")
)

cases = []

print()
print("=" * 80)
print("AIR-DISTANCE MT POSTPROCESSING")
print("=" * 80)

print(
    f"Validation cases: {len(case_ids)}"
)

print(
    "TEST DATA LOADED: NO"
)


for case_id in case_ids:

    scan, scan_header = nrrd.read(
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

    if (
        scan.shape != gt.shape
        or
        pred.shape != gt.shape
    ):
        raise RuntimeError(
            f"{case_id}: shape mismatch. "
            f"scan={scan.shape}, "
            f"GT={gt.shape}, "
            f"pred={pred.shape}"
        )

    spacing = get_spacing(
        scan_header
    )

    gt_mt = (
        gt == MT
    )

    pred_mt = (
        pred == MT
    )

    pred_air = (
        pred == AIR
    )

    # -------------------------------------------------------------
    # Distance of EVERY voxel to nearest predicted AIR voxel.
    #
    # AIR itself = distance 0.
    # As you move outward from air, distance increases.
    # -------------------------------------------------------------

    distance_from_air = (
        distance_transform_edt(
            ~pred_air,
            sampling=spacing
        )
    )

    original_metrics = (
        calculate_metrics(
            pred_mt,
            gt_mt
        )
    )

    cases.append(
        {
            "case": case_id,
            "scan": scan,
            "gt_mt": gt_mt,
            "pred_mt": pred_mt,
            "pred_air": pred_air,
            "distance_from_air":
                distance_from_air,
            "spacing": spacing,
            "original":
                original_metrics,
        }
    )

    print(
        f"{case_id:<20} "
        f"Original Dice="
        f"{original_metrics['dice']:.4f} "
        f"P={original_metrics['precision']:.4f} "
        f"R={original_metrics['recall']:.4f}"
    )


# =============================================================================
# ORIGINAL MEAN
# =============================================================================

original_mean_dice = np.mean(
    [
        case["original"]["dice"]
        for case in cases
    ]
)

original_mean_precision = np.mean(
    [
        case["original"]["precision"]
        for case in cases
    ]
)

original_mean_recall = np.mean(
    [
        case["original"]["recall"]
        for case in cases
    ]
)


print()
print("=" * 80)
print("ORIGINAL NNUNET MT")
print("=" * 80)

print(
    f"Mean Dice:      "
    f"{original_mean_dice:.4f}"
)

print(
    f"Mean Precision: "
    f"{original_mean_precision:.4f}"
)

print(
    f"Mean Recall:    "
    f"{original_mean_recall:.4f}"
)


# =============================================================================
# DISTANCE SWEEP
# =============================================================================

sweep_results = []

print()
print("=" * 80)
print("DISTANCE SWEEP")
print("=" * 80)

print()
print(
    "Maximum distance from predicted AIR"
)

print("-" * 80)


for maximum_distance in DISTANCES:

    case_metrics = []

    for case in cases:

        # ---------------------------------------------------------
        # Keep predicted MT only if it lies close enough
        # to the predicted AIR cavity.
        # ---------------------------------------------------------

        filtered_mt = (
            case["pred_mt"]
            &
            (
                case["distance_from_air"]
                <= maximum_distance
            )
        )

        metrics = calculate_metrics(
            filtered_mt,
            case["gt_mt"]
        )

        case_metrics.append(
            metrics
        )


    mean_dice = float(
        np.mean(
            [
                metric["dice"]
                for metric
                in case_metrics
            ]
        )
    )

    mean_precision = float(
        np.mean(
            [
                metric["precision"]
                for metric
                in case_metrics
            ]
        )
    )

    mean_recall = float(
        np.mean(
            [
                metric["recall"]
                for metric
                in case_metrics
            ]
        )
    )


    sweep_results.append(
        {
            "distance":
                maximum_distance,

            "mean_dice":
                mean_dice,

            "mean_precision":
                mean_precision,

            "mean_recall":
                mean_recall,

            "case_metrics":
                case_metrics,
        }
    )


    print(
        f"{maximum_distance:>5.1f} mm | "
        f"Dice={mean_dice:.4f} | "
        f"P={mean_precision:.4f} | "
        f"R={mean_recall:.4f} | "
        f"change="
        f"{mean_dice - original_mean_dice:+.4f}"
    )


# =============================================================================
# BEST GLOBAL DISTANCE
# =============================================================================

best_result = max(
    sweep_results,
    key=lambda item:
        item["mean_dice"]
)

best_distance = (
    best_result["distance"]
)


print()
print("=" * 80)
print("BEST GLOBAL AIR-DISTANCE CONSTRAINT")
print("=" * 80)

print(
    f"Maximum MT distance from air: "
    f"{best_distance:.1f} mm"
)

print()

print(
    f"Original mean Dice: "
    f"{original_mean_dice:.4f}"
)

print(
    f"Filtered mean Dice: "
    f"{best_result['mean_dice']:.4f}"
)

print(
    f"Change: "
    f"{best_result['mean_dice'] - original_mean_dice:+.4f}"
)

print()

print(
    f"Original precision: "
    f"{original_mean_precision:.4f}"
)

print(
    f"Filtered precision: "
    f"{best_result['mean_precision']:.4f}"
)

print()

print(
    f"Original recall: "
    f"{original_mean_recall:.4f}"
)

print(
    f"Filtered recall: "
    f"{best_result['mean_recall']:.4f}"
)


# =============================================================================
# PER-CASE BEFORE / AFTER
# =============================================================================

print()
print("=" * 80)
print("PER-CASE RESULTS")
print("=" * 80)


per_case_rows = []


for case, filtered_metrics in zip(
    cases,
    best_result["case_metrics"]
):

    original = case["original"]

    change = (
        filtered_metrics["dice"]
        - original["dice"]
    )

    per_case_rows.append(
        {
            "case":
                case["case"],

            "original_dice":
                original["dice"],

            "filtered_dice":
                filtered_metrics["dice"],

            "change":
                change,

            "original_precision":
                original["precision"],

            "filtered_precision":
                filtered_metrics["precision"],

            "original_recall":
                original["recall"],

            "filtered_recall":
                filtered_metrics["recall"],
        }
    )


per_case_rows.sort(
    key=lambda row:
        row["original_dice"]
)


for row in per_case_rows:

    print(
        f"{row['case']:<20} "
        f"{row['original_dice']:.4f} "
        f"-> "
        f"{row['filtered_dice']:.4f} "
        f"({row['change']:+.4f})   "
        f"P "
        f"{row['original_precision']:.3f}"
        f"->{row['filtered_precision']:.3f}   "
        f"R "
        f"{row['original_recall']:.3f}"
        f"->{row['filtered_recall']:.3f}"
    )


# =============================================================================
# SAVE CSV
# =============================================================================

with open(
    OUT_DIR
    / "distance_sweep.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "max_distance_mm",
            "mean_dice",
            "mean_precision",
            "mean_recall",
            "change_vs_original",
        ]
    )

    for result in sweep_results:

        writer.writerow(
            [
                result["distance"],
                result["mean_dice"],
                result["mean_precision"],
                result["mean_recall"],
                (
                    result["mean_dice"]
                    - original_mean_dice
                ),
            ]
        )


with open(
    OUT_DIR
    / "best_distance_per_case.csv",
    "w",
    newline=""
) as file:

    fields = list(
        per_case_rows[0].keys()
    )

    writer = csv.DictWriter(
        file,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(
        per_case_rows
    )


# =============================================================================
# VISUALIZE WORST ORIGINAL CASES
# =============================================================================

worst_cases = sorted(
    cases,
    key=lambda item:
        item["original"]["dice"]
)[
    :N_WORST_VISUALS
]


for case in worst_cases:

    case_id = case["case"]

    filtered_mt = (
        case["pred_mt"]
        &
        (
            case["distance_from_air"]
            <= best_distance
        )
    )

    filtered_metrics = (
        calculate_metrics(
            filtered_mt,
            case["gt_mt"]
        )
    )


    # -------------------------------------------------------------------------
    # Find slices where ORIGINAL prediction makes most mistakes
    # -------------------------------------------------------------------------

    slice_scores = []

    for z in range(
        case["scan"].shape[0]
    ):

        gt_slice = (
            case["gt_mt"][z]
        )

        pred_slice = (
            case["pred_mt"][z]
        )

        fp = np.logical_and(
            pred_slice,
            ~gt_slice
        ).sum()

        fn = np.logical_and(
            ~pred_slice,
            gt_slice
        ).sum()

        if (
            gt_slice.sum() > 0
            or
            pred_slice.sum() > 0
        ):

            slice_scores.append(
                (
                    z,
                    int(fp + fn)
                )
            )


    slice_scores.sort(
        key=lambda item:
            item[1],
        reverse=True
    )

    selected_slices = [
        z
        for z, _
        in slice_scores[
            :N_SLICES_PER_CASE
        ]
    ]

    selected_slices.sort()


    # =========================================================================
    # FIGURE
    # =========================================================================

    fig, axes = plt.subplots(
        len(selected_slices),
        5,
        figsize=(
            15,
            3 * len(selected_slices)
        )
    )

    if len(selected_slices) == 1:
        axes = axes[np.newaxis, :]


    for row, z in enumerate(
        selected_slices
    ):

        image = normalize_image(
            case["scan"][z]
        )

        gt = case["gt_mt"][z]

        air = case["pred_air"][z]

        original = case["pred_mt"][z]

        filtered = filtered_mt[z]


        image = np.flipud(
            image
        )

        gt = np.flipud(
            gt
        )

        air = np.flipud(
            air
        )

        original = np.flipud(
            original
        )

        filtered = np.flipud(
            filtered
        )


        # ---------------------------------------------------------------------
        # CBCT
        # ---------------------------------------------------------------------

        axes[row, 0].imshow(
            image,
            cmap="gray"
        )


        # ---------------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------------

        axes[row, 1].imshow(
            image,
            cmap="gray"
        )

        axes[row, 1].imshow(
            np.ma.masked_where(
                ~gt,
                gt
            ),
            cmap="Reds",
            alpha=0.8,
            vmin=0,
            vmax=1
        )


        # ---------------------------------------------------------------------
        # AIR + ORIGINAL MT
        # ---------------------------------------------------------------------

        axes[row, 2].imshow(
            image,
            cmap="gray"
        )

        axes[row, 2].imshow(
            np.ma.masked_where(
                ~air,
                air
            ),
            cmap="Blues",
            alpha=0.35,
            vmin=0,
            vmax=1
        )

        axes[row, 2].imshow(
            np.ma.masked_where(
                ~original,
                original
            ),
            cmap="autumn",
            alpha=0.85,
            vmin=0,
            vmax=1
        )


        # ---------------------------------------------------------------------
        # FILTERED MT
        # ---------------------------------------------------------------------

        axes[row, 3].imshow(
            image,
            cmap="gray"
        )

        axes[row, 3].imshow(
            np.ma.masked_where(
                ~air,
                air
            ),
            cmap="Blues",
            alpha=0.30,
            vmin=0,
            vmax=1
        )

        axes[row, 3].imshow(
            np.ma.masked_where(
                ~filtered,
                filtered
            ),
            cmap="Greens",
            alpha=0.85,
            vmin=0,
            vmax=1
        )


        # ---------------------------------------------------------------------
        # FILTERED ERROR MAP
        # ---------------------------------------------------------------------

        correct = (
            filtered
            & gt
        )

        missed = (
            gt
            & ~filtered
        )

        false_positive = (
            filtered
            & ~gt
        )


        axes[row, 4].imshow(
            image,
            cmap="gray"
        )

        axes[row, 4].imshow(
            np.ma.masked_where(
                ~correct,
                correct
            ),
            cmap="Greens",
            alpha=0.85,
            vmin=0,
            vmax=1
        )

        axes[row, 4].imshow(
            np.ma.masked_where(
                ~missed,
                missed
            ),
            cmap="Reds",
            alpha=0.90,
            vmin=0,
            vmax=1
        )

        axes[row, 4].imshow(
            np.ma.masked_where(
                ~false_positive,
                false_positive
            ),
            cmap="autumn",
            alpha=0.90,
            vmin=0,
            vmax=1
        )


        axes[row, 0].set_ylabel(
            f"Slice {z}"
        )

        for column in range(5):

            axes[row, column].axis(
                "off"
            )


    axes[0, 0].set_title(
        "CBCT"
    )

    axes[0, 1].set_title(
        "Ground Truth MT"
    )

    axes[0, 2].set_title(
        "Original nnU-Net\nBlue=Air | Yellow=MT"
    )

    axes[0, 3].set_title(
        f"Air-Constrained MT\n"
        f"≤ {best_distance:.1f} mm from Air"
    )

    axes[0, 4].set_title(
        "Filtered Errors\n"
        "Green=Correct | Red=Missed | Yellow=False +"
    )


    fig.suptitle(
        f"{case_id} | "
        f"Original Dice "
        f"{case['original']['dice']:.3f} "
        f"→ Filtered "
        f"{filtered_metrics['dice']:.3f}",
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
        / f"{case_id}_AIR_DISTANCE_COMPARISON.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# FINAL
# =============================================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Best distance: "
    f"{best_distance:.1f} mm"
)

print(
    f"Mean Dice: "
    f"{original_mean_dice:.4f} "
    f"-> "
    f"{best_result['mean_dice']:.4f}"
)

print(
    f"Improvement: "
    f"{best_result['mean_dice'] - original_mean_dice:+.4f}"
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