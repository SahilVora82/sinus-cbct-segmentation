from pathlib import Path

import numpy as np
import nrrd
import matplotlib.pyplot as plt

from scipy.ndimage import label


# =============================================================================
# SETTINGS
# =============================================================================

CASE_ID = "FileB13_MT_L"

N_SLICES = 10

BG = 0
MT = 1
AIR = 2


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
    / "nnUNet_region_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FIND PREDICTION
# =============================================================================

def find_prediction(case_id):

    candidates = list(
        NNUNET_RESULTS.rglob(
            f"{case_id}.nrrd"
        )
    )

    val_candidates = [
        p for p in candidates
        if "validation" in str(p).lower()
    ]

    if val_candidates:
        return val_candidates[0]

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No nnU-Net prediction found for {case_id}"
    )


# =============================================================================
# HELPERS
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


def dice(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    denominator = (
        pred.sum()
        + gt.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2
        * np.logical_and(
            pred,
            gt
        ).sum()
        / denominator
    )


# =============================================================================
# LOAD
# =============================================================================

scan, _ = nrrd.read(
    str(
        SCAN_DIR
        / f"{CASE_ID}.nrrd"
    )
)

gt, _ = nrrd.read(
    str(
        GT_DIR
        / f"{CASE_ID}_GT.nrrd"
    )
)

pred_path = find_prediction(
    CASE_ID
)

pred, _ = nrrd.read(
    str(pred_path)
)


print()
print("=" * 80)
print(CASE_ID)
print("=" * 80)

print(
    "Prediction:",
    pred_path
)

print(
    "Shape:",
    scan.shape
)


# =============================================================================
# CLASS MASKS
# =============================================================================

gt_mt = (
    gt == MT
)

gt_air = (
    gt == AIR
)

gt_sinus = (
    gt_mt
    | gt_air
)


pred_mt = (
    pred == MT
)

pred_air = (
    pred == AIR
)

pred_sinus = (
    pred_mt
    | pred_air
)


# =============================================================================
# ORIGINAL METRIC
# =============================================================================

original_dice = dice(
    pred_mt,
    gt_mt
)

print()
print(
    f"Original MT Dice: "
    f"{original_dice:.4f}"
)

print(
    f"GT MT voxels:       "
    f"{gt_mt.sum()}"
)

print(
    f"Predicted MT voxels:"
    f" {pred_mt.sum()}"
)

print(
    f"GT air voxels:      "
    f"{gt_air.sum()}"
)

print(
    f"Predicted air voxels:"
    f" {pred_air.sum()}"
)


# =============================================================================
# CONNECTED COMPONENT ANALYSIS OF PREDICTED WHOLE SINUS
# =============================================================================

connectivity = np.ones(
    (3, 3, 3),
    dtype=np.uint8
)

labeled_sinus, n_components = label(
    pred_sinus,
    structure=connectivity
)

sizes = np.bincount(
    labeled_sinus.ravel()
)

sizes[0] = 0

largest_component_id = (
    np.argmax(sizes)
)

largest_predicted_sinus = (
    labeled_sinus
    == largest_component_id
)


# Keep MT only if it belongs to the
# largest predicted sinus component.
cleaned_mt = (
    pred_mt
    & largest_predicted_sinus
)


cleaned_dice = dice(
    cleaned_mt,
    gt_mt
)


removed_mt = (
    pred_mt
    & ~largest_predicted_sinus
)


print()
print(
    "Predicted sinus components:",
    n_components
)

print(
    "Largest component voxels:",
    largest_predicted_sinus.sum()
)

print(
    "MT voxels outside largest sinus component:",
    removed_mt.sum()
)

print()
print(
    f"Dice after largest-sinus cleanup: "
    f"{cleaned_dice:.4f}"
)

print(
    f"Change: "
    f"{cleaned_dice - original_dice:+.4f}"
)


# =============================================================================
# CHOOSE THE MOST INFORMATIVE SLICES
# =============================================================================

slice_scores = []

for z in range(
    scan.shape[0]
):

    gt_mt_slice = (
        gt_mt[z]
    )

    pred_mt_slice = (
        pred_mt[z]
    )

    false_positive = (
        pred_mt_slice
        & ~gt_mt_slice
    )

    false_negative = (
        gt_mt_slice
        & ~pred_mt_slice
    )

    disagreement = (
        false_positive.sum()
        + false_negative.sum()
    )

    # We care about slices where
    # either GT or prediction has MT.
    if (
        gt_mt_slice.sum() > 0
        or
        pred_mt_slice.sum() > 0
    ):

        slice_scores.append(
            (
                z,
                disagreement
            )
        )


slice_scores.sort(
    key=lambda x: x[1],
    reverse=True
)

selected_slices = [
    z
    for z, _
    in slice_scores[:N_SLICES]
]

selected_slices.sort()


print()
print(
    "Slices shown:",
    selected_slices
)


# =============================================================================
# VISUALIZATION
# =============================================================================
#
# COL 1 = CBCT
#
# COL 2 = GT:
#         cyan = air
#         red  = MT
#
# COL 3 = nnU-Net:
#         cyan   = predicted air
#         yellow = predicted MT
#
# COL 4 = predicted whole sinus:
#         green  = AIR + MT
#         yellow = MT
#
# COL 5 = MT error:
#         green  = correct MT
#         red    = missed MT
#         yellow = false positive MT
#
# =============================================================================

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
        scan[z]
    )

    gt_air_slice = (
        gt_air[z]
    )

    gt_mt_slice = (
        gt_mt[z]
    )

    pred_air_slice = (
        pred_air[z]
    )

    pred_mt_slice = (
        pred_mt[z]
    )

    pred_sinus_slice = (
        pred_sinus[z]
    )


    correct_mt = (
        gt_mt_slice
        & pred_mt_slice
    )

    missed_mt = (
        gt_mt_slice
        & ~pred_mt_slice
    )

    false_mt = (
        pred_mt_slice
        & ~gt_mt_slice
    )


    # ---------------------------------------------------------
    # Flip for visual display
    # ---------------------------------------------------------

    image = np.flipud(
        image
    )

    gt_air_slice = np.flipud(
        gt_air_slice
    )

    gt_mt_slice = np.flipud(
        gt_mt_slice
    )

    pred_air_slice = np.flipud(
        pred_air_slice
    )

    pred_mt_slice = np.flipud(
        pred_mt_slice
    )

    pred_sinus_slice = np.flipud(
        pred_sinus_slice
    )

    correct_mt = np.flipud(
        correct_mt
    )

    missed_mt = np.flipud(
        missed_mt
    )

    false_mt = np.flipud(
        false_mt
    )


    # =========================================================
    # 1 — ORIGINAL CBCT
    # =========================================================

    axes[row, 0].imshow(
        image,
        cmap="gray"
    )


    # =========================================================
    # 2 — GROUND TRUTH AIR + MT
    # =========================================================

    axes[row, 1].imshow(
        image,
        cmap="gray"
    )

    axes[row, 1].imshow(
        np.ma.masked_where(
            ~gt_air_slice,
            gt_air_slice
        ),
        cmap="Blues",
        alpha=0.40,
        vmin=0,
        vmax=1
    )

    axes[row, 1].imshow(
        np.ma.masked_where(
            ~gt_mt_slice,
            gt_mt_slice
        ),
        cmap="Reds",
        alpha=0.80,
        vmin=0,
        vmax=1
    )


    # =========================================================
    # 3 — PREDICTED AIR + MT
    # =========================================================

    axes[row, 2].imshow(
        image,
        cmap="gray"
    )

    axes[row, 2].imshow(
        np.ma.masked_where(
            ~pred_air_slice,
            pred_air_slice
        ),
        cmap="Blues",
        alpha=0.40,
        vmin=0,
        vmax=1
    )

    axes[row, 2].imshow(
        np.ma.masked_where(
            ~pred_mt_slice,
            pred_mt_slice
        ),
        cmap="autumn",
        alpha=0.80,
        vmin=0,
        vmax=1
    )


    # =========================================================
    # 4 — PREDICTED WHOLE SINUS
    # =========================================================

    axes[row, 3].imshow(
        image,
        cmap="gray"
    )

    axes[row, 3].imshow(
        np.ma.masked_where(
            ~pred_sinus_slice,
            pred_sinus_slice
        ),
        cmap="Greens",
        alpha=0.35,
        vmin=0,
        vmax=1
    )

    axes[row, 3].imshow(
        np.ma.masked_where(
            ~pred_mt_slice,
            pred_mt_slice
        ),
        cmap="autumn",
        alpha=0.85,
        vmin=0,
        vmax=1
    )


    # =========================================================
    # 5 — MT ERRORS
    # =========================================================

    axes[row, 4].imshow(
        image,
        cmap="gray"
    )

    # Correct MT = green
    axes[row, 4].imshow(
        np.ma.masked_where(
            ~correct_mt,
            correct_mt
        ),
        cmap="Greens",
        alpha=0.85,
        vmin=0,
        vmax=1
    )

    # Missed MT = red
    axes[row, 4].imshow(
        np.ma.masked_where(
            ~missed_mt,
            missed_mt
        ),
        cmap="Reds",
        alpha=0.90,
        vmin=0,
        vmax=1
    )

    # False positive MT = yellow
    axes[row, 4].imshow(
        np.ma.masked_where(
            ~false_mt,
            false_mt
        ),
        cmap="autumn",
        alpha=0.90,
        vmin=0,
        vmax=1
    )


    # ---------------------------------------------------------
    # Slice label
    # ---------------------------------------------------------

    axes[row, 0].set_ylabel(
        f"Slice {z}",
        fontsize=10
    )

    for col in range(5):

        axes[row, col].axis(
            "off"
        )


# =============================================================================
# TITLES
# =============================================================================

axes[0, 0].set_title(
    "CBCT"
)

axes[0, 1].set_title(
    "Ground Truth\nBlue=Air | Red=MT"
)

axes[0, 2].set_title(
    "nnU-Net\nBlue=Air | Yellow=MT"
)

axes[0, 3].set_title(
    "Predicted Sinus\nGreen=Air+MT | Yellow=MT"
)

axes[0, 4].set_title(
    "MT Errors\nGreen=Correct | Red=Missed | Yellow=False +"
)


fig.suptitle(
    f"{CASE_ID}\n"
    f"Original MT Dice = {original_dice:.3f} | "
    f"Largest-sinus cleanup = {cleaned_dice:.3f}",
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
    / f"{CASE_ID}_AIR_MT_DIAGNOSTIC.png"
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "Saved:",
    output_path
)

print()
print(
    "TEST DATA WERE NEVER LOADED."
)