from pathlib import Path
import numpy as np
import nrrd
import matplotlib.pyplot as plt

# Put this file in the repository root:
# sinus-cbct-segmentation/
#   inspect_mt_1000_worst_simple_colors.py
#   MT/
#   nnUNet_work/

ROOT = Path(__file__).resolve().parent

SCAN_DIR = ROOT / "MT" / "data" / "scan_valid"
GT_DIR = ROOT / "MT" / "data" / "segment_valid"
PRED_DIR = (
    ROOT
    / "nnUNet_work"
    / "results"
    / "Dataset501_SinusMT"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)
OUT_DIR = ROOT / "mt_1000_simple_color_inspection"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    "FileB13_MT_L",
    "FileU36_MT_L",
    "FileB13_MT_R",
]

def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / denom if denom else 1.0

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find:\n{path}")
    return path

for case in CASES:
    scan_path = require_file(SCAN_DIR / f"{case}.nrrd")
    gt_path = require_file(GT_DIR / f"{case}_GT.nrrd")
    pred_path = require_file(PRED_DIR / f"{case}.nrrd")

    scan, _ = nrrd.read(str(scan_path))
    gt, _ = nrrd.read(str(gt_path))
    pred, _ = nrrd.read(str(pred_path))

    gt_mt = gt == 1
    pred_mt = pred == 1

    # Find slice with largest GT MT area
    best_axis = None
    best_slice = None
    best_area = -1

    for axis in range(3):
        other_axes = tuple(i for i in range(3) if i != axis)
        areas = np.sum(gt_mt, axis=other_axes)
        idx = int(np.argmax(areas))
        if areas[idx] > best_area:
            best_area = int(areas[idx])
            best_axis = axis
            best_slice = idx

    # Only show 2 slices to keep it cleaner
    indices = [best_slice, min(scan.shape[best_axis] - 1, best_slice + 2)]
    if indices[1] == indices[0] and indices[0] > 0:
        indices[1] = indices[0] - 1

    fig, axes = plt.subplots(len(indices), 4, figsize=(13, 6.5))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        s = np.take(scan, idx, axis=best_axis)
        g = np.take(gt_mt, idx, axis=best_axis)
        p = np.take(pred_mt, idx, axis=best_axis)

        lo, hi = np.percentile(s, [1, 99])
        s_display = np.clip(s, lo, hi)

        # 1. Plain CBCT
        axes[row, 0].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"CBCT - slice {idx}")

        # 2. GT in red
        axes[row, 1].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 1].imshow(
            np.ma.masked_where(~g.T, g.T),
            cmap="Reds",
            alpha=0.70,
            origin="lower",
        )
        axes[row, 1].set_title("Ground Truth (red)")

        # 3. Pred in blue
        axes[row, 2].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 2].imshow(
            np.ma.masked_where(~p.T, p.T),
            cmap="Blues",
            alpha=0.70,
            origin="lower",
        )
        axes[row, 2].set_title("Prediction (blue)")

        # 4. Both together: GT red, Pred blue
        axes[row, 3].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 3].imshow(
            np.ma.masked_where(~g.T, g.T),
            cmap="Reds",
            alpha=0.55,
            origin="lower",
        )
        axes[row, 3].imshow(
            np.ma.masked_where(~p.T, p.T),
            cmap="Blues",
            alpha=0.55,
            origin="lower",
        )
        axes[row, 3].set_title("Overlay: GT red, Pred blue")

        for ax in axes[row]:
            ax.axis("off")

    d = dice(gt_mt, pred_mt)
    gt_n = int(gt_mt.sum())
    pred_n = int(pred_mt.sum())
    ratio = pred_n / gt_n if gt_n else float("nan")

    fig.suptitle(
        f"{case} | MT Dice={d:.4f} | GT={gt_n:,} voxels | Pred={pred_n:,} voxels | Pred/GT={ratio:.2f}x",
        fontsize=14,
    )

    plt.tight_layout()
    out = OUT_DIR / f"{case}_simple_overlay.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"{case}: saved {out}")

print()
print("DONE")
print(f"Open this folder:\n{OUT_DIR}")
