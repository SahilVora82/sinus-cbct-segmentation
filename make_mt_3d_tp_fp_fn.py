from pathlib import Path
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(".")
GT_DIR = ROOT / "MT/data/segment_valid"
PRED_DIR = ROOT / "nnUNet_work/results/Dataset501_SinusMT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation"
OUT_DIR = ROOT / "mt_3d_tp_fp_fn"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    "FileB13_MT_L",
    "FileU36_MT_L",
]

def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / denom if denom else 1.0

def crop_to_union(*masks, margin=2):
    union = np.logical_or.reduce(masks)
    coords = np.argwhere(union)
    if len(coords) == 0:
        return masks
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, union.shape)
    slices = tuple(slice(mins[i], maxs[i]) for i in range(3))
    return [m[slices] for m in masks]

for case in CASES:
    gt, _ = nrrd.read(str(GT_DIR / f"{case}_GT.nrrd"))
    pred, _ = nrrd.read(str(PRED_DIR / f"{case}.nrrd"))

    gt_mt = gt == 1
    pred_mt = pred == 1

    tp = gt_mt & pred_mt
    fn = gt_mt & (~pred_mt)
    fp = pred_mt & (~gt_mt)

    tp, fn, fp = crop_to_union(tp, fn, fp, margin=2)

    filled = tp | fn | fp
    colors = np.empty(filled.shape, dtype=object)
    colors[tp] = "limegreen"
    colors[fn] = "red"
    colors[fp] = "royalblue"

    d = dice(gt_mt, pred_mt)

    fig = plt.figure(figsize=(14, 6))

    # View 1
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.voxels(filled, facecolors=colors, edgecolor="k", linewidth=0.05)
    ax1.set_title("View 1")
    ax1.view_init(elev=24, azim=35)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # View 2
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.voxels(filled, facecolors=colors, edgecolor="k", linewidth=0.05)
    ax2.set_title("View 2")
    ax2.view_init(elev=18, azim=120)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    legend_handles = [
        Patch(facecolor="limegreen", edgecolor="k", label="True Positive (overlap)"),
        Patch(facecolor="red", edgecolor="k", label="False Negative (GT only)"),
        Patch(facecolor="royalblue", edgecolor="k", label="False Positive (Pred only)")
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(
        f"{case} | Dice={d:.4f} | GT={gt_mt.sum():,} voxels | Pred={pred_mt.sum():,} voxels",
        fontsize=16,
        y=1.03
    )

    plt.tight_layout()
    out_path = OUT_DIR / f"{case}_tp_fp_fn_3d.png"
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")

print()
print("DONE")
