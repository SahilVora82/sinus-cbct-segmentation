from pathlib import Path
import numpy as np
import nrrd
import matplotlib.pyplot as plt

# Put this file in the repository root:
# sinus-cbct-segmentation/
#   inspect_mt_1000_worst.py
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
OUT_DIR = ROOT / "mt_1000_worst_case_inspection"
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

    indices = [
        max(0, best_slice - 2),
        best_slice,
        min(scan.shape[best_axis] - 1, best_slice + 2),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14, 11))

    for row, idx in enumerate(indices):
        s = np.take(scan, idx, axis=best_axis)
        g = np.take(gt_mt, idx, axis=best_axis)
        p = np.take(pred_mt, idx, axis=best_axis)

        lo, hi = np.percentile(s, [1, 99])
        s_display = np.clip(s, lo, hi)

        axes[row, 0].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"CBCT - slice {idx}")

        axes[row, 1].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 1].imshow(
            np.ma.masked_where(~g.T, g.T),
            cmap="Reds",
            alpha=0.55,
            origin="lower",
        )
        axes[row, 1].set_title("Ground Truth MT")

        axes[row, 2].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 2].imshow(
            np.ma.masked_where(~p.T, p.T),
            cmap="Blues",
            alpha=0.55,
            origin="lower",
        )
        axes[row, 2].set_title("Predicted MT")

        fp = np.logical_and(p, ~g)
        fn = np.logical_and(g, ~p)

        err = np.zeros((*g.shape, 4), dtype=float)
        err[fp] = [1.0, 0.0, 0.0, 0.65]
        err[fn] = [0.0, 0.0, 1.0, 0.65]

        axes[row, 3].imshow(s_display.T, cmap="gray", origin="lower")
        axes[row, 3].imshow(err.transpose(1, 0, 2), origin="lower")
        axes[row, 3].set_title("Errors: red=FP, blue=FN")

        for ax in axes[row]:
            ax.axis("off")

    d = dice(gt_mt, pred_mt)
    gt_n = int(gt_mt.sum())
    pred_n = int(pred_mt.sum())
    ratio = pred_n / gt_n if gt_n else float("nan")

    fig.suptitle(
        f"{case} | MT Dice={d:.4f} | GT={gt_n:,} voxels | "
        f"Pred={pred_n:,} voxels | Pred/GT={ratio:.2f}x",
        fontsize=14,
    )

    plt.tight_layout()
    out = OUT_DIR / f"{case}_inspection.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"{case}: Dice={d:.4f}, GT={gt_n}, Pred={pred_n}, Pred/GT={ratio:.2f}x")
    print(f"Saved: {out}")

print()
print("DONE")
print(f"Open this folder:\n{OUT_DIR}")
