from pathlib import Path
import numpy as np
import nrrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path('.')
GT_DIR = ROOT / 'MT/data/segment_valid'
PRED_DIR = ROOT / 'nnUNet_work/results/Dataset501_SinusMT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation'
OUT_DIR = ROOT / 'mt_3d_worst_cases'
OUT_DIR.mkdir(exist_ok=True)

# Two worst cases from your earlier evaluation
CASES = [
    'FileB13_MT_L',  # Dice 0.4896
    'FileU36_MT_L',  # Dice 0.6504
]


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / denom if denom else 1.0


def crop_union(mask_a, mask_b, pad=3):
    union = np.logical_or(mask_a, mask_b)
    coords = np.argwhere(union)
    if coords.size == 0:
        return (slice(0, mask_a.shape[0]), slice(0, mask_a.shape[1]), slice(0, mask_a.shape[2]))

    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, mask_a.shape)
    return tuple(slice(int(mins[i]), int(maxs[i])) for i in range(3))


for case in CASES:
    gt, _ = nrrd.read(str(GT_DIR / f'{case}_GT.nrrd'))
    pred, _ = nrrd.read(str(PRED_DIR / f'{case}.nrrd'))

    gt_mt = gt == 1
    pred_mt = pred == 1

    crop = crop_union(gt_mt, pred_mt, pad=4)
    gt_c = gt_mt[crop]
    pred_c = pred_mt[crop]

    d = dice(gt_mt, pred_mt)
    gt_n = int(gt_mt.sum())
    pred_n = int(pred_mt.sum())

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.voxels(gt_c, facecolors='red', edgecolor='k', linewidth=0.1)
    ax1.set_title(f'Ground Truth MT\n{case}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect(gt_c.shape)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.voxels(pred_c, facecolors='blue', edgecolor='k', linewidth=0.1)
    ax2.set_title(f'Prediction MT\n{case}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect(pred_c.shape)

    fig.suptitle(
        f'{case} | Dice={d:.4f} | GT voxels={gt_n:,} | Pred voxels={pred_n:,}',
        fontsize=14
    )
    plt.tight_layout()

    out = OUT_DIR / f'{case}_3d_gt_vs_pred.png'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

print('\nDone.')
print(f'Output folder: {OUT_DIR.resolve()}')
