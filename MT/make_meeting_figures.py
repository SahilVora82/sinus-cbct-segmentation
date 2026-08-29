from pathlib import Path
import numpy as np
import nrrd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
# If this file is copied into MT/, ROOT becomes repo root. If run elsewhere, edit ROOT.
if BASE.name == 'MT':
    ROOT = BASE.parent
    MT_DIR = BASE
else:
    ROOT = Path.cwd()
    MT_DIR = ROOT / 'MT'

SCAN_DIR = MT_DIR / 'data' / 'scan_valid'
GT_DIR = MT_DIR / 'data' / 'segment_valid'
RESULTS_DIR = ROOT / 'nnUNet_work' / 'results'
OUT_DIR = ROOT / 'meeting_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG, MT, AIR = 0, 1, 2
SELECTED = [
    ('FileB16_MT_R', 'Best case'),
    ('FileU42_MT_L', 'Typical case'),
    ('FileU36_MT_L', 'Over-segmentation case'),
    ('FileB13_MT_L', 'Most challenging case'),
]


def find_prediction(case_id):
    candidates = list(RESULTS_DIR.rglob(f'{case_id}.nrrd'))
    val = [p for p in candidates if 'validation' in str(p).lower()]
    if val:
        return val[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(case_id)


def norm(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    d = 2 * tp / (pred.sum() + gt.sum()) if pred.sum() + gt.sum() else 1.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(d), float(p), float(r), int(fp), int(fn)


def load_case(case_id):
    scan, _ = nrrd.read(str(SCAN_DIR / f'{case_id}.nrrd'))
    gt, _ = nrrd.read(str(GT_DIR / f'{case_id}_GT.nrrd'))
    pred, _ = nrrd.read(str(find_prediction(case_id)))
    gt_mt, pred_mt = gt == MT, pred == MT
    return scan, gt_mt, pred_mt


def best_display_slice(gt, pred):
    # Choose an informative slice with both target and prediction, weighted toward GT amount.
    scores = []
    for z in range(gt.shape[0]):
        g = int(gt[z].sum())
        p = int(pred[z].sum())
        if g or p:
            overlap = int(np.logical_and(gt[z], pred[z]).sum())
            scores.append((g + 0.25 * p + 0.25 * overlap, z))
    return max(scores)[1] if scores else gt.shape[0] // 2


def overlay(ax, image, mask, cmap, alpha=0.65):
    ax.imshow(image, cmap='gray')
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=cmap, alpha=alpha, vmin=0, vmax=1)


# 1) Meeting overview: best, typical, hard, worst
fig, axes = plt.subplots(len(SELECTED), 4, figsize=(13, 3.1 * len(SELECTED)))
for row, (case_id, label_text) in enumerate(SELECTED):
    scan, gt, pred = load_case(case_id)
    d, p, r, fp, fn = metrics(pred, gt)
    z = best_display_slice(gt, pred)
    image = np.flipud(norm(scan[z]))
    gs = np.flipud(gt[z]); ps = np.flipud(pred[z])
    correct = gs & ps; missed = gs & ~ps; falsep = ps & ~gs

    axes[row, 0].imshow(image, cmap='gray')
    overlay(axes[row, 1], image, gs, 'Reds')
    overlay(axes[row, 2], image, ps, 'Blues')
    axes[row, 3].imshow(image, cmap='gray')
    axes[row, 3].imshow(np.ma.masked_where(~correct, correct), cmap='Greens', alpha=.8, vmin=0, vmax=1)
    axes[row, 3].imshow(np.ma.masked_where(~missed, missed), cmap='Reds', alpha=.85, vmin=0, vmax=1)
    axes[row, 3].imshow(np.ma.masked_where(~falsep, falsep), cmap='autumn', alpha=.85, vmin=0, vmax=1)

    axes[row, 0].set_ylabel(f'{label_text}\n{case_id}\n3D Dice {d:.3f}\nP {p:.3f} | R {r:.3f}', fontsize=9)
    for c in range(4):
        axes[row, c].axis('off')

for c, title in enumerate(['CBCT', 'Ground Truth MT', 'nnU-Net MT', 'Error Map']):
    axes[0, c].set_title(title, fontsize=12)
fig.suptitle('nnU-Net MT Segmentation — Representative Validation Cases\nError map: green = correct, red = missed, yellow = false positive', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, .96])
plt.savefig(OUT_DIR / 'meeting_overview.png', dpi=220, bbox_inches='tight')
plt.close()


# 2) Multi-slice figures for the two main over-segmentation failures
for case_id in ['FileB13_MT_L', 'FileU36_MT_L']:
    scan, gt, pred = load_case(case_id)
    d, p, r, fp, fn = metrics(pred, gt)
    slice_rows = []
    for z in range(gt.shape[0]):
        g, pr = gt[z], pred[z]
        if g.sum() or pr.sum():
            disagreement = int(np.logical_xor(g, pr).sum())
            slice_rows.append((disagreement, z))
    selected = sorted([z for _, z in sorted(slice_rows, reverse=True)[:6]])

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 2.7 * len(selected)))
    if len(selected) == 1:
        axes = axes[np.newaxis, :]
    for row, z in enumerate(selected):
        image = np.flipud(norm(scan[z])); gs = np.flipud(gt[z]); ps = np.flipud(pred[z])
        correct = gs & ps; missed = gs & ~ps; falsep = ps & ~gs
        axes[row, 0].imshow(image, cmap='gray')
        overlay(axes[row, 1], image, gs, 'Reds')
        overlay(axes[row, 2], image, ps, 'Blues')
        axes[row, 3].imshow(image, cmap='gray')
        axes[row, 3].imshow(np.ma.masked_where(~correct, correct), cmap='Greens', alpha=.8, vmin=0, vmax=1)
        axes[row, 3].imshow(np.ma.masked_where(~missed, missed), cmap='Reds', alpha=.85, vmin=0, vmax=1)
        axes[row, 3].imshow(np.ma.masked_where(~falsep, falsep), cmap='autumn', alpha=.85, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f'Slice {z}', fontsize=9)
        for c in range(4): axes[row, c].axis('off')
    for c, title in enumerate(['CBCT', 'Ground Truth MT', 'Prediction', 'Errors']):
        axes[0, c].set_title(title)
    fig.suptitle(f'{case_id} — 3D Dice {d:.3f}, Precision {p:.3f}, Recall {r:.3f}\nMost-disagreeing slices', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, .96])
    plt.savefig(OUT_DIR / f'{case_id}_multislice.png', dpi=220, bbox_inches='tight')
    plt.close()

print('Saved meeting figures to:', OUT_DIR)
print(' - meeting_overview.png')
print(' - FileB13_MT_L_multislice.png')
print(' - FileU36_MT_L_multislice.png')
