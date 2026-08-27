from pathlib import Path
import csv
import numpy as np
import nrrd
import torch
from scipy import ndimage
from unet import UNet

BASE = Path(__file__).resolve().parent
# When copied into MT/, BASE is MT/. If run directly elsewhere, set these manually.
SCAN_DIR = BASE / 'data' / 'scan_valid'
MASK_DIR = BASE / 'data' / 'segment_valid'
CKPT = BASE / 'results_hybrid' / 'best_mt_model.pt'
OUT = BASE / 'validation_postprocess'
OUT.mkdir(parents=True, exist_ok=True)

MEAN = -46.1730322190273
STD = 293.1394271328278
BG, MT, AIR = 0, 1, 2
CHANNELS = [32, 64, 128, 256]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print('Device:', device)
print('TEST DATA LOADED: NO')


def state_from(path):
    x = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(x, dict):
        for k in ('model', 'model_state_dict', 'state_dict'):
            if k in x:
                return x[k]
        if all(torch.is_tensor(v) for v in x.values()):
            return x
    raise RuntimeError('Could not find model weights')


def dice(p, t, eps=1e-8):
    p, t = p.astype(bool), t.astype(bool)
    d = p.sum() + t.sum()
    return 1.0 if d == 0 else float((2*np.logical_and(p,t).sum()+eps)/(d+eps))


def pr(p, t):
    p, t = p.astype(bool), t.astype(bool)
    tp = np.logical_and(p,t).sum(); fp = np.logical_and(p,~t).sum(); fn = np.logical_and(~p,t).sum()
    return (float(tp/(tp+fp)) if tp+fp else 0.0,
            float(tp/(tp+fn)) if tp+fn else 0.0)


def remove_small(mask, n):
    if n <= 0: return mask.copy()
    lab, count = ndimage.label(mask)
    if count == 0: return mask.copy()
    sizes = np.bincount(lab.ravel())
    keep = sizes >= n; keep[0] = False
    return keep[lab]


def pp(mask, close=0, fill=False, min_size=0):
    out = mask.astype(bool)
    if close:
        out = ndimage.binary_closing(out, structure=ndimage.generate_binary_structure(3,1), iterations=close)
    if fill:
        out = ndimage.binary_fill_holes(out)
    return remove_small(out, min_size)

model = UNet(CHANNELS).to(device)
model.load_state_dict(state_from(CKPT), strict=True)
model.eval()

cases = []
for sp in sorted(SCAN_DIR.glob('*.nrrd')):
    name = sp.stem
    scan, _ = nrrd.read(str(sp))
    gt, _ = nrrd.read(str(MASK_DIR / f'{name}_GT.nrrd'))
    x = ((scan.astype(np.float32)-MEAN)/STD).astype(np.float32)
    x = torch.from_numpy(x)[None,None].to(device)
    with torch.no_grad():
        pred = torch.argmax(model(x),1)[0].cpu().numpy().astype(np.uint8)
    gtm = gt == MT; pm = pred == MT
    missed = np.logical_and(gtm, ~pm)
    p0 = int(np.logical_and(missed, pred == BG).sum())
    p2 = int(np.logical_and(missed, pred == AIR).sum())
    prec, rec = pr(pm, gtm)
    cases.append(dict(name=name, gt=gtm, pred=pm, raw=dice(pm,gtm), precision=prec, recall=rec,
                      missed=int(missed.sum()), missed_bg=p0, missed_air=p2))

raw_mean = float(np.mean([c['raw'] for c in cases]))
print(f'\nRaw mean per-case MT Dice: {raw_mean:.4f}')
missed = sum(c['missed'] for c in cases); mbg = sum(c['missed_bg'] for c in cases); mair = sum(c['missed_air'] for c in cases)
print('\nWHERE MISSED GT-MT VOXELS GO')
if missed:
    print(f'Background: {mbg:,} ({100*mbg/missed:.1f}%)')
    print(f'Air:        {mair:,} ({100*mair/missed:.1f}%)')

recipes=[]
for close in (0,1,2):
    for fill in (False, True):
        for min_size in (0,25,50,100):
            scores=[dice(pp(c['pred'],close,fill,min_size), c['gt']) for c in cases]
            recipes.append((float(np.mean(scores)), close, fill, min_size, scores))
recipes.sort(reverse=True, key=lambda x:x[0])
best = recipes[0]
print('\nBEST VALIDATION-ONLY POSTPROCESS')
print('Raw Dice:          ', f'{raw_mean:.4f}')
print('Postprocessed Dice:', f'{best[0]:.4f}')
print('Change:            ', f'{best[0]-raw_mean:+.4f}')
print('Closing iterations:', best[1])
print('Fill holes:        ', best[2])
print('Min component size:', best[3])

with open(OUT/'per_case.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['case','raw_dice','post_dice','precision','recall','missed_as_bg','missed_as_air'])
    for c,s in zip(cases,best[4]):
        w.writerow([c['name'],c['raw'],s,c['precision'],c['recall'],c['missed_bg'],c['missed_air']])

print('\nPer-case:')
for c,s in sorted(zip(cases,best[4]), key=lambda z:z[0]['raw']):
    print(f"{c['name']:20s} raw={c['raw']:.4f} post={s:.4f} P={c['precision']:.3f} R={c['recall']:.3f}")
print('\nSaved:', OUT/'per_case.csv')
print('NOTE: validation only; test set untouched.')
