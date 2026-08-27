from pathlib import Path
import numpy as np
import nrrd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "MT_input_use_allSeg_padded_updated"

gt_files = sorted(DATA_DIR.glob("*_GT.nrrd"))

print(f"Found {len(gt_files)} masks\n")

all_stats = {0: [], 1: [], 2: []}

for gt_path in gt_files:
    scan_name = gt_path.name.replace("_GT.nrrd", ".nrrd")
    scan_path = DATA_DIR / scan_name

    if not scan_path.exists():
        continue

    scan, _ = nrrd.read(str(scan_path))
    mask, _ = nrrd.read(str(gt_path))

    scan = scan.astype(np.float32)
    mask = mask.astype(np.int64)

    for label in [0, 1, 2]:
        values = scan[mask == label]

        if values.size > 0:
            all_stats[label].append(
                (
                    float(np.mean(values)),
                    float(np.median(values)),
                    int(values.size),
                )
            )

print("=" * 70)
print("INTENSITY STATISTICS BY MASK LABEL")
print("=" * 70)

for label in [0, 1, 2]:
    stats = all_stats[label]

    means = np.array([x[0] for x in stats])
    medians = np.array([x[1] for x in stats])
    voxels = np.array([x[2] for x in stats])

    print(f"\nLABEL {label}")
    print(f"  Number of scans containing label: {len(stats)}")
    print(f"  Average voxel count:             {voxels.mean():,.0f}")
    print(f"  Mean intensity across scans:     {means.mean():.2f}")
    print(f"  Median intensity across scans:   {medians.mean():.2f}")

print("\nInterpretation:")
print("The label with the LOWEST intensity inside the sinus is likely AIR.")
print("The higher-intensity sinus-region label is likely MUCOSAL THICKENING.")
print("Label 0 is expected to be background.")