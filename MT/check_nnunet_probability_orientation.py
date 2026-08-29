from pathlib import Path
import itertools

import numpy as np
import nrrd


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

VAL_DIR = (
    ROOT
    / "nnUNet_work"
    / "results"
    / "Dataset501_SinusMT"
    / "nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)

CASE = "FileB13_MT_L"


# -------------------------------------------------------------------------
# Load saved nnU-Net segmentation
# -------------------------------------------------------------------------

seg, _ = nrrd.read(
    str(
        VAL_DIR
        / f"{CASE}.nrrd"
    )
)


# -------------------------------------------------------------------------
# Load probability array
# -------------------------------------------------------------------------

data = np.load(
    VAL_DIR
    / f"{CASE}.npz"
)

probs = data["probabilities"]

print("Probability shape:", probs.shape)
print("Segmentation shape:", seg.shape)


# -------------------------------------------------------------------------
# Try every permutation of the THREE spatial dimensions.
#
# Channel dimension stays first.
# We want argmax(probabilities) to reproduce the exported segmentation.
# -------------------------------------------------------------------------

results = []

for permutation in itertools.permutations(
    [1, 2, 3]
):

    order = (
        0,
        *permutation
    )

    transformed = np.transpose(
        probs,
        order
    )

    prediction = np.argmax(
        transformed,
        axis=0
    )

    agreement = np.mean(
        prediction == seg
    )

    results.append(
        (
            agreement,
            order
        )
    )


results.sort(
    reverse=True
)


print()
print("BEST ORIENTATIONS")
print("=" * 60)

for agreement, order in results:

    print(
        f"{order} -> "
        f"{agreement * 100:.6f}% agreement"
    )


best_agreement, best_order = results[0]

print()
print("=" * 60)
print("BEST")
print("=" * 60)

print(
    "Transpose:",
    best_order
)

print(
    "Agreement:",
    f"{best_agreement * 100:.6f}%"
)


if best_agreement > 0.999:

    print()
    print(
        "FOUND CORRECT PROBABILITY ORIENTATION."
    )

else:

    print()
    print(
        "Orientation alone did not fully explain it."
    )