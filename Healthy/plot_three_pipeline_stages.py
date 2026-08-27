from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIRECTORY = Path("graphs")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "three_pipeline_stages_test_performance.png"
)


# Test-set results from the three meaningful pipeline stages.
metric_names = [
    "Dice",
    "IoU",
    "Precision",
    "Recall",
]

original_normalization = [
    0.7502,
    0.7346,
    0.9806,
    0.7485,
]

fixed_normalization = [
    0.9454,
    0.8978,
    0.9425,
    0.9533,
]

final_pipeline = [
    0.9548,
    0.9170,
    0.9959,
    0.9209,
]


def add_value_labels(bars):
    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    x_positions = np.arange(len(metric_names))
    bar_width = 0.25

    plt.figure(figsize=(11, 6.5))

    original_bars = plt.bar(
        x_positions - bar_width,
        original_normalization,
        width=bar_width,
        label="Original per-scan normalization",
    )

    fixed_bars = plt.bar(
        x_positions,
        fixed_normalization,
        width=bar_width,
        label="Fixed training-set normalization",
    )

    final_bars = plt.bar(
        x_positions + bar_width,
        final_pipeline,
        width=bar_width,
        label="Final: threshold 0.84 + cleanup",
    )

    add_value_labels(original_bars)
    add_value_labels(fixed_bars)
    add_value_labels(final_bars)

    plt.xticks(
        x_positions,
        metric_names,
        fontsize=11,
    )

    plt.title(
        "Test Performance Across Major Pipeline Improvements",
        fontsize=15,
    )

    plt.ylabel(
        "Average test metric score",
        fontsize=11,
    )

    # Begin at zero so score differences are not exaggerated.
    plt.ylim(0.0, 1.10)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend(
        loc="lower center",
        ncol=1,
        fontsize=10,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 65)
    print("GRAPH CREATED")
    print()
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("STAGE 1: ORIGINAL PER-SCAN NORMALIZATION")
    print("Dice:      0.7502")
    print("IoU:       0.7346")
    print("Precision: 0.9806")
    print("Recall:    0.7485")
    print()
    print("STAGE 2: FIXED TRAINING-SET NORMALIZATION")
    print("Dice:      0.9454")
    print("IoU:       0.8978")
    print("Precision: 0.9425")
    print("Recall:    0.9533")
    print()
    print("STAGE 3: FINAL THRESHOLD AND CLEANUP")
    print("Dice:      0.9548")
    print("IoU:       0.9170")
    print("Precision: 0.9959")
    print("Recall:    0.9209")
    print("=" * 65)


if __name__ == "__main__":
    main()
