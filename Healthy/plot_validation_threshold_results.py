from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALIDATION_CSV_PATH = Path(
    "validation_threshold_with_largest_component.csv"
)

TEST_CSV_PATH = Path(
    "test_metrics_final_threshold_084_largest_component.csv"
)

OUTPUT_DIRECTORY = Path("graphs")


def verify_file_exists(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find required file: {file_path}"
        )


def add_bar_labels(bars):
    for bar in bars:
        value = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main():
    verify_file_exists(VALIDATION_CSV_PATH)
    verify_file_exists(TEST_CSV_PATH)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_data = pd.read_csv(
        VALIDATION_CSV_PATH
    )

    test_data = pd.read_csv(
        TEST_CSV_PATH
    )

    required_validation_columns = [
        "threshold",
        "raw_dice",
        "cleaned_dice",
        "cleaned_iou",
        "cleaned_precision",
        "cleaned_recall",
    ]

    required_test_columns = [
        "dice",
        "iou",
        "precision",
        "recall",
    ]

    missing_validation_columns = [
        column
        for column in required_validation_columns
        if column not in validation_data.columns
    ]

    missing_test_columns = [
        column
        for column in required_test_columns
        if column not in test_data.columns
    ]

    if missing_validation_columns:
        raise ValueError(
            "Validation CSV is missing columns: "
            f"{missing_validation_columns}"
        )

    if missing_test_columns:
        raise ValueError(
            "Test CSV is missing columns: "
            f"{missing_test_columns}"
        )

    # Select the threshold with the highest cleaned
    # validation Dice score.
    best_validation_row = validation_data.loc[
        validation_data["cleaned_dice"].idxmax()
    ]

    best_threshold = float(
        best_validation_row["threshold"]
    )

    validation_dice = float(
        best_validation_row["cleaned_dice"]
    )

    validation_iou = float(
        best_validation_row["cleaned_iou"]
    )

    validation_precision = float(
        best_validation_row["cleaned_precision"]
    )

    validation_recall = float(
        best_validation_row["cleaned_recall"]
    )

    # Calculate the mean across the four final test scans.
    test_dice = float(
        test_data["dice"].mean()
    )

    test_iou = float(
        test_data["iou"].mean()
    )

    test_precision = float(
        test_data["precision"].mean()
    )

    test_recall = float(
        test_data["recall"].mean()
    )

    # ==================================================
    # GRAPH 1:
    # Dice before and after largest-component cleanup
    # ==================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        validation_data["threshold"],
        validation_data["raw_dice"],
        marker="o",
        markersize=3,
        markevery=3,
        label="Before cleanup",
    )

    plt.plot(
        validation_data["threshold"],
        validation_data["cleaned_dice"],
        marker="o",
        markersize=3,
        markevery=3,
        label="After largest-component cleanup",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=(
            f"Selected threshold = "
            f"{best_threshold:.2f}"
        ),
    )

    plt.title(
        "Effect of Largest-Component Cleanup "
        "on Validation Dice"
    )

    plt.xlabel("Probability threshold")
    plt.ylabel("Average Dice score")

    plt.ylim(0.94, 1.00)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    graph_1_path = (
        OUTPUT_DIRECTORY
        / "raw_vs_cleaned_validation_dice.png"
    )

    plt.savefig(
        graph_1_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================
    # GRAPH 2:
    # Dice, precision, and recall across thresholds
    # ==================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        validation_data["threshold"],
        validation_data["cleaned_dice"],
        marker="o",
        markersize=3,
        markevery=3,
        label="Dice",
    )

    plt.plot(
        validation_data["threshold"],
        validation_data["cleaned_precision"],
        marker="o",
        markersize=3,
        markevery=3,
        label="Precision",
    )

    plt.plot(
        validation_data["threshold"],
        validation_data["cleaned_recall"],
        marker="o",
        markersize=3,
        markevery=3,
        label="Recall",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=(
            f"Selected threshold = "
            f"{best_threshold:.2f}"
        ),
    )

    plt.scatter(
        best_threshold,
        validation_dice,
        s=70,
        zorder=5,
    )

    plt.title(
        "Validation Performance Across "
        "Probability Thresholds"
    )

    plt.xlabel("Probability threshold")
    plt.ylabel("Average metric score")

    plt.ylim(0.93, 1.00)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    graph_2_path = (
        OUTPUT_DIRECTORY
        / "validation_threshold_tradeoff.png"
    )

    plt.savefig(
        graph_2_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================
    # GRAPH 3:
    # Validation versus final test performance
    # ==================================================

    metric_names = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
    ]

    validation_values = [
        validation_dice,
        validation_iou,
        validation_precision,
        validation_recall,
    ]

    test_values = [
        test_dice,
        test_iou,
        test_precision,
        test_recall,
    ]

    x_positions = np.arange(
        len(metric_names)
    )

    bar_width = 0.36

    plt.figure(figsize=(10, 6))

    validation_bars = plt.bar(
        x_positions - bar_width / 2,
        validation_values,
        width=bar_width,
        label="Validation",
    )

    test_bars = plt.bar(
        x_positions + bar_width / 2,
        test_values,
        width=bar_width,
        label="Final test",
    )

    add_bar_labels(validation_bars)
    add_bar_labels(test_bars)

    plt.xticks(
        x_positions,
        metric_names,
    )

    plt.title(
        "Validation and Final Test Performance\n"
        f"Threshold {best_threshold:.2f} "
        "with Largest-Component Cleanup"
    )

    plt.ylabel("Average metric score")

    # Starting at zero avoids visually exaggerating
    # the differences between validation and test.
    plt.ylim(0.0, 1.08)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    graph_3_path = (
        OUTPUT_DIRECTORY
        / "validation_vs_test_metrics.png"
    )

    plt.savefig(
        graph_3_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 65)
    print("GRAPHS CREATED")
    print()
    print(f"Graph 1: {graph_1_path}")
    print(f"Graph 2: {graph_2_path}")
    print(f"Graph 3: {graph_3_path}")
    print()
    print("SELECTED VALIDATION PIPELINE")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Dice:      {validation_dice:.4f}")
    print(f"IoU:       {validation_iou:.4f}")
    print(f"Precision: {validation_precision:.4f}")
    print(f"Recall:    {validation_recall:.4f}")
    print()
    print("FINAL TEST RESULTS")
    print(f"Dice:      {test_dice:.4f}")
    print(f"IoU:       {test_iou:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()