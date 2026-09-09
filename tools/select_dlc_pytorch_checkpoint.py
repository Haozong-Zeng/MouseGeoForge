#!/usr/bin/env python3
"""Select the best saved DeepLabCut PyTorch checkpoint from validation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.train_dir.resolve()
    output = (args.output or train_dir / "checkpoint-selection.json").resolve()
    stats_path = train_dir / "learning_stats.csv"
    stats = pd.read_csv(stats_path)
    metric = "metrics/test.mAP"
    valid = stats.dropna(subset=[metric]).copy()
    if valid.empty:
        raise RuntimeError(f"No validation rows with {metric} in {stats_path}")
    valid = valid.sort_values(
        [metric, "metrics/test.rmse", "step"],
        ascending=[False, True, True],
    )
    best_row = valid.iloc[0]
    best_epoch = int(best_row["step"])
    best_checkpoint = train_dir / f"snapshot-best-{best_epoch:03d}.pt"
    if best_checkpoint.is_file():
        snapshot_index = -1
    else:
        best_checkpoint = train_dir / f"snapshot-{best_epoch:03d}.pt"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(best_checkpoint)
        snapshots = sorted(
            (
                path
                for path in train_dir.glob("snapshot-*.pt")
                if "-best-" not in path.name
            ),
            key=lambda path: int(path.stem.rsplit("-", 1)[1]),
        )
        snapshot_index = snapshots.index(best_checkpoint)

    selection = {
        "train_dir": str(train_dir),
        "learning_stats": str(stats_path),
        "selection_metric": metric,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint),
        "snapshot_index": snapshot_index,
        "metrics": {
            key: float(value)
            for key, value in best_row.items()
            if key != "step" and pd.notna(value)
        },
        "evaluated_epochs": [int(value) for value in valid["step"]],
    }
    output.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    figure_path = output.with_name("learning_curves.png")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        stats["step"], stats["losses/train.total_loss"], label="train loss"
    )
    if "losses/eval.total_loss" in stats:
        axes[0].plot(
            stats["step"],
            stats["losses/eval.total_loss"],
            marker="o",
            label="validation loss",
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        valid.sort_values("step")["step"],
        valid.sort_values("step")[metric],
        marker="o",
        label="validation mAP",
    )
    axes[1].axvline(best_epoch, color="tab:red", linestyle="--", label="selected")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    selection["learning_curves"] = str(figure_path)
    output.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
