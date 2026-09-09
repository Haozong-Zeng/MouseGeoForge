#!/usr/bin/env python3
"""Build a portable CamC DLC training split and PyTorch configuration."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("DeepLabCut/wr_camc-tf-2024-07-12"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path(
            "DeepLabCut/wr_camc-tf-2024-07-12/labeled-data-audit.json"
        ),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path(
            "DeepLabCut/wr_camb-tf-2024-07-11/dlc-models-pytorch/"
            "iteration-167/wr_cambJul11-trainset90shuffle1/train/snapshot-140.pt"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args()


class DisjointSet:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def choose_validation_groups(
    group_rows: dict[str, list[int]], target_rows: int, seed: int
) -> set[str]:
    """Choose whole groups with a deterministic subset sum near the target."""
    groups = list(group_rows)
    random.Random(seed).shuffle(groups)
    max_sum = min(sum(len(group_rows[group]) for group in groups), target_rows * 2)
    reachable: dict[int, tuple[int, str] | None] = {0: None}
    for group in groups:
        size = len(group_rows[group])
        for prior_sum in sorted(list(reachable), reverse=True):
            new_sum = prior_sum + size
            if new_sum <= max_sum and new_sum not in reachable:
                reachable[new_sum] = (prior_sum, group)
    selected_sum = min(reachable, key=lambda value: (abs(value - target_rows), value))
    selected: set[str] = set()
    while selected_sum:
        prior_sum, group = reachable[selected_sum]  # type: ignore[misc]
        selected.add(group)
        selected_sum = prior_sum
    return selected


def folder_from_index(index: object) -> str:
    if isinstance(index, tuple) and len(index) >= 2:
        return str(index[1])
    parts = str(index).replace("\\", "/").split("/")
    if "labeled-data" in parts:
        position = parts.index("labeled-data")
        if position + 1 < len(parts):
            return parts[position + 1]
    raise ValueError(f"Cannot identify labeled-data folder from index: {index!r}")


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    audit_path = args.audit.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    usable_folders = sorted(
        item["folder"]
        for item in audit["folders"]
        if item["h5_files"] and not item["read_error"]
    )
    if not usable_folders:
        raise RuntimeError("No usable labeled-data folders found")

    source_config_path = project / "config.yaml"
    config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "Task": "wr_camc",
            "date": "Jul12",
            "project_path": str(project),
            "engine": "pytorch",
            "iteration": 0,
            "TrainingFraction": [round(1.0 - args.validation_fraction, 2)],
            "default_net_type": "hrnet_w18",
            "default_augmenter": "albumentations",
            "snapshotindex": -1,
        }
    )
    config["video_sets"] = {
        str(project / "videos" / f"{folder}.mp4"): {"crop": "0, 1200, 0, 800"}
        for folder in usable_folders
    }
    local_config_path = project / "config.local-pytorch.yaml"
    local_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, width=1000),
        encoding="utf-8",
    )

    scorer = config["scorer"]
    frames = []
    for folder in usable_folders:
        table = pd.read_hdf(
            project / "labeled-data" / folder / f"CollectedData_{scorer}.h5"
        )
        frames.append(table)
    merged = pd.concat(frames).sort_index()

    disjoint = DisjointSet(usable_folders)
    for paths in audit.get("duplicate_image_hash_groups", []):
        folders = sorted(
            {
                Path(path.replace("\\", "/")).parts[1]
                for path in paths
                if len(Path(path.replace("\\", "/")).parts) >= 3
            }
        )
        for folder in folders[1:]:
            if folders[0] in disjoint.parent and folder in disjoint.parent:
                disjoint.union(folders[0], folder)

    group_rows: dict[str, list[int]] = {}
    row_folders: list[str] = []
    for row_index, frame_index in enumerate(merged.index):
        folder = folder_from_index(frame_index)
        row_folders.append(folder)
        group_rows.setdefault(disjoint.find(folder), []).append(row_index)

    target_validation_rows = round(len(merged) * args.validation_fraction)
    validation_groups = choose_validation_groups(
        group_rows, target_validation_rows, args.seed
    )
    test_indices = np.asarray(
        [
            row_index
            for row_index, folder in enumerate(row_folders)
            if disjoint.find(folder) in validation_groups
        ],
        dtype=int,
    )
    train_indices = np.asarray(
        sorted(set(range(len(merged))) - set(test_indices.tolist())), dtype=int
    )
    validation_folders = sorted(
        {
            folder
            for folder in usable_folders
            if disjoint.find(folder) in validation_groups
        }
    )

    split_report = {
        "project": str(project),
        "config": str(local_config_path),
        "source_checkpoint": str(source_checkpoint),
        "seed": args.seed,
        "folders_total": len(usable_folders),
        "folders_train": len(usable_folders) - len(validation_folders),
        "folders_validation": len(validation_folders),
        "rows_total": len(merged),
        "rows_train": len(train_indices),
        "rows_validation": len(test_indices),
        "train_fraction": len(train_indices) / len(merged),
        "validation_folders": validation_folders,
    }
    split_report_path = project / "camc-training-split.json"
    split_report_path.write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )

    import deeplabcut
    from deeplabcut.core.engine import Engine
    from deeplabcut.core.weight_init import WeightInitialization

    weight_init = WeightInitialization(
        snapshot_path=source_checkpoint,
        dataset="wr_camb",
        with_decoder=False,
        memory_replay=False,
    )
    splits = deeplabcut.create_training_dataset(
        local_config_path,
        Shuffles=[1],
        trainIndices=[train_indices],
        testIndices=[test_indices],
        net_type="hrnet_w18",
        augmenter_type="albumentations",
        weight_init=weight_init,
        engine=Engine.PYTORCH,
        userfeedback=False,
    )
    if not splits:
        raise RuntimeError("DeepLabCut did not create a training dataset")

    model_config_paths = sorted(
        (project / "dlc-models-pytorch" / "iteration-0").glob(
            "*/train/pytorch_config.yaml"
        )
    )
    if len(model_config_paths) != 1:
        raise RuntimeError(
            f"Expected one generated PyTorch config, found {model_config_paths}"
        )
    model_config_path = model_config_paths[0]
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    model_config["train_settings"]["weight_init"] = {
        "snapshot_path": str(source_checkpoint),
        "detector_snapshot_path": None,
        "dataset": "wr_camb",
        "with_decoder": True,
        "memory_replay": False,
        "conversion_array": np.arange(
            len(config["bodyparts"]), dtype=int
        ).tolist(),
        "bodyparts": list(config["bodyparts"]),
    }
    model_config["runner"]["eval_interval"] = 5
    model_config["runner"]["optimizer"]["params"]["lr"] = 5e-5
    model_config["runner"]["scheduler"] = {
        "type": "LRListScheduler",
        "params": {
            "lr_list": [[1e-5], [1e-6]],
            "milestones": [40, 55],
        },
    }
    model_config["runner"]["snapshots"].update(
        {"max_snapshots": 12, "save_epochs": 5, "save_optimizer_state": True}
    )
    model_config["train_settings"].update(
        {
            "batch_size": 8,
            "dataloader_workers": 4,
            "display_iters": 100,
            "epochs": 60,
            "seed": args.seed,
        }
    )
    model_config_path.write_text(
        yaml.safe_dump(model_config, sort_keys=False, width=1000),
        encoding="utf-8",
    )
    split_report["model_config"] = str(model_config_path)
    split_report_path.write_text(
        json.dumps(split_report, indent=2), encoding="utf-8"
    )
    print(json.dumps(split_report, indent=2))


if __name__ == "__main__":
    main()
