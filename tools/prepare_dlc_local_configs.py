#!/usr/bin/env python3
"""Create portable DeepLabCut inference configs without changing imported projects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


PROJECTS = {
    "A": {
        "directory": "wr_cama-tf-2024-07-13",
        "iteration": 406,
        "training_fraction": 0.85,
        "shuffle": 1,
        "snapshot": 200,
    },
    "B": {
        "directory": "wr_camb-tf-2024-07-11",
        "iteration": 167,
        "training_fraction": 0.90,
        "shuffle": 1,
        "snapshot": 140,
    },
    "C": {
        "directory": "wr_camc-tf-2024-07-12",
        "iteration": 0,
        "training_fraction": 0.90,
        "shuffle": 1,
        "snapshot": 15,
        "snapshot_file": "snapshot-best-015.pt",
        "task": "wr_camc",
        "date": "Jul12",
        "preserve_local_config": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def latest_evaluation_metrics(stats_path: Path) -> dict:
    if not stats_path.is_file():
        return {}
    with stats_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in reversed(rows):
        if row.get("metrics/test.mAP"):
            return {
                key: float(value)
                for key, value in row.items()
                if key.startswith("metrics/") and value
            }
    return {}


def evaluation_metrics_for_epoch(stats_path: Path, epoch: int) -> dict:
    if not stats_path.is_file():
        return {}
    with stats_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row["step"]) == epoch:
            return {
                key: float(value)
                for key, value in row.items()
                if key.startswith("metrics/") and value
            }
    return {}


def selected_model_folder(project: Path, cfg: dict, spec: dict) -> Path:
    task = cfg["Task"]
    date = cfg["date"]
    percent = int(round(spec["training_fraction"] * 100))
    return (
        project
        / "dlc-models-pytorch"
        / f"iteration-{spec['iteration']}"
        / f"{task}{date}-trainset{percent}shuffle{spec['shuffle']}"
        / "train"
    )


def main() -> None:
    args = parse_args()
    dlc_root = args.dlc_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else dlc_root / "local-configs"
    )
    output.mkdir(parents=True, exist_ok=True)

    report = {"dlc_root": str(dlc_root), "views": {}}
    for view, spec in PROJECTS.items():
        project = dlc_root / spec["directory"]
        source_config = project / "config.yaml"
        if not source_config.is_file():
            raise FileNotFoundError(source_config)
        cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        common = {
            "project": str(project),
            "source_config": str(source_config),
            "source_engine": cfg.get("engine"),
            "source_project_path": cfg.get("project_path"),
            "bodyparts": cfg.get("bodyparts", []),
            "labeled_data_directories": sum(
                child.is_dir() for child in (project / "labeled-data").glob("*")
            ),
        }

        if spec["iteration"] is None:
            report["views"][view] = {
                **common,
                "status": "missing_checkpoint",
                "tensorflow_snapshots": len(
                    list((project / "dlc-models").rglob("snapshot-*.index"))
                ),
                "pytorch_snapshots": len(
                    list((project / "dlc-models-pytorch").rglob("snapshot-*.pt"))
                ),
            }
            continue

        local_config_path = project / "config.local-pytorch.yaml"
        if spec.get("preserve_local_config") and local_config_path.is_file():
            local_cfg = yaml.safe_load(
                local_config_path.read_text(encoding="utf-8")
            )
        else:
            local_cfg = dict(cfg)
            local_cfg["project_path"] = str(project)
            local_cfg["engine"] = "pytorch"
            local_cfg["iteration"] = spec["iteration"]
            local_cfg["TrainingFraction"] = [spec["training_fraction"]]
            local_cfg["snapshotindex"] = -1
            if spec.get("task"):
                local_cfg["Task"] = spec["task"]
            if spec.get("date"):
                local_cfg["date"] = spec["date"]
            local_config_path.write_text(
                yaml.safe_dump(local_cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

        model_folder = selected_model_folder(project, local_cfg, spec)
        snapshot_path = model_folder / spec.get(
            "snapshot_file", f"snapshot-{spec['snapshot']:03d}.pt"
        )
        model_config_path = model_folder / "pytorch_config.yaml"
        if not snapshot_path.is_file():
            raise FileNotFoundError(snapshot_path)
        if not model_config_path.is_file():
            raise FileNotFoundError(model_config_path)

        report["views"][view] = {
            **common,
            "status": "ready",
            "local_config": str(local_config_path),
            "iteration": spec["iteration"],
            "training_fraction": spec["training_fraction"],
            "shuffle": spec["shuffle"],
            "snapshot": spec["snapshot"],
            "snapshot_path": str(snapshot_path),
            "snapshot_bytes": snapshot_path.stat().st_size,
            "model_config": str(model_config_path),
            "latest_evaluation_metrics": latest_evaluation_metrics(
                model_folder / "learning_stats.csv"
            ),
            "selected_evaluation_metrics": evaluation_metrics_for_epoch(
                model_folder / "learning_stats.csv", spec["snapshot"]
            ),
        }

    report_path = output / "project_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
