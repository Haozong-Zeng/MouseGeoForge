#!/usr/bin/env python3
"""Replace rejected slots in a CRIM13 discrete-pose plan with diverse poses."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from select_crim13_diverse_dual_mouse_frames import (
    CRIM_ROOT,
    IDENTITIES,
    descriptor,
    frame_from_key,
    make_plan,
    read_labels,
    robust_standardize,
    session_from_key,
    xy_from_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-frames", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--labels", type=Path, default=CRIM_ROOT / "labels_InD.csv")
    return parser.parse_args()


def source_key(record: dict) -> str:
    return record["source_image"].replace("\\", "/").split("/crim13/", 1)[1]


def main() -> None:
    args = parse_args()
    rejected_slots = sorted({
        int(value.strip()) for value in args.scene_frames.split(",") if value.strip()
    })
    base = json.loads(args.base_plan.resolve().read_text(encoding="utf-8"))
    by_scene = {int(row["scene_frame"]): row for row in base["keyframes"]}
    missing = sorted(set(rejected_slots) - set(by_scene))
    if missing:
        raise ValueError(f"Scene frames are absent from the base plan: {missing}")

    table = read_labels(args.labels.resolve())
    valid_keys = []
    raw_descriptors = []
    for key, row in table.iterrows():
        points = [xy_from_row(row, identity) for identity in IDENTITIES]
        if not all(np.all(np.isfinite(value)) for value in points):
            continue
        valid_keys.append(key)
        raw_descriptors.append(descriptor(*points))
    values = robust_standardize(np.asarray(raw_descriptors, dtype=np.float64))
    descriptor_by_key = dict(zip(valid_keys, values, strict=True))

    rejected_keys = {source_key(by_scene[frame]) for frame in rejected_slots}
    retained_keys = [
        source_key(row) for row in base["keyframes"]
        if int(row["scene_frame"]) not in rejected_slots
    ]
    selected = set(retained_keys)
    unavailable_source_ids = {
        int(value)
        for value in base.get("excluded_source_frames_after_geometry_qa", [])
    }
    unavailable_source_ids.update(frame_from_key(key) for key in rejected_keys)
    available = [
        key for key in valid_keys
        if session_from_key(key) == args.session
        and key not in selected
        and frame_from_key(key) not in unavailable_source_ids
    ]
    chosen = [descriptor_by_key[key] for key in retained_keys]
    replacements = []
    while len(replacements) < len(rejected_slots):
        candidate_values = np.asarray([descriptor_by_key[key] for key in available])
        selected_values = np.asarray(chosen)
        minimum_distance = np.min(
            np.sum(
                (candidate_values[:, None, :] - selected_values[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
        position = int(np.argmax(minimum_distance))
        key = available.pop(position)
        replacements.append(key)
        chosen.append(descriptor_by_key[key])

    mapping = dict(zip(rejected_slots, replacements, strict=True))
    generated = make_plan(
        table,
        args.session,
        replacements,
        float(base["homecage_mapping"]["pixel_to_mm"]),
        float(base["minimum_body_center_mm"]),
        float(base["homecage_mapping"]["pair_midpoint_scene_scale_mm_per_px"]),
        float(base["homecage_mapping"]["inter_mouse_distance_mapping"]["far_excess_scale"]),
        base.get("video_ranking", []),
        preserve_input_order=True,
    )
    generated_by_key = {
        source_key(row): row for row in generated["keyframes"]
    }

    output = copy.deepcopy(base)
    removed_sources = []
    added_sources = []
    for scene_frame, key in mapping.items():
        index = scene_frame - 1
        removed_sources.append(int(output["keyframes"][index]["source_image_frame"]))
        replacement = copy.deepcopy(generated_by_key[key])
        replacement["scene_frame"] = scene_frame
        output["keyframes"][index] = replacement
        output["source_label_frames"][index] = int(replacement["source_image_frame"])
        added_sources.append(int(replacement["source_image_frame"]))

    excluded = set(output.get("excluded_source_frames_after_geometry_qa", []))
    excluded.update(removed_sources)
    output["excluded_source_frames_after_geometry_qa"] = sorted(int(v) for v in excluded)
    output["incremental_replacement"] = {
        "removed_source_frames": removed_sources,
        "added_source_frames": added_sources,
        "replacement_scene_slots": rejected_slots,
        "reason": "strict dual-mouse tail geometry rejection",
    }
    output["frame_order_policy"] = (
        "base plan order preserved; geometry-rejected poses replaced in their original slots"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "replacement_session": args.session,
        "replacement_scene_slots": rejected_slots,
        "removed_source_frames": removed_sources,
        "added_source_frames": added_sources,
    }, indent=2))


if __name__ == "__main__":
    main()
