"""Render controlled C57 ear-material variants without saving the source blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


PROFILES = {
    "subtle": {
        "fur": (0.0060, 0.0035, 0.0028, 1.0),
        "skin": ((0.200, 0.085, 0.075, 1.0), (0.055, 0.022, 0.020, 1.0)),
        "sss_scale": 0.02,
    },
    "balanced": {
        "fur": (0.0120, 0.0060, 0.0045, 1.0),
        "skin": ((0.280, 0.120, 0.100, 1.0), (0.075, 0.030, 0.026, 1.0)),
        "sss_scale": 0.10,
    },
    "warm": {
        "fur": (0.0220, 0.0100, 0.0070, 1.0),
        "skin": ((0.380, 0.170, 0.140, 1.0), (0.100, 0.040, 0.034, 1.0)),
        "sss_scale": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args(argv)


def apply_profile(profile: dict[str, object]) -> None:
    skin = bpy.data.materials["MGF_C57_LocalV2_EarSkin"]
    fur = bpy.data.materials["MGF_C57_LocalV2_EarFur"]
    hair = next(node for node in fur.node_tree.nodes if node.type == "BSDF_HAIR_PRINCIPLED")
    hair.inputs["Color"].default_value = profile["fur"]
    for node in skin.node_tree.nodes:
        if node.type == "VALTORGB":
            node.color_ramp.elements[0].color = profile["skin"][0]
            node.color_ramp.elements[-1].color = profile["skin"][1]
        elif node.type == "SUBSURFACE_SCATTERING" and any(
            link.from_node == node for link in skin.node_tree.links
        ):
            node.inputs["Scale"].default_value = profile["sss_scale"]


def render_profile(name: str, output: Path, frame: int, samples: int) -> None:
    import render_homecage_multiview_preview as preview

    old_argv = sys.argv
    try:
        sys.argv = [
            "blender",
            "--",
            "--output-dir",
            str(output / name),
            "--engine",
            "CYCLES",
            "--width",
            "1280",
            "--height",
            "720",
            "--samples",
            str(samples),
            "--views",
            "side_neg_x,side_neg_y,side_pos_y",
            "--frame",
            str(frame),
        ]
        preview.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, profile in PROFILES.items():
        apply_profile(profile)
        render_profile(name, args.output_dir, args.frame, args.samples)
    (args.output_dir / "profiles.json").write_text(
        json.dumps(PROFILES, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
