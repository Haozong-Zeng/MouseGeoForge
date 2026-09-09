#!/usr/bin/env python3
"""Re-render gradcam/cam{A,B,C}_gradcam.png from surviving per-frame arrays.

Recovers the gradcam overlay PNGs (deterministically rendered from
patch_importance.npy over images.npy) without any GPU or model inference.
Resumable: skips frames whose gradcam dir already has the expected PNGs.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tony/vggt-omega")

DATA_ROOT = Path("/media/tony/backup_inte/vggt_paper/data_set")


def frame_done(frame: Path, num_frames: int) -> bool:
    gc = frame / "gradcam"
    if not gc.is_dir():
        return False
    expected = {f"cam{chr(ord('A') + f)}_gradcam.png" for f in range(num_frames)}
    return expected.issubset({p.name for p in gc.glob("*.png")})


def render_one(frame_str: str, force: bool) -> str:
    """Worker: re-render one frame's gradcam PNGs. Returns 'done'/'skip'/'fail:...'."""
    import torch
    from vggt_calibration_pipeline.model_inspector import _save_gradcam_pngs
    frame = Path(frame_str)
    imp_path, img_path = frame / "patch_importance.npy", frame / "images.npy"
    if not (imp_path.is_file() and img_path.is_file()):
        return "skip"
    try:
        imp = np.load(imp_path)
        num_frames = int(imp.shape[0])
        if not force and frame_done(frame, num_frames):
            return "skip"
        img = np.load(img_path)
        _, patch_h, patch_w = imp.shape
        _save_gradcam_pngs(imp, torch.from_numpy(img), patch_h, patch_w, num_frames, frame)
        return "done"
    except Exception as error:  # noqa: BLE001
        return f"fail:{frame}:{error}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="re-render even if PNGs exist")
    args = parser.parse_args()

    frames = [str(p) for p in sorted(args.data_root.glob("videos*/frame_*"))]
    total = len(frames)
    done = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, f, args.force): f for f in frames}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r == "done":
                done += 1
            elif r == "skip":
                skipped += 1
            else:
                failed += 1
                print(r, flush=True)
            if i % 500 == 0 or i == total:
                print(f"[{i}/{total}] rendered={done} skipped={skipped} failed={failed}", flush=True)

    print(f"DONE rendered={done} skipped={skipped} failed={failed} of {total} frame dirs")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
