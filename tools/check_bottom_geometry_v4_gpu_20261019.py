"""Opt-in GPU engineering check: complete objectives, ZERO optimizer updates."""
import argparse
import fcntl
import gc
import json
from pathlib import Path

import torch

import bottom_geometry_v4_runtime_20261019 as rt
import run_bottom_geometry_v4_20261019 as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve old checks; use a fresh output directory')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); rt.geo.old.cv2.setNumThreads(1)
    manifest_path = args.previous_run/'manifest.json'
    manifest = rt.read_manifest(manifest_path)
    items = json.loads((args.previous_run/'registered_schedule.json').read_text())[0]
    weights = json.loads((args.previous_run/'common_loss_weights_lock.json').read_text())['weights']
    # Provisional engineering coefficient only, NOT a calibrated training recipe.
    weights['teacher'] = .05
    lock_path = rt.gpu_lock_path(); lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        teacher = rt.model(args.release)
        for name, checkpoint in [('release', None), ('adapted', args.previous_run/'A1-SI/latest.pt')]:
            student, optimizer = rt.model(args.release, checkpoint=checkpoint, trainable=True)
            if name == 'release':
                runner.preflight(student, None, manifest, items, weights, args.output/'P0_release.json')
            runner.preflight(student, teacher, manifest, items, weights, args.output/f'P1_{name}.json')
            del student, optimizer
            gc.collect(); torch.cuda.empty_cache()
        rt.write_new(args.output/'receipt.json', dict(passed=True, optimizer_steps=0, sealed_inference=0,
                     purpose='engineering only; no development evaluation or recipe promotion',
                     manifest_sha256=rt.sha(manifest_path), release_sha256=rt.sha(args.release),
                     source_pins=runner.source_files(), weights=weights,
                     gpu=torch.cuda.get_device_name(), torch_version=torch.__version__))


if __name__ == '__main__':
    main()
