"""Quantify same-frame RGB/depth agreement before a temporal completion run."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from complete_subsampled_blender_datasets import BASE, OUT, DATA, pfm
from core23_temporal_format import write_json_atomic

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='PyMouseStudio-WhiteFur-WaterReaching-FrameToFrame-FGcamB')
parser.add_argument('--render-root', default='output/temporal_completion/whitefur_reference_frame1')
args = parser.parse_args()
source = DATA/args.dataset/'frame_000001'
test = BASE/args.render_root/'frame_000001'
rows = []
for path in sorted((source/'blended_images').glob('*.jpg')):
    if '_masked' in path.name:
        continue
    a = np.asarray(Image.open(path)).astype(float)
    b = np.asarray(Image.open(test/'blended_images'/path.name)).astype(float)
    d0 = pfm(source/'rendered_depth_maps'/f'{path.stem}.pfm')
    d1 = pfm(test/'rendered_depth_maps'/f'{path.stem}.pfm')
    valid0 = d0 > 0
    valid1 = d1 > 0
    common = valid0 & valid1
    relative = np.abs(d1[common]-d0[common])/np.maximum(d0[common],1e-6)
    rows.append(dict(view=path.stem, rgb_mae_255=float(np.abs(a-b).mean()),
        depth_valid_iou=float((valid0&valid1).sum()/max(1,(valid0|valid1).sum())),
        depth_relative_median=float(np.median(relative)) if common.any() else None,
        depth_relative_p99=float(np.percentile(relative,99)) if common.any() else None))
report = dict(dataset=args.dataset, same_frame=1, views=rows,
    note='Pixel-identical Cycles sampling is not required; large geometry/depth discrepancies must be inspected.')
write_json_atomic(OUT/f'{args.dataset}_reference_comparison.json',report)
print(json.dumps(report,indent=2))
