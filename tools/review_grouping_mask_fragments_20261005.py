"""Visual audit of the three most fragmented FIT/CAL synthetic masks."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'vggt-omega')]
import torch
import numpy as np
from PIL import Image, ImageDraw
from run_surface_curve_pilot_20261003 import load_item
from training.instance_pose_losses import instance_targets_from_batch

def main():
    torch.set_num_threads(2)
    cycle = ROOT / 'output/local_grouping_cycle_20261005'
    audit = cycle / 'instance_supervision_audit_v1'
    inputs = ROOT / 'output/singleview_fusion_cycle_20261005/mixed_local_fulltrain_rgb_inputs_v1'
    manifest = {z['index']: z for z in json.loads((inputs / 'inference_manifest.json').read_text())}
    rows = json.loads((audit / 'details.json').read_text())
    cases = sorted([(p['largest_fraction'], r) for r in rows if r['metadata']['kind'] == 'dual'
                    for p in r['instances']], key=lambda z: z[0])[:3]
    for fraction, r in cases:
        i = r['index']; m = r['metadata']; h, w = manifest[i]['image_hw']
        assert m['stage'] in ['fit', 'calibration']
        item = load_item(dict(scene=m['scene'], views=[m['view']], image_hw=[h, w], kind='dual'))
        t = instance_targets_from_batch(item['batch'])
        rgb = (item['batch']['images'][0, 0].permute(1, 2, 0) * 255).round().byte().numpy()
        arr = rgb.astype(float)
        for q in range(2):
            mask = t['masks'][0, 0, q].bool().numpy()
            arr[mask] = .65 * arr[mask] + .35 * np.array([255, 50, 50] if q == 0 else [40, 180, 255])
        canvas = Image.new('RGB', (w * 2, h + 30), 'white')
        canvas.paste(Image.fromarray(rgb), (0, 30)); canvas.paste(Image.fromarray(arr.astype('uint8')), (w, 30))
        ImageDraw.Draw(canvas).text((4, 5), f'index{i} {m["stage"]}: RGB | visible instance masks; minimum component fraction {fraction:.3f}', fill='black')
        canvas.save(audit / f'fragment_{i}.jpg', quality=94)
        canvas.save(Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005/instance_supervision_audit_v1') / f'fragment_{i}.jpg', quality=94)
        print(i, fraction, flush=True)

if __name__ == '__main__':
    main()
