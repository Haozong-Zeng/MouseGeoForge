"""Verify exclusion, immutability, shape conventions, and delivery integrity."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
OUT=ROOT/'output/pymousestudio_anipose_20261001'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Anipose-Review-20261001')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    config=json.loads((OUT/'config.json').read_text());groups=json.loads((OUT/'frame_manifest.json').read_text())
    manifest=json.loads((BASE/'ingestion_manifest.json').read_text())
    excluded={(r['session'],r['frame']) for r in config['excluded_cases']}
    included={(g['session'],g['frame']) for g in groups}
    assert len(included)==276 and len(excluded)==24 and not (included&excluded)
    assert len(list((OUT/'diagnostic_npz_NOT_GT').rglob('*.npz')))==276
    for g in groups:
        with np.load(OUT/g['diagnostic_path']) as z:
            original=z['uv_original'];proposed=z['uv_proposed_for_review'];updated=z['updated_mask']
            assert not z['train_allowed'] and z['not_ground_truth']
            assert not updated[:,[7,8,9,10,11,12,13,22]].any()
            np.testing.assert_equal(original[~updated],proposed[~updated])
            assert not updated[:,np.isfinite(original).all(-1).sum(0)<2].any()
            assert np.all(np.isfinite(proposed[updated]))
            with np.load(BASE/g['normalized_file']) as raw:
                for vi,cam in enumerate(g['cameras']):
                    i=z['camera_names'].tolist().index(cam)
                    np.testing.assert_equal(original[i],raw['uv'][vi,0])
    for row in manifest['files']:assert digest(BASE/row['path'])==row['sha256']
    for row in manifest['calibration']:assert digest(BASE/row['path'])==row['sha256']
    images=[p for p in OUT.rglob('*') if p.suffix.lower() in ('.jpg','.png')]
    for p in images:
        with Image.open(p) as im:im.verify()
    links=re.findall(r'(?:src|href)="([^"]+)"',(OUT/'index.html').read_text())
    for link in links:assert (OUT/link).is_file()
    for p in OUT.rglob('*'):
        if p.is_file():assert digest(p)==digest(DEST/p.relative_to(OUT))
    result=dict(status='passed',included_groups=276,excluded_groups=24,
                original_image_and_json_hashes_verified=len(manifest['files']),
                camera_hashes_verified=len(manifest['calibration']),abdomen_tail_singleview_preserved=True,
                visualization_images_decoded=len(images),html_links_verified=len(links),windows_copy_hashes_match=True)
    for folder in [OUT,DEST]:(folder/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
