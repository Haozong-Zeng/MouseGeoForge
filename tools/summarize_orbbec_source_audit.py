"""Summarize frame coverage and label defects without modifying source data."""
import gzip
import json
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path('/home/haozong/projects/MouseGeoForge')
OUT = BASE / 'output/orbbec_openfield_audit'
ROOT = Path('/home/haozong/projects/Orbbec OpenField/Orbbec OpenField')
with gzip.open(OUT / 'inventory.json.gz','rt') as stream:
    inv=json.load(stream)
pairs=json.loads((OUT/'labeled_pairs.json').read_text())
summary={'removed_count':inv['removed_count'],'removed_bytes':inv['removed_bytes'],'sequences':[]}
for row in pairs:
    d=ROOT/row['sequence']
    rgb={int(p.stem.removeprefix('rgb_')) for p in d.glob('rgb_*.jpg')}
    dep={int(p.stem.removeprefix('depth_')) for p in d.glob('depth_*.png')}
    sizes={}
    for idx in sorted(rgb)[::max(len(rgb)//5,1)]:
        with Image.open(d/f'rgb_{idx:012d}.jpg') as im:
            sizes[str(im.size)]=sizes.get(str(im.size),0)+1
    record={'sequence':row['sequence'],'rgb_frames':len(rgb),'depth_frames':len(dep),'paired_frames':len(rgb&dep),
            'rgb_only':len(rgb-dep),'depth_only':len(dep-rgb),'min_id':min(rgb),'max_id':max(rgb),
            'missing_ids_in_range':max(rgb)-min(rgb)+1-len(rgb),'sample_sizes':sizes,'labeled_frames':row['label_count']}
    summary['sequences'].append(record)
summary['labeled_sequence_full_pairs']=sum(r['paired_frames'] for r in summary['sequences'])
exports=[(k,v) for k,v in inv['folders'].items() if k.count('/')==1 and k.endswith('-crop') and v['counts'].get('.jpg') and v['counts'].get('.png')]
summary['all_top_level_rgbd_exports']=len(exports)
summary['all_top_level_rgb_count']=sum(v['counts']['.jpg'] for k,v in exports)
summary['all_top_level_png_count']=sum(v['counts']['.png'] for k,v in exports)
label_summary=json.loads((OUT/'label_summary.json').read_text())
summary['json_coordinate_mismatch_files']=len({x['file'] for x in label_summary['json_coordinate_mismatches']})
empty=Path(label_summary['empty_labels'][0])
summary['empty_image']=str(empty.with_suffix('.jpg'))
Image.open(empty.with_suffix('.jpg')).save(OUT/'empty_annotation_rgb.jpg')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,ensure_ascii=False,indent=2))
