"""Read-only label/depth pairing audit and diagnostic contact sheet."""
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('/home/haozong/projects/Orbbec OpenField/Orbbec OpenField')
OUT = Path('/home/haozong/projects/MouseGeoForge/output/orbbec_openfield_audit')
OUT.mkdir(parents=True, exist_ok=True)
rows, panels = [], []
slot_names = [Counter() for _ in range(10)]
visibility = Counter()
ambiguous_json = []
empty_labels = []
for labeled in sorted(ROOT.glob('*-labeled')):
    full = labeled.with_name(labeled.name.removesuffix('-labeled'))
    labels = sorted(labeled.glob('rgb_*.txt'))
    counts, linecounts, classes, sizes = Counter(), Counter(), Counter(), Counter()
    missing, json_names, json_orders, paired = [], Counter(), Counter(), 0
    for label in labels:
        lines = [line.split() for line in label.read_text().splitlines() if line.strip()]
        linecounts[len(lines)] += 1
        if not lines:
            empty_labels.append(str(label))
        for tokens in lines:
            counts[len(tokens)] += 1
            classes[tokens[0]] += 1
        rgb = labeled / (label.stem + '.jpg')
        depth = full / (label.stem.replace('rgb_', 'depth_') + '.png')
        full_rgb = full / rgb.name
        if not all(p.exists() for p in (rgb, depth, full_rgb)):
            missing.append({'label': label.name, 'rgb': rgb.exists(), 'depth': depth.exists(), 'full_rgb': full_rgb.exists()})
        else:
            paired += 1
        jp = label.with_suffix('.json')
        if jp.exists():
            jd = json.loads(jp.read_text())
            js = jd.get('labels', [])
            json_names.update(x.get('label', '') for x in js)
            json_orders[tuple((x.get('id'), x.get('label')) for x in js if x.get('drawType') == 'POINT')] += 1
            points = [x for x in js if x.get('drawType') == 'POINT' and len(x.get('points', [])) == 2]
            if len(lines) == 1 and len(lines[0]) == 25 and points:
                coords = np.array(lines[0][5:], float).reshape(10, 2) * [jd['size']['width'], jd['size']['height']]
                jpcoords = np.array([p['points'] for p in points])
                for slot, coord in enumerate(coords):
                    dist = np.linalg.norm(jpcoords-coord,axis=1)
                    nearest = int(dist.argmin())
                    if dist[nearest] < .01:
                        slot_names[slot][points[nearest]['label']] += 1
                        visibility.update([json.dumps(points[nearest].get('attr',{}),ensure_ascii=False)])
                    else:
                        ambiguous_json.append({'file': str(jp), 'slot': slot, 'nearest_error_px': float(dist[nearest])})
    sample = None
    for label in labels:
        rp = labeled / (label.stem + '.jpg')
        dp = full / (label.stem.replace('rgb_', 'depth_') + '.png')
        if not rp.exists() or not dp.exists():
            continue
        im, dep = cv2.imread(str(rp)), cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        orig = cv2.imread(str(full / rp.name))
        vals = dep[dep > 0]
        sample = {'file': rp.name, 'rgb_shape': list(im.shape), 'depth_shape': list(dep.shape),
                  'depth_dtype': str(dep.dtype), 'depth_quantiles': np.percentile(vals, [0, 1, 50, 99, 100]).tolist(),
                  'zero_fraction': float(np.mean(dep == 0)), '255_fraction': float(np.mean(dep == 255)),
                  'full_rgb_shape': list(orig.shape) if orig is not None else None,
                  'rgb_equal': bool(np.array_equal(im, orig))}
        sample['depth_channels_identical'] = bool(dep.ndim == 3 and np.array_equal(dep[:,:,0],dep[:,:,1]) and np.array_equal(dep[:,:,0],dep[:,:,2]))
        tokens = np.array(label.read_text().splitlines()[0].split(), float)
        h, w = im.shape[:2]
        xy = tokens[5:].reshape(-1, 2) * [w, h]
        for i, (x, y) in enumerate(xy):
            cv2.circle(im, (round(x), round(y)), 3, (0, 0, 255), -1)
            cv2.putText(im, str(i), (round(x)+3, round(y)-3), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        lo, hi = np.percentile(vals, [1, 99])
        dvis = cv2.applyColorMap(np.uint8(np.clip((dep.astype(float)-lo)/max(hi-lo,1)*255,0,255)), cv2.COLORMAP_TURBO)
        dvis[dep == 0] = 0
        panel = np.hstack([cv2.resize(im,(360,360)),cv2.resize(dvis,(360,360),interpolation=cv2.INTER_NEAREST)])
        banner = np.full((45,720,3), 255, np.uint8)
        cv2.putText(banner, labeled.name.replace('Orbbec-', '')[:85], (4,18), cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1)
        cv2.putText(banner, f'{rp.stem} | RGB points / raw depth', (4,36), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        panels.append(np.vstack([banner,panel]))
        break
    row = {'sequence': full.name, 'full_exists': full.exists(), 'label_count': len(labels),
           'json_count': len(list(labeled.glob('*.json'))), 'token_counts': dict(counts), 'instances_per_file': dict(linecounts),
           'classes': dict(classes), 'paired_count': paired, 'missing': missing,
           'json_label_names': dict(json_names), 'json_orders': [{'order': k, 'count': v} for k,v in json_orders.items()], 'sample': sample}
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('json_orders','missing')}, ensure_ascii=False),flush=True)
if panels:
    cv2.imwrite(str(OUT / 'labeled_rgb_depth.jpg'), np.vstack(panels))
(OUT / 'labeled_pairs.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
summary = {'labeled_sequences': len(rows), 'label_files': sum(r['label_count'] for r in rows),
           'paired_frames': sum(r['paired_count'] for r in rows), 'empty_labels': empty_labels,
           'slot_names': [dict(c) for c in slot_names], 'visibility_attributes': dict(visibility),
           'json_coordinate_mismatch_count': len(ambiguous_json), 'json_coordinate_mismatches': ambiguous_json}
(OUT / 'label_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in summary.items() if k!='json_coordinate_mismatches'},ensure_ascii=False))
