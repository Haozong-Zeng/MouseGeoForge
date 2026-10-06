"""Audit an isolated Labelme return and render scientific overlays, never train.

Raw shapes are preserved. Parsed coordinates are observations, not accepted GT.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['nose', 'head_center', 'left_ear', 'right_ear', 'neck',
         'spine_center', 'lumbar_spine', 'tail_base', 'tail_q1', 'tail_q2',
         'tail_q3', 'tail_q4', 'tail_q5', 'tail_tip', 'left_forepaw_center',
         'right_forepaw_center', 'left_hindpaw_center', 'right_hindpaw_center',
         'left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'abdomen_center']
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(n):
    return ImageFont.truetype(FONT_PATH, n)


def color(k):
    if 7 <= k <= 13:
        return '#ffc83d'
    if k in (2, 14, 16, 18, 20):
        return '#00eaff'
    if k in (3, 15, 17, 19, 21):
        return '#ff69dc'
    return '#8dff6e'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'dataset/PyMouseStudio/annotation_pack_v2/manifest.json').read_text())
    by_hash = {}
    for row in manifest:
        by_hash.setdefault(row['sha256'], []).append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, parsed, previews = [], [], []
    for path in sorted(args.source.glob('*.json')):
        data = json.loads(path.read_text())
        image_path = path.with_suffix('.png')
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        matches = by_hash.get(digest, [])
        # Do not silently inspect held-out images or images of unknown provenance.
        if len(matches) != 1 or matches[0]['split'] != 'real_development':
            raise ValueError(f'Unverified/development-only provenance: {path}')
        source = matches[0]
        im = Image.open(image_path).convert('RGB')
        w, h = im.size
        assert (w, h) == (data['imageWidth'], data['imageHeight'])
        assert Path(data['imagePath']).name == image_path.name
        points, polys, errors = {}, [], []
        for shape in data['shapes']:
            xy = np.asarray(shape['points'], dtype=float)
            if not np.isfinite(xy).all() or (xy < 0).any() or (xy[:, 0] >= w).any() or (xy[:, 1] >= h).any():
                errors.append('out_of_bounds_or_nonfinite:' + shape['label'])
            if shape['shape_type'] == 'polygon':
                polys.append(xy)
            elif shape['shape_type'] == 'point':
                match = re.fullmatch(r'(\d{2})([a-z0-9_]+)', shape['label'])
                if not match:
                    errors.append('unknown_label:' + shape['label'])
                    continue
                k = int(match[1])-1
                if k not in range(23) or NAMES[k] != match[2] or len(xy) != 1:
                    errors.append('label_contract:' + shape['label'])
                    continue
                if k in points:
                    errors.append('duplicate:' + shape['label'])
                points[k] = xy[0]
        mask = np.zeros((h, w), np.uint8)
        polygon_checks = []
        for poly in polys:
            cv2.fillPoly(mask, [np.rint(poly).astype(np.int32)], 255)
            try:
                from shapely.geometry import Polygon
                from shapely.validation import explain_validity
                geom = Polygon(poly)
                polygon_checks.append(dict(valid=geom.is_valid, reason=explain_validity(geom)))
            except ImportError:
                polygon_checks.append(dict(valid=None, reason='shapely unavailable'))
        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        outside = [k+1 for k, xy in points.items() if not mask[min(h-1, max(0, round(xy[1]))), min(w-1, max(0, round(xy[0])))]]
        Image.fromarray(mask).save(args.output/f'{path.stem}_mask.png')
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        for poly in polys:
            pp = [tuple(v) for v in poly]
            draw.line(pp+[pp[0]], fill='#41ff56', width=3)
        for k, xy in points.items():
            x, y = xy
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color(k), outline='black', width=1)
            draw.text((x+7, y-19), str(k+1), font=font(21), fill=color(k), stroke_width=2, stroke_fill='black')
        overlay.save(args.output/f'{path.stem}_overlay.png')
        ys, xs = np.where(mask > 0)
        crop = (max(0, int(xs.min())-25), max(0, int(ys.min())-25),
                min(w, int(xs.max())+26), min(h, int(ys.max())+26))
        clean_crop, labeled_crop = im.crop(crop), overlay.crop(crop)
        cw, ch = clean_crop.size
        detail = Image.new('RGB', (2*cw, ch+40), '#18202b')
        detail.paste(clean_crop, (0,40)); detail.paste(labeled_crop, (cw,40))
        ImageDraw.Draw(detail).text((8,8), f'{path.stem}: original / returned labels | crop origin {crop[:2]}', font=font(20), fill='white')
        detail.save(args.output/f'{path.stem}_detail.jpg', quality=96)
        header = 66
        panel = Image.new('RGB', (3*w, h+header), '#18202b')
        panel.paste(im, (0, header)); panel.paste(overlay, (w, header))
        panel.paste(Image.fromarray(mask).convert('RGB'), (2*w, header))
        pd = ImageDraw.Draw(panel)
        for i, title in enumerate(['Original rectified RGB', 'Returned polygon + point IDs (1-based)', 'Rasterized returned polygon (NOT repaired)']):
            pd.text((i*w+14, 9), f'{path.stem} | {title}', font=font(24), fill='white')
        pd.text((14, 39), f"{source['session']} frame {source['frame']} | visibility UNKNOWN | observation only", font=font(19), fill='#ffcf71')
        panel.save(args.output/f'{path.stem}_review.jpg', quality=94)
        previews.append(panel.resize((1296, 506)))
        row = dict(camera=path.stem, source=source, image_sha256=digest,
                   json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   size=[w,h], point_count=len(points), missing_ids=[k+1 for k in range(23) if k not in points],
                   polygon_count=len(polys), polygon_vertices=[len(p) for p in polys], polygon_checks=polygon_checks,
                   raster_components=components-1, component_areas=stats[1:, cv2.CC_STAT_AREA].tolist(),
                   points_outside_polygon=outside, errors=errors,
                   spine_abdomen_distance_px=float(np.linalg.norm(points[5]-points[22])) if 5 in points and 22 in points else None,
                   tail_tip_edge_distance_px=float(min(points[13][0],points[13][1],w-1-points[13][0],h-1-points[13][1])) if 13 in points else None,
                   group_ids=sorted(set(str(s.get('group_id')) for s in data['shapes'])),
                   state_fields_present=any(bool(s.get('flags')) or bool(s.get('description')) for s in data['shapes']))
        summaries.append(row)
        parsed.append(dict(camera=path.stem, source=source, train_allowed=False, acceptance='pending_review',
                           instance_id=None, visibility='unknown',
                           points=[dict(core23_id=k, name=NAMES[k], xy=xy.tolist(), observation_present=True,
                                        visibility='unknown', semantic_verified=False, train_valid=False) for k,xy in points.items()],
                           missing_point_status='unknown_not_negative', raw_shapes=data['shapes']))
    sheet = Image.new('RGB', (1296, 506*len(previews)), '#18202b')
    for i, preview in enumerate(previews):
        sheet.paste(preview, (0, 506*i))
    sheet.save(args.output/'all_views_contact_sheet.jpg', quality=94)
    summary = dict(source=str(args.source), train_allowed=False, views=summaries,
                   total_points=sum(r['point_count'] for r in summaries),
                   point_definitions={str(k+1):v for k,v in enumerate(NAMES)})
    (args.output/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.output/'parsed_observations.json').write_text(json.dumps(parsed, ensure_ascii=False, indent=2))
    links = ''.join(f'<h2>{r["camera"]}</h2><img src="{r["camera"]}_review.jpg" style="width:100%">' for r in summaries)
    legend = '<br>'.join(f'{k+1:02d}: {name}' for k,name in enumerate(NAMES))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AZL3 first-frame annotation review</title>'
        '<body style="background:#18202b;color:white;font-family:sans-serif"><h1>AZL3 frame 0: returned annotations</h1>'
        '<p>7 views. Original / numbered contour overlay / raw raster mask. Not accepted training GT. '
        'Cyan=left, pink=right (animal anatomy); yellow=tail. No inferred skeleton connections.</p>'+links+'<h2>Point legend</h2>'+legend+'</body>')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('source','image_sha256','json_sha256')} for r in summaries], indent=2))


if __name__ == '__main__':
    main()
