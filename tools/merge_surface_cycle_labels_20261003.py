"""Versioned manual-label merge. Raw sources remain immutable; exclusions persist."""
import collections, copy, hashlib, json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from review_pymousestudio_labelme_return import NAMES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'
OUT = ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1'
CYCLE = ROOT/'output/surface_curve_cycle_20261003'
PROTECTED = set(range(8, 15)) | {23}
FOCUS = {6, 7, 15, 16, 17, 18}

def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def points(a):
    out = collections.defaultdict(list)
    for s in a['shapes']:
        if s['shape_type'] == 'point': out[int(s['label'][:2])].append(s)
    return dict(out)
def polygons(a): return [s for s in a['shapes'] if s['shape_type'] == 'polygon']
def shape_signature(x): return json.dumps(x, sort_keys=True, ensure_ascii=False)

def main():
    assert not OUT.exists(), 'Immutable version already exists'
    manifest = load(BASE/'ingestion_manifest.json')
    splitplan = load(ROOT/'output/teacher_temporal_cycle_20261001/plan.json')
    review = load(ROOT/splitplan['exclusion_manifest'])
    exclusions = {(r['session'], r['frame']) for r in review['cases']}
    assert len(exclusions) == 24
    audit1 = load(ROOT/'output/pymousestudio_repair_audit_20261002/summary.json')
    audit2 = load(ROOT/'output/pymousestudio_repair_audit_20261003/summary.json')
    pkg1 = Path(audit1['extracted'])
    cases = {r['id']: r for r in review['cases']}
    ret1 = {}
    for r in audit1['files']:
        p = pkg1/cases[r['case']]['folder']/'originals'/r['source']
        if r['source'] in ret1: assert sha(ret1[r['source']]) == sha(p)
        ret1[r['source']] = p
    ret2 = {(r['session'], r['frame'], r['camera']): Path(r['annotation']) for r in audit2['files']}
    sent = next(p for p in (ROOT/'output/pymousestudio_repair_audit_20261003/sent_package').iterdir() if p.is_dir())
    rejected_key = ('FR2', 584, 'B')
    old_r12 = sent/'human_review_pilot/R12_FR2_f000584/camB.json'
    assert old_r12.is_file() and rejected_key in ret2
    assert polygons(load(old_r12)) != polygons(load(ret2[rejected_key]))
    semantics = NAMES.copy()
    for i, n in [(5, 'dorsal_thorax_surface'), (6, 'dorsal_lumbar_surface'), (22, 'ventral_abdomen_surface')]: semantics[i] = n
    rows, groups, pending, source_hashes = [], [], [], {}
    def source(p):
        source_hashes[p.relative_to(ROOT).as_posix()] = sha(p)
        return p.relative_to(ROOT).as_posix()
    for g in manifest['frame_groups']:
        role = 'locked_holdout' if g['split'] == 'real_locked_holdout' else 'validation' if g['session'] in splitplan['development_validation_sessions'] else 'train'
        excluded = (g['session'], g['frame']) in exclusions
        uv = np.full((len(g['cameras']), 1, 23, 2), np.nan)
        group_rows = []
        for vi, cam in enumerate(g['cameras']):
            key = (g['session'], g['frame'], cam)
            rel = Path('real_images')/g['split']/g['session']/f"frame_{g['frame']:06d}"/f'cam{cam}.json'
            bp = BASE/rel; a = load(bp); ps = points(a); masks = polygons(a)
            origins = {str(k): source(bp) for k in range(1, 24)}; mask_origin = source(bp)
            # Only the approved body slots are promoted from the first repair.
            # Protected abdomen/tail retain the original coordinates, even if ambiguous.
            if rel.as_posix() in ret1:
                rp = ret1[rel.as_posix()]; b = load(rp); qs = points(b); rs = source(rp)
                for k in range(1, 24):
                    if ps.get(k, []) == qs.get(k, []): continue
                    if k in PROTECTED:
                        pending.append(dict(image=str(rel),stage='return1',id=k,reason='protected_tail_or_abdomen',candidate=rs))
                    else: ps[k] = qs.get(k, []); origins[str(k)] = rs
                masks = polygons(b); mask_origin = rs
            if key in ret2:
                rp = ret2[key]; b = load(rp); qs = points(b); rs = source(rp)
                for k in range(1, 24):
                    if ps.get(k, []) == qs.get(k, []): continue
                    if k in FOCUS: ps[k] = qs.get(k, []); origins[str(k)] = rs
                    else: pending.append(dict(image=str(rel),stage='return2',id=k,reason='outside_review_scope',candidate=rs))
                masks = polygons(b); mask_origin = rs
            if key == rejected_key:
                masks = polygons(load(old_r12)); mask_origin = source(old_r12)
                assert masks != polygons(load(ret2[key]))
            merged = copy.deepcopy(a); merged['shapes'] = [s for k in range(1, 24) for s in ps.get(k, [])] + copy.deepcopy(masks)
            target = OUT/rel
            # Keep one canonical copy of RGB; Labelme uses a valid relative imagePath.
            image = bp.with_suffix('.png')
            merged['imagePath'] = os.path.relpath(image, target.parent).replace('\\', '/')
            merged['imageData'] = None
            merged['mousegeoforge_label_version'] = '20261003_effective_v1'
            assert (merged['imageWidth'], merged['imageHeight']) == (1080, 1200)
            issues = []
            for k, shapes in ps.items():
                if len(shapes) != 1:
                    if shapes: issues.append(f'duplicate_point_{k}')
                    continue
                xy = np.asarray(shapes[0]['points'], float)
                if xy.shape != (1, 2) or not np.isfinite(xy).all() or not (0 <= xy[0, 0] < 1080 and 0 <= xy[0, 1] < 1200):
                    issues.append(f'invalid_point_{k}'); continue
                uv[vi, 0, k-1] = xy[0]
            mask = Image.new('L', (1080, 1200)); d = ImageDraw.Draw(mask)
            mask_valid = bool(masks)
            for s in masks:
                xy = np.asarray(s['points'], float)
                valid = s['label'] == 'mouse' and xy.ndim == 2 and xy.shape[1] == 2 and len(xy) >= 3 and np.isfinite(xy).all()
                valid = valid and (xy >= 0).all() and (xy[:, 0] < 1080).all() and (xy[:, 1] < 1200).all()
                if not valid: issues.append('invalid_mask_polygon'); mask_valid = False
                elif not s.get('flags', {}).get('hole', False): d.polygon([tuple(p) for p in xy], fill=255)
                else: issues.append('unsupported_polygon_hole'); mask_valid = False
            save(target, merged)
            mp = target.with_name(target.stem+'_mask.png'); mask.save(mp)
            if key == rejected_key:
                old_mask = Image.new('L', (1080,1200)); dd = ImageDraw.Draw(old_mask)
                for s in polygons(load(old_r12)): dd.polygon([tuple(p) for p in s['points']], fill=255)
                assert np.array_equal(mask, old_mask)
                assert polygons(load(target)) == polygons(load(old_r12))
            rec = dict(session=g['session'],frame=g['frame'],camera=cam,role=role,excluded_group=excluded,
                       annotation=target.relative_to(ROOT).as_posix(),source_image=source(image),
                       mask=mp.relative_to(ROOT).as_posix(),mask_valid=mask_valid,issues=issues,
                       point_source=origins,mask_source=mask_origin,
                       explicit_visibility=False,mask_area_px=int(np.count_nonzero(np.asarray(mask))),
                       mask_sha256=sha(mp),annotation_sha256=sha(target))
            rows.append(rec); group_rows.append(rec)
        out_npz = OUT/'normalized_observations'/g['split']/g['session']/f"frame_{g['frame']:06d}.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        eligible = not excluded and role != 'locked_holdout'
        np.savez_compressed(out_npz, uv=uv, xy_observed=np.isfinite(uv).all(-1), names=np.asarray(NAMES),semantic_names=np.asarray(semantics),
                            camera_names=np.asarray(g['cameras']),instance_ids=np.asarray([0]),frame=np.asarray(g['frame']),
                            visibility_state=np.full(uv.shape[:-1], -1, np.int8),
                            loss_eligible=np.isfinite(uv).all(-1)&eligible,train_allowed=np.asarray(eligible and role=='train'),
                            role=np.asarray(role),excluded_group=np.asarray(excluded),source_frame=np.asarray(g['frame']))
        groups.append(dict(**g,role=role,excluded=excluded,effective_normalized_file=out_npz.relative_to(ROOT).as_posix(),eligible_for_development=eligible))
    assert len(rows)==1950 and len(groups)==300
    assert sum(g['excluded'] for g in groups)==24
    assert not any(g['eligible_for_development'] for g in groups if g['session'] in splitplan['locked_holdout_sessions'])
    for p, digest in source_hashes.items(): assert sha(ROOT/p)==digest, p
    r12 = next(r for r in rows if (r['session'], r['frame'], r['camera'])==rejected_key)
    result = dict(version='20261003_effective_v1',status='merged_for_controlled_development_not_strong_3d_gt',
                  images=len(rows),groups=len(groups),roles=dict(collections.Counter(r['role'] for r in rows)),excluded_groups=24,
                  preserved_protected_ids=sorted(PROTECTED),pending_edits=len(pending),
                  rejected_return2_mask=r12,source_files_unchanged=True,
                  calibration_source=(BASE/'calibration').relative_to(ROOT).as_posix(),
                  mask_policy='Rasterized manual polygon union; invalid polygons are excluded from mask loss, not silently clipped. Not a topology certification.',
                  visibility_policy='Unknown explicit visibility; missing coordinates are unavailable, never invisible.',
                  provenance=rows,frame_groups=groups,source_hashes=source_hashes)
    save(OUT/'manifest.json',result); save(OUT/'pending_edits.json',pending)
    save(OUT/'exclusions.json',review); save(OUT/'split_policy.json',splitplan)
    save(CYCLE/'real_merge_verification.json',{k:v for k,v in result.items() if k not in ('provenance','frame_groups','source_hashes')})
    save(ROOT/'dataset/PyMouseStudio/annotation_versions/current.json',dict(version=result['version'],manifest=(OUT/'manifest.json').relative_to(ROOT).as_posix(),automatic_global_training_import=False))
    state=load(CYCLE/'state.json');state['stage']='real_labels_merged';state['completed'].append('merge_real_annotations_with_field_provenance_and_exclusions');state['next']='Promote accepted final Blender assets, verify dependencies, archive iterations, render all-view pilot.';save(CYCLE/'state.json',state)
    print(json.dumps({k:v for k,v in result.items() if k in ('images','groups','roles','pending_edits','excluded_groups')},indent=2),flush=True)

if __name__ == '__main__': main()
