"""Copy accepted assets to a stable final location; archive only after Blender audit."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
CYL=ROOT/'BlenderPyMouseProjects/Cylinder'
FINAL=CYL/'PyMouseStudio-RealCapture-Final'
ARCHIVE=ROOT/'backup/blender/PyMouseStudio-RealCapture-Iterations-20261003'
STUDIES=['PyMouseStudio-RealCapture-Replica-20261002','PyMouseStudio-RealCapture-EarStudy-20261003','PyMouseStudio-RealCapture-BottomTailFix-20261003']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    if not args.archive:
        assert not FINAL.exists()
        src=CYL/STUDIES[2]/'DELIVERY-Balanced'
        evidence=load(ROOT/'output/bottom_tail_fix_20261003/final_balanced/manifest.json')
        assert evidence['complete']
        FINAL.mkdir();records=[]
        for pose,label in [('climb','Climb'),('quadruped','Quadruped'),('headraise','HeadRaise')]:
            p=src/f'PyMouseStudio-BottomTailFix-{label}-Static.blend';dst=FINAL/f'PyMouseStudio-RealCapture-{label}-Final.blend'
            digest=sha(p);assert digest==evidence['poses'][pose]['blend_sha256']
            shutil.copy2(p,dst);assert sha(dst)==digest
            records.append(dict(pose=pose,source=p.relative_to(ROOT).as_posix(),final=dst.relative_to(ROOT).as_posix(),sha256=digest,frame=evidence['poses'][pose]['frame']))
        # Copy complete accepted evidence/delivery except the 3 large blend duplicates.
        evidence_dst=FINAL/'accepted_evidence';evidence_dst.mkdir()
        for p in src.iterdir():
            if p.suffix=='.blend':continue
            if p.is_dir():shutil.copytree(p,evidence_dst/p.name)
            else:shutil.copy2(p,evidence_dst/p.name)
        save(FINAL/'final_manifest.json',dict(status='accepted_visuals_pending_relocation_preflight',records=records,
             source='DELIVERY-Balanced, not rejected DELIVERY',static_poses_not_animation=True,
             limitations=evidence['limitations'],renderer='BlenderScripts/render_surface_curve_dataset_20261003.py'))
        save(CYCLE/'promotion.json',dict(records=records,archive_pending=True))
        print('PROMOTED',len(records))
    else:
        audit=load(CYCLE/'blender_preflight.json');assert audit['complete'] and audit['all_dependencies_packed']
        assert not ARCHIVE.exists();ARCHIVE.mkdir(parents=True)
        inventory=[]
        for name in STUDIES:
            src=(CYL/name).resolve();dest=(ARCHIVE/name).resolve()
            assert src.parent==CYL.resolve() and dest.parent==ARCHIVE.resolve() and src.is_dir() and not dest.exists()
            files=[dict(path=p.relative_to(src).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(src.rglob('*')) if p.is_file()]
            assert files
            row=dict(source=src.relative_to(ROOT).as_posix(),destination=dest.relative_to(ROOT).as_posix(),files=files,status='planned')
            inventory.append(row);save(CYCLE/'archive_manifest.json',inventory)
            # Same-filesystem atomic rename. No deletion and no cross-shell path construction.
            src.rename(dest)
            assert not src.exists()
            for f in files:assert sha(dest/f['path'])==f['sha256']
            row['status']='moved_verified';save(CYCLE/'archive_manifest.json',inventory)
        m=load(FINAL/'final_manifest.json');m['status']='final_relocation_verified';m['archive_mapping']=(CYCLE/'archive_manifest.json').relative_to(ROOT).as_posix();save(FINAL/'final_manifest.json',m)
        state=load(CYCLE/'state.json');state['completed'].append('promote_balanced_final_blends_and_archive_only_known_iterations');state['next']='Render and validate new all-view static-pose dataset before training.';save(CYCLE/'state.json',state)
        print('ARCHIVED',sum(len(x['files']) for x in inventory),'files; reversible via archive_manifest.json')
if __name__=='__main__':main()
