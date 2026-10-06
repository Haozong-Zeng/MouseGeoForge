"""Read-only physical-camera/source coverage audit for current RGB training."""
import collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from audit_bottom_view_coverage import orientation_from_extrinsic
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OUT=CYCLE/'pixel_training_coverage_audit_v1'
def main():
    OUT.mkdir(exist_ok=True);data=torch.load(CYCLE/'point_quality_inputs_v1/inputs.pt',map_location='cpu',weights_only=False)
    records=[];cache={};counts=collections.Counter();dataset_sources=collections.defaultdict(set);dataset_stages=collections.defaultdict(set);selected=collections.defaultdict(set)
    for i,m in enumerate(data['metadata']):
        if m['kind']=='real':orientation='bottom' if m['physical_bottom'] else 'nonbottom_real_unclassified';dataset=m['source']
        else:
            path=ROOT/m['scene']/'scene_meta.json'
            if path not in cache:cache[path]=json.loads(path.read_text())
            v=next(v for v in cache[path]['views'] if v['view_id']==m['view']);elevation,orientation=orientation_from_extrinsic(v['extrinsic']);dataset=Path(m['scene']).parts[2]
            dataset_sources[dataset].add(m['source']);dataset_stages[dataset].add(m['stage']);selected[dataset].add((m['scene'],m['view']))
        counts[m['stage'],m['kind'],dataset,orientation]+=1
        records.append(dict(index=i,stage=m['stage'],kind=m['kind'],dataset=dataset,source=m['source'],scene=m['scene'],view=m['view'],orientation=orientation,known_points=int(data['valid'][i].sum())))
    available=[]
    for dataset,sources in dataset_sources.items():
        if dataset_stages[dataset]!={'fit'} or sources!={'moseq_sample10'}:continue
        root=ROOT/'dataset/BlenderPyMouse'/dataset;frames=sorted(root.glob('frame_*/scene_meta.json'));summary=collections.Counter();samples={};bad=[]
        for f in frames:
            meta=json.loads(f.read_text())
            for v in meta['views']:
                elevation,orientation=orientation_from_extrinsic(v['extrinsic']);summary[orientation]+=1;samples.setdefault(orientation,dict(scene=f.parent.relative_to(ROOT).as_posix(),view=v['view_id'],camera=v.get('camera'),elevation=elevation))
            if not (f.parent/'keypoints.npz').is_file():bad.append(f.parent.name)
        available.append(dict(dataset=dataset,source_group='moseq_sample10',source_evidence='same existingTRAINdataset source_group in current qualitymetadata; extra frame-specific pose provenance stilltoverify before sampling',rendered_frames=len(frames),rendered_view_counts=dict(summary),currently_selected_images=len(selected[dataset]),examples=samples,missing_keypoint_frames=bad))
    note=dict(dataset='PyMouseStudio-RealCapture-SurfaceCurve-Multiview',images=21,poses=3,
        prior_omission='Excluded from current pixel/qualitystage because source_group missing; original files and previousbasecheckpoints unchanged.',
        findings=['finalmetadata31/111/216; realism sourceplan originalBrownFurClimbWall rig/action. build_realism_quadruped_20261003 imports action fromBrownFurClimbWall and picks111; headraisev027 sets216. DatasetClimbWalltemporal metadata identifiesframes1-91 asmouse_climb_wall_91, so31overlaps thiscalibrationsource;111/216originalmotionsegments remainunresolved and mustnotallbecalledclimb solelyfromblendfilename.', 'Geometry seeds/material/cameras explicitly fitted to realdevelopment FR2 andAZR2L2 in build scripts; notindependent of development. Atleastclimb sourceoverlap plusunresolvedsegments prevents silentlyrestoringtheseimageswithoutregrouping.'],
        decision='Do not silently add21images tofit orcall newindependentdata. Motion-source andrealreference dependencies bothmustbelisted; existingbaselinealreadytrainedwiththesehistorically, so calonlyheldfornewstage.',
        evidence=[dict(path=p,sha256=fingerprint(ROOT/p)) for p in ['dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview/metadata.json','BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Final/final_manifest.json','output/pymousestudio_realism_cycle_20261002/plan.json','BlenderScripts/build_realism_quadruped_20261003.py','BlenderScripts/build_realism_headraise_v027_20261003.py']])
    result=dict(scope='Blender Z-up extrinsic physicalorientation; real onlyknowncamA bottom semantic, not arbitraryrealZ.',current_counts=[dict(stage=k[0],kind=k[1],dataset=k[2],orientation=k[3],images=v) for k,v in sorted(counts.items())],available_existing_fit_sources=available,realcapture_source_audit=note,records=records)
    write_json(OUT/'coverage.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['records','realcapture_source_audit']},indent=2))
if __name__=='__main__':main()
