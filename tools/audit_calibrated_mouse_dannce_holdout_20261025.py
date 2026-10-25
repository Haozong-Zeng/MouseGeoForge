"""Read-only split/provenance audit; no 4D-GS import, image scoring, or model call."""
from pathlib import Path
import ast,datetime as dt,hashlib,json,os,sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
OUT=ROOT/'output/calibrated_mouse_next12h_20261025'
BASE=ROOT/'output/DANNCE-4DGS-Pilot-mm1-f0000-0079'
LOADER=ROOT/'third_party/4DGaussians/scene/multipleview_dataset.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parse_namespace(text):
 tree=ast.parse(text,mode='eval').body
 if not isinstance(tree,ast.Call) or not isinstance(tree.func,ast.Name) or tree.func.id!='Namespace' or tree.args:raise ValueError('Expected literal Namespace keywords')
 if any(k.arg is None for k in tree.keywords):raise ValueError('No Namespace expansions')
 return {k.arg:ast.literal_eval(k.value) for k in tree.keywords}

def replay_split(loader_source,frame_count,camera_count,folder):
 """Execute only exact path-enumerator AST with in-memory directory inventory and dummy poses."""
 if frame_count<2 or camera_count<1:raise ValueError('Invalid fixture counts')
 tree=ast.parse(loader_source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='multipleview_dataset');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='load_images_path')
 module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace=dict(np=np,os=SimpleNamespace(path=os.path,listdir=lambda _: [f'frame_{i+1:05d}.jpg' for i in range(frame_count)]),qvec2rotmat=lambda _:np.eye(3))
 exec(compile(module,'<pinned-loader-path-method>','exec'),namespace)
 extr={i:SimpleNamespace(name=f'frame{i:02d}.jpg',qvec=[1,0,0,0],tvec=[0,0,0]) for i in range(1,camera_count+1)};results={}
 for split in ['train','test']:
  paths,poses,times=namespace['load_images_path'](None,str(folder),extr,{},split);assert len(paths)==len(poses)==len(times);results[split]=[dict(path=p,time=t) for p,t in zip(paths,times)]
 return results

def evidence(path,start,end):
 lines=path.read_text().splitlines();return dict(path=str(path.relative_to(ROOT)),source_sha256=sha(path),first_line=start,last_line=end,text='\n'.join(lines[start-1:end]))

def main():
 dest=OUT/'dannce_holdout_audit_v1.json';assert not dest.exists();registration=OUT/'registrations/dannce_holdout_audit_v1.json';reg=json.loads(registration.read_text())
 for p,h in reg['source_sha256'].items():assert sha(ROOT/p)==h
 for p,h in reg['input_sha256'].items():assert sha(ROOT/p)==h
 cfg=parse_namespace((BASE/'model_v3_foreground/cfg_args').read_text());summary=json.loads((BASE/'visualizations_v3_foreground/visualization_summary.json').read_text());mask_manifest=json.loads((BASE/'sam2_masks_v1/mask_manifest.json').read_text());frame_count=summary['frames_per_camera'];camera_count=summary['camera_count'];assert (frame_count,camera_count)==(80,6)
 source=Path(cfg['source_path']);inventory=replay_split(LOADER.read_text(),frame_count,camera_count,source);train={r['path'] for r in inventory['train']};test={r['path'] for r in inventory['test']};overlap=sorted(train&test)
 assert len(train)==480 and len(test)==18 and len(overlap)==18
 selected=[dict(camera=r['camera'],frame=r['frame'],path=str(source/f"cam{r['camera']:02d}/frame_{r['frame']+1:05d}.jpg")) for r in summary['selected_records']];assert len(selected)==30;assert all(r['path'] in train for r in selected)
 mask_files={};counts={}
 for camera in range(1,camera_count+1):
  files=sorted((BASE/f'sam2_masks_v1/masks/cam{camera:02d}').glob('frame_*.png'));assert len(files)==80;assert [f.name for f in files]==[f'frame_{i+1:05d}.png' for i in range(80)];counts[str(camera)]=len(files)
  for f in files:mask_files[str(f.relative_to(ROOT))]=sha(f)
 result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),registration_sha256=sha(registration),model_calls=0,training_updates=0,new_data_admission=False,new_image_or_joint_scoring=False,passed_audit_execution=True,
 current_source_path=dict(path=str(source),exists=source.exists(),historical_input_directory_replay_available=False if not source.exists() else None),
 loader_path_replay=dict(method='Actual pinned load_images_path AST executed with in-memory80frame/6camera inventory; posecalculationstubidentity irrelevantforpathsplit, no officialCUDAmodule import',train_count=len(train),test_count=len(test),train_test_overlap=len(overlap),disjoint_test_count=len(test-train),test_zero_based_frame_indices=[0,26,53],config_eval=cfg['eval'],config_llffhold=cfg['llffhold'],llffhold_affects_multipleview_split=False,inventory=inventory,overlap_paths=overlap),
 stored_render_scores=dict(selected_count=30,all_selected_paths_train_eligible=True,selected_records=selected,exporter_uses_getTrainCameras=True,heldout_accuracy_evidence=False),
 existing_masks=dict(prompt_frames=mask_manifest['prompt_frames_zero_based'],camera_file_counts=counts,total=480,input_SHA256=mask_files),
 dependencies=[dict(name='SourceCOM centering/framing',detail='prepare_pilot medianCOM andmeanCOM target overALL80 sourceframes; no trainonlypartition in preparation. Translationgaugeisnotfittedtoerrors butheldout framingindependence notestablished.'),dict(name='3D prompt and hull bounds',detail='SAM2prompts27/72 projectedreleased3D; foregroundcandidatebounds use3D extrema ofpromptposes andvisualhullvotes fromall6 cameras. Withheldviews/time cannot reuseallview/allprompt assets asindependentinput.'),dict(name='Temporal propagation',detail='SAM2video forwardandbackward propagation usesclipframes; splittingpointsafterthis pipeline doesnot establishindependent labels.'),dict(name='Orbit renders',detail='Novel interpolatedcamera orbit has no independentRGB/surfaceGT reference in storedexports; plausibleappearance notheldoutquantitativegeometryproof.')],
 evidence=[evidence(LOADER,30,57),evidence(ROOT/'third_party/4DGaussians/scene/dataset_readers.py',596,607),evidence(ROOT/'third_party/4DGaussians/scene/__init__.py',58,74),evidence(ROOT/'tools/export_dannce_4dgs_visualizations.py',211,232),evidence(ROOT/'tools/prepare_dannce_4dgs_pilot.py',226,247),evidence(ROOT/'tools/prepare_dannce_4dgs_foreground.py',197,245),evidence(ROOT/'tools/segment_dannce_4dgs_sam2.py',190,219)],
 conclusion='ExistingV3 is observed-view appearance/weakdepth/consistency evidence. CurrentMultipleViewtestsubsetisnotdisjoint; noindependentheldoutgeometryclaim fromeval=True orllffhold8.',
 historical_limits=['Currentpinnedloader source reconstruction, not archived historical perupdateimageexposure proof','Originalpreparedinputsourcepath currentlymissing; cannot enumerateactualhistoricalJPEGbytes or reconstructofficialdatasetonthishost withoutadditionalprovenance','No rawDANNCE labels admitted fornewtraining/scoring by thisaudit; sparse internal/anatomicaljoints cannotbe usedasvisible-skinGT','Expectedalpha-normalizedcameraZ depth semantics unchanged; no firstopaque-surface inference'],
 prospective_protocol_not_launched=dict(status='proposalonly; no newGSjob/dataadmission',view_holdout='Choosecamera beforepreparation; withholdRGB/mask/prompt/hullvotefromoptimization; calibratedextrinsicsmayremainknown. UseheldoutRGB/masks onlyforpreviouslyadmittedevaluation, masks requireindependentreference.',time_holdout='Choose contiguous timeinterval beforepreparation; alltrainingCOM/framing/bounds/prompt propagation/init andtemporalinputs mustfollowdeclaredconditioning. Classifyinterpolationvsextrapolation; neverclaimindependenttimeifheldoutframesconditionthemodel.',metrics='Appearance/silhouette onadmittedheldoutviews separatefromsurfaceGT. Densemmaccuracy requiresindependentvisible-surface reference; no internaljointlookup metric substitution.',budget='DoNOTretrainexistingpilot; currentcyclekeepread-onlyuntilvalidadmissionandfiniteprotocol established'),source_sha256=reg['source_sha256'],input_sha256=reg['input_sha256'])
 dest.write_text(json.dumps(result,indent=2)+'\n');compact={k:v for k,v in result.items() if k not in ['evidence','existing_masks']};compact['loader_path_replay']={k:v for k,v in result['loader_path_replay'].items() if k not in ['inventory','overlap_paths']};compact['stored_render_scores']={k:v for k,v in result['stored_render_scores'].items() if k!='selected_records'};compact['mask_count_by_camera']=counts;compact['full_audit_sha256']=sha(dest);(ROOT/'docs/experiments/calibrated_mouse_next12h_dannce_holdout_20261025.json').write_text(json.dumps(compact,indent=2)+'\n');print(json.dumps(dict(train=len(train),test=len(test),overlap=len(overlap),disjoint_test=len(test-train),selected_train=len(selected),source_path_exists=source.exists(),model_calls=0)))
if __name__=='__main__':main()
