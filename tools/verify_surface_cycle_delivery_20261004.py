"""Read-only delivery integrity and targeted regression audit; no training."""
import datetime as dt, fcntl, hashlib, importlib.util, io, json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vggt-omega')]
import numpy as np
from PIL import Image
CYCLE=ROOT/'output/surface_curve_cycle_20261003'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    report={'utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    dataset=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview'
    metadata=read(dataset/'metadata.json')
    for r in metadata['files']:
        p=dataset/r['path'];assert p.stat().st_size==r['bytes'];assert sha(p)==r['sha256'],str(p)
    report['dataset_files_sha_verified']=len(metadata['files'])
    assert metadata['images']==21 and metadata['temporal_reliable'] is False
    final=read(ROOT/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-Final/final_manifest.json')
    for r in final['records']:assert sha(ROOT/r['final'])==r['sha256']
    report['final_blends_sha_verified']=len(final['records'])
    archive=read(CYCLE/'archive_manifest.json');count=0
    for group in archive:
        for f in group['files']:
            assert (ROOT/group['destination']/f['path']).stat().st_size==f['bytes'];count+=1
    report['archive_files_exist_size_verified']=count
    report['archive_hash_scope']='SHA verified at move; final audit rechecks existence/size, not full38GiB SHA.'
    labels=read(ROOT/'dataset/PyMouseStudio/annotation_versions/20261003_effective_v1/manifest.json')
    for path,digest in labels['source_hashes'].items():assert sha(ROOT/path)==digest,path
    report['immutable_source_hashes_verified']=len(labels['source_hashes'])
    for row in labels['provenance']:
        assert sha(ROOT/row['annotation'])==row['annotation_sha256']
        if row['mask_valid']:assert sha(ROOT/row['mask'])==row['mask_sha256']
    report['effective_annotations_sha_verified']=len(labels['provenance'])
    r12=labels['rejected_return2_mask']
    old=read(ROOT/r12['mask_source']);current=read(ROOT/r12['annotation'])
    polygon=lambda a:[s for s in a['shapes'] if s['shape_type']=='polygon']
    assert polygon(old)==polygon(current)
    assert sha(ROOT/r12['mask'])=='489078833bd766573be17a405a0185fe14883d7d88205824a1eb9d5f77fa5d3d'
    with Image.open(ROOT/r12['mask']) as im:area=int((np.asarray(im)>0).sum())
    assert area==146504
    report['r12_exact_old_polygon_and_mask']={'passed':True,'area_px':area}
    assert labels['excluded_groups']==24 and labels['pending_edits']==135
    report['loader_qa']=read(dataset/'loader_qa.json');assert report['loader_qa']['passed']
    report['depth_qa']=read(dataset/'independent_geometry_qa.json')
    patterns=['test_surface_curve_supervision.py','test_curve_material_loss.py','test_surface_curve_model.py',
              'test_surface_curve_refinement.py','test_sparse_surface_observations.py','test_uncertain_back_points.py',
              'test_back_point_integral.py','test_surface_curve_integral.py','test_joint_surface_point_integral.py',
              'test_point_heatmap_supervision.py','test_surface_curve_projection.py','test_multiview_point_audit.py',
              'test_controlled_back_noise.py']
    suite=unittest.TestSuite();loader=unittest.TestLoader()
    for pattern in patterns:suite.addTests(loader.discover(str(ROOT/'tests'),pattern=pattern))
    spec=importlib.util.spec_from_file_location('curve_function_tests',ROOT/'tests/test_surface_curve_supervision.py')
    functions=importlib.util.module_from_spec(spec);spec.loader.exec_module(functions)
    for name in sorted(dir(functions)):
        if name.startswith('test_'):suite.addTest(unittest.FunctionTestCase(getattr(functions,name)))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report['tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'output':stream.getvalue()}
    assert result.wasSuccessful(),stream.getvalue()
    with (ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report['shared_gpu_lock_free']=True
    for name in ('controlled_back_noise_probe_v1','controlled_back_curve_aux_v1'):
        assert read(CYCLE/name/'status.json')['stage']=='complete'
    report['passed']=True
    (CYCLE/'final_verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('depth_qa','loader_qa','tests')},indent=2));print(stream.getvalue())


if __name__=='__main__':main()
