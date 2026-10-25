"""Verify parsed RGB PLY vertices against saved native pixel/color/depth receipts."""
from pathlib import Path
import importlib.util,json,datetime,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';BASE=OUT/'pointcloud_review_v1'
s=importlib.util.spec_from_file_location('export',ROOT/'tools/export_calibrated_mouse_pointcloud_review_20261025.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=BASE/'binary_roundtrip_verification.json';assert not target.exists()
    completion=json.loads((BASE/'completion.json').read_text());assert completion['model_calls']==0
    for name,h in completion['output_sha256'].items():assert sha(BASE/name)==h
    for name,h in completion['input_sha256'].items():assert sha(ROOT/name)==h
    result={}
    for path in sorted(BASE.glob('*_rgb_world_mm.ply')):
        vertex=m.read_ply(path);world=np.c_[vertex['x'],vertex['y'],vertex['z']].astype(float)
        with np.load(BASE/path.name.replace('_rgb_world_mm.ply','_pixel_receipt.npz')) as z:data={k:z[k] for k in z.files}
        np.testing.assert_array_equal(np.c_[vertex['red'],vertex['green'],vertex['blue']],data['rgb']);np.testing.assert_array_equal(vertex['source_view'],data['view'])
        rows=[]
        for view in range(3):
            keep=data['view']==view;uv,depth=m.project(world[keep],data['K'][view],data['E'][view]);p=float(np.max(np.abs(uv-data['xy'][keep])));e=float(np.max(np.abs(depth-data['z'][keep])));assert p<.001 and e<.0001
            rows.append(dict(view=view,points=int(keep.sum()),decoded_PLY_native_pixel_max_abs=p,decoded_PLY_cameraZ_max_abs_mm=e))
        result[path.name]=rows
    receipt=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),export_source_sha256=sha(ROOT/'tools/export_calibrated_mouse_pointcloud_review_20261025.py'),export_completion_sha256=sha(BASE/'completion.json'),all_recorded_input_and_output_hashes_match=True,PLY_colors_and_source_view_exact=True,pixel_tolerance=.001,cameraZ_mm_tolerance=.0001,clouds=result)
    target.write_text(json.dumps(receipt,indent=2)+'\n');(ROOT/'docs/experiments/calibrated_mouse_pointcloud_binary_verification_20261025.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'max_pixel_error':max(v['decoded_PLY_native_pixel_max_abs'] for rows in result.values() for v in rows),'max_Z_mm_error':max(v['decoded_PLY_cameraZ_max_abs_mm'] for rows in result.values() for v in rows)}))
if __name__=='__main__':main()
