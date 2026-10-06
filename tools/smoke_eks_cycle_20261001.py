"""CPU-only official EKS import and calibrated projection/smoother smoke test."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['NUMBA_DISABLE_JIT']='1'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json,time
from pathlib import Path
import numpy as np
import jax
from importlib.metadata import version
from aniposelib.cameras import Camera,CameraGroup
from eks.marker_array import MarkerArray
from eks.multicam_smoother import ensemble_kalman_smoother_multicam,make_projection_from_camgroup

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/teacher_temporal_cycle_20261001'
def main():
    print('VERSIONS',{n:version(n) for n in ['ensemble-kalman-smoother','jax','jaxlib','numpy','dynamax']},flush=True)
    cams=[Camera(matrix=np.array([[800,0,540],[0,800,600],[0,0,1.]]),dist=np.zeros(5),rvec=np.zeros(3),tvec=np.array(t),name=c,size=(1080,1200)) for c,t in zip('ABC',[[-50.,0,0],[0,-50.,0],[50.,0,0]])]
    cg=CameraGroup(cams);T=40;K=2;t=np.arange(T)
    xyz=np.stack([np.stack([10*np.sin(t/10)+k*5,5*np.cos(t/12)+k*3,np.full(T,250.)],-1) for k in range(K)],1)
    uv=cg.project(xyz.reshape(-1,3)).reshape(3,T,K,2)
    project,_=make_projection_from_camgroup(cg)
    projected=np.asarray(jax.vmap(project)(xyz.reshape(-1,3))).reshape(T,K,3,2).transpose(2,0,1,3)
    maxdiff=float(np.max(np.abs(projected-uv)));assert maxdiff<.001,maxdiff
    rng=np.random.default_rng(100);xy=uv[None]+rng.normal(0,.5,(3,*uv.shape));raw=np.concatenate([xy,np.ones((*xy.shape[:-1],1))*.95],-1)
    # API checked from the official release; construct a typed marker array.
    markers=MarkerArray(raw,data_fields=['x','y','likelihood'])
    start=time.monotonic()
    dfs,s,df3=ensemble_kalman_smoother_multicam(markers,['p1','p2'],list('ABC'),smooth_param=1.,avg_mode='median',var_mode='var',camgroup=cg)
    vals=df3.to_numpy();assert np.isfinite(vals).all()
    xyz_pred=vals.reshape(T,K,6)[...,:3]
    error=np.linalg.norm(xyz_pred-xyz,axis=-1)
    assert np.median(error)<2 and np.max(error)<5,error.max()
    result=dict(versions={n:version(n) for n in ['ensemble-kalman-smoother','jax','jaxlib','numpy','dynamax']},
        projection_max_error_px=maxdiff,output_3d_shape=list(vals.shape),seconds=time.monotonic()-start,
        synthetic_3d_median_error_mm=float(np.median(error)),synthetic_3d_max_error_mm=float(np.max(error)),
        synthetic_test_only=True,real_data_tested=False,train_allowed=False)
    (OUT/'eks_smoke_result.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':main()
