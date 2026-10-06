"""Create a derived DLC dataset with strict source-session split and exclusions."""
import os
os.environ['DLClight']='True'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/teacher_temporal_cycle_20261001'
BASE=ROOT/'dataset/PyMouseStudio/annotation_returns/20260930_full_v2'

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    prepared=OUT/'dlc_prepared.json'
    assert not prepared.exists(), 'Already prepared; never recreate a training project.'
    manifest=json.loads((BASE/'ingestion_manifest.json').read_text())
    review=json.loads((ROOT/plan['exclusion_manifest']).read_text())['cases']
    excluded={(r['session'],r['frame']) for r in review}
    dev=[g for g in manifest['frame_groups'] if g['split']=='real_development' and (g['session'],g['frame']) not in excluded]
    assert len(dev)==205
    srcvideo=ROOT/'dataset/PyMouseStudio/sessions/AZL3/videos/camA.mp4'
    project_config=Path(deeplabcut.create_new_project('PyMouseStudioBodyTeacher','manualv3',[str(srcvideo)],working_directory=OUT,copy_videos=False))
    cfg=auxiliaryfunctions.read_config(project_config);project=project_config.parent
    ids=np.array(plan['approved_body_ids_1based'])-1
    names=['nose','dorsal_thorax_surface','dorsal_lumbar_surface','left_forepaw_center','right_forepaw_center','left_hindpaw_center','right_hindpaw_center']
    cfg['bodyparts']=names;cfg['skeleton']=[];cfg['default_net_type']='resnet_50';cfg['engine']='pytorch'
    cfg['default_augmenter']='default';cfg['batch_size']=8;cfg['snapshotindex']=-1
    cfg['pcutoff']=0.1;cfg['dotsize']=3
    destination=project/'labeled-data/camA'
    destination.mkdir(exist_ok=True,parents=True)
    entries=[];values=[];indices=[]
    A=np.array([[691/1080,0,6],[0,768/1200,0],[0,0,1]],float)
    for g in sorted(dev,key=lambda x:(x['session'],x['frame'])):
        assert g['session'] not in plan['locked_holdout_sessions']
        with np.load(BASE/g['normalized_file']) as z:uv=z['uv'][:,0,ids]
        for vi,cam in enumerate(g['cameras']):
            source=BASE/'real_images'/g['split']/g['session']/f"frame_{g['frame']:06d}"/f'cam{cam}.png'
            filename=f"{g['session']}__{g['frame']:06d}__cam{cam}.png"
            target=destination/filename
            assert not target.exists()
            with Image.open(source) as im:
                assert im.size==(1080,1200)
                resized=im.convert('RGB').resize((691,768),Image.Resampling.LANCZOS)
                canvas=Image.new('RGB',(704,768),(0,0,0));canvas.paste(resized,(6,0));canvas.save(target)
            transformed=uv[vi]*np.array([A[0,0],A[1,1]])+A[:2,2]
            valid=np.isfinite(uv[vi]).all(-1)
            assert np.isfinite(transformed).all(-1).tolist()==valid.tolist()
            back=(transformed-A[:2,2])/np.array([A[0,0],A[1,1]])
            np.testing.assert_allclose(back,uv[vi],atol=1e-8,equal_nan=True)
            index=('labeled-data','camA',filename)
            role='validation' if g['session'] in plan['development_validation_sessions'] else 'train'
            assert g['session'] in plan['training_sessions']+plan['development_validation_sessions']
            indices.append(index);values.append(transformed.ravel())
            entries.append(dict(index=list(index),role=role,session=g['session'],frame=g['frame'],camera=cam,
                                source=source.relative_to(ROOT).as_posix(),derived=target.relative_to(ROOT).as_posix(),
                                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),xy_observed=valid.tolist()))
    columns=pd.MultiIndex.from_product([['manualv3'],names,['x','y']],names=['scorer','bodyparts','coords'])
    df=pd.DataFrame(values,index=pd.MultiIndex.from_tuples(indices),columns=columns).sort_index()
    lookup={tuple(e['index']):e for e in entries};entries=[lookup[tuple(idx)] for idx in df.index]
    train=[i for i,e in enumerate(entries) if e['role']=='train'];valid=[i for i,e in enumerate(entries) if e['role']=='validation']
    cfg['TrainingFraction']=[round(len(train)/len(entries),2)]
    auxiliaryfunctions.write_config(project_config,cfg)
    df.to_hdf(destination/'CollectedData_manualv3.h5',key='df_with_missing',mode='w')
    result=dict(project_config=str(project_config),framework=deeplabcut.__version__,names=names,ids_1based=(ids+1).tolist(),
                input_wh=[704,768],source_wh=[1080,1200],affine_source_to_input=A.tolist(),
                train_indices=train,validation_indices=valid,train_images=len(train),validation_images=len(valid),
                development_groups=205,excluded_groups=24,locked_holdout_loaded=False,entries=entries)
    (OUT/'dlc_dataset_manifest.json').write_text(json.dumps(result,indent=2))
    deeplabcut.create_training_dataset(str(project_config),Shuffles=[1],trainIndices=[train],testIndices=[valid],
        net_type='resnet_50',engine=deeplabcut.Engine.PYTORCH,userfeedback=False)
    # Verify DLC's merged table is exactly the row ordering used for session separation.
    merged=list((project/'training-datasets').rglob('CollectedData_manualv3.h5'))
    assert len(merged)==1
    pd.testing.assert_frame_equal(pd.read_hdf(merged[0]),df)
    prepared.write_text(json.dumps({k:result[k] for k in ['project_config','framework','names','ids_1based','input_wh','train_images','validation_images','development_groups','excluded_groups']},indent=2))
    print(prepared.read_text(),flush=True)

if __name__=='__main__':main()
