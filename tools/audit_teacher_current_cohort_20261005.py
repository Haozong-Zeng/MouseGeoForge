"""Read-only, same-current-label teacher comparison; never replay old cycle entrypoints."""
import argparse,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'vggt-omega')]
import torch,numpy as np
from PIL import Image,ImageDraw
from train_pixel_adapter_matched_20261005 import prepare_rows,KP
from run_depth_pose_ablation import write_json,fingerprint
from run_surface_curve_pilot_20261003 import load_item
from review_pixel_adapter_20261005 import font
CYCLE=ROOT/'output/singleview_fusion_cycle_20261005';OLD=ROOT/'output/bottom_anchor_cycle_20261002';OUT=CYCLE/'teacher_current_cohort_cache_audit_v3'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-SingleView-Fusion-20261005')/OUT.name
def stats(x):
    z=np.array(x,float);return dict(n=len(z),mean=float(z.mean()) if len(z) else None,median=float(np.median(z)) if len(z) else None,p95=float(np.quantile(z,.95)) if len(z) else None,pck32=float((z<=32).mean()) if len(z) else None)
def main():
    global OUT,DEST
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');args=ap.parse_args();fresh=None
    if args.fresh:
        OUT=CYCLE/'teacher_current_cohort_fresh_review_v1';DEST=DEST.parent/OUT.name;fresh=np.load(CYCLE/'teacher_current_rgb_replay_v1/predictions.npz');fresh={int(i):p for i,p in zip(fresh['indices'],fresh['current'])}
    assert not OUT.exists();OUT.mkdir();DEST.mkdir(exist_ok=True);torch.set_num_threads(2)
    dm=json.loads((OLD/'superanimal_quadruped_hrnet_v1/dataset_manifest.json').read_text());entries=[dm['entries'][i] for i in dm['validation_indices']]
    assert all(e['role']=='validation' and e['session'] in ['FR2','AZR2L2'] for e in entries)
    trainkeys={(dm['entries'][i]['session'],dm['entries'][i]['frame'],dm['entries'][i]['camera']) for i in dm['train_indices']};lookup={(e['session'],e['frame'],e['camera']):i for i,e in enumerate(entries)};assert len(lookup)==281
    cached=np.load(OLD/'quadruped_epoch110_review_v1/predictions_source_pixels.npz');assert cached['ids'].tolist()==dm['ids_1based'];assert cached['context_manifest_indices'].tolist()==[dm['context_manifest_indices'][i] for i in dm['validation_indices']]
    ci=[dm['ids_1based'].index(k+1) for k in KP];rows=prepare_rows();selected=[(i,r) for i,r in enumerate(rows) if r['metadata']['stage']=='validation' and r['metadata']['kind']=='real'];assert len(selected)==48
    records=[];points=[];oldlabel_delta=[]
    for i,r in selected:
        rec=r['record'];m=r['metadata'];camera=chr(65+m['view']);key=rec['session'],int(rec['frame']),camera;assert key not in trainkeys
        j=lookup.get(key);meta=json.loads((ROOT/rec['scene']/'scene_meta.json').read_text());view=next(v for v in meta['views'] if v['view_id']==m['view']);entry=entries[j] if j is not None else dict(session=key[0],frame=key[1],camera=camera,source=view['source_image'],source_sha256=view['source_sha256'],derived=None)
        assert view['source_image']==entry['source'];assert fingerprint(ROOT/entry['source'])==view['source_sha256']==entry['source_sha256']
        A=np.array(meta['image_affine']);assert np.array_equal(A,np.array(dm['affine_source_to_input']))
        derived=ROOT/rec['scene']/'blended_images'/f'{m["view"]:08d}.png';im1=np.asarray(Image.open(derived).convert('RGB'))
        im2=np.asarray(Image.open(ROOT/entry['derived']).convert('RGB')) if entry['derived'] else None
        assert im1.shape==(768,704,3)
        image_difference=np.abs(im1.astype(float)-im2.astype(float)) if im2 is not None else None
        records.append(dict(index=i,entry=entry,metadata=m,current_image=derived.relative_to(ROOT).as_posix(),current_image_sha256=fingerprint(derived),source_hash=view['source_sha256'],input_pixels_equal=bool(np.array_equal(im1,im2)) if im2 is not None else None,input_uint8_mean_difference=float(image_difference.mean()) if im2 is not None else None,input_uint8_max_difference=float(image_difference.max()) if im2 is not None else None,affine_equal=True,teacher_was_train=False,old_cache_index=j))
        if j is None and fresh is None:continue
        teacher=torch.tensor(fresh[i][ci,:2]).float() if fresh is not None else torch.tensor(cached['uv'][j,ci]*np.diag(A)[:2]+A[:2,2]).float();original=r['baseline']['xy'][:,KP]*r['pixels'];truth=r['gt']*r['pixels']
        oldgt=torch.tensor(cached['gt'][j,ci]*np.diag(A)[:2]+A[:2,2]).float() if j is not None else torch.full((5,2),float('nan'))
        for q,k in r['valid'].nonzero().tolist():
            oe=float((oldgt[k]-truth[q,k]).norm()) if torch.isfinite(oldgt[k]).all() else None
            if oe is not None:oldlabel_delta.append(oe)
            points.append(dict(index=i,query=q,k=k,point=KP[k]+1,session=key[0],frame=key[1],camera=camera,bottom=m['physical_bottom'],foundation_error=float((original[q,k]-truth[q,k]).norm()),teacher_error=float((teacher[k]-truth[q,k]).norm()),teacher_xy=teacher[k].tolist(),gt_xy=truth[q,k].tolist(),teacher_old_GT_delta=oe))
    result={}
    for name,filtered in [('all',points),('bottom_feet',[p for p in points if p['bottom'] and p['point']!=1]),('side_feet',[p for p in points if not p['bottom'] and p['point']!=1])]:
        result[name]={a:stats([p[a+'_error'] for p in filtered]) for a in ['foundation','teacher']}
    report=dict(summary=result,current_images=48,old_cached_images=sum(e['old_cache_index'] is not None for e in records),current_known_points=len(points),old_derived_pixel_equal=sum(e['input_pixels_equal'] is True for e in records),current_RGB_teacher_replayed=args.fresh,no_teacher_training_overlap=True,old_to_current_GT_delta=stats(oldlabel_delta),teacher_checkpoint=json.loads((OLD/'quadruped_epoch110_review_v1/summary.json').read_text())['checkpoint'],prediction_cache_sha256=fingerprint(OLD/'quadruped_epoch110_review_v1/predictions_source_pixels.npz'),limitations=['Same corrected current known labels and originalfoundation qualifiedquery subset; not AP or detector comparison.','Teacher is single-mouse seven-point Quadruped-HRNet150epochbest110, not a multi-instance/foundation replacement.','Different pretraining,trainingduration,data,head andoldlabelversion; not architecture-only orcompute-matched.','Teacher checkpoint selected onreuseddevelopment. No pristineexternaltest, no newpseudoGT, nofusion orlabelchanges.','Originalteacher usesLanczos whilefoundation usesBicubic withsameaffine; fresh flag indicateswhether frozenHRNet replayedcurrentBicubicRGB.','Cache-only reference excludes currentimagesmissingold281cache; freshinference evaluatesall48.'])
    write_json(OUT/'results.json',report);write_json(OUT/'points.json',points);write_json(OUT/'image_parity.json',records)
    chosen=[]
    for label,predicate in [('bottom',lambda p:p['bottom'] and p['point']!=1),('side',lambda p:not p['bottom'] and p['point']!=1)]:
        pp=[p for p in points if predicate(p)]
        for selection,p in [('largest_foundation_error',max(pp,key=lambda p:p['foundation_error'])),('largest_teacher_error',max(pp,key=lambda p:p['teacher_error']))]:
            r=rows[p['index']];item=load_item(r['record']);vi=r['record']['views'].index(r['metadata']['view']);rgb=item['batch']['images'][0,vi];image=Image.fromarray((rgb.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8));canvas=Image.new('RGB',(1440,875),'white');d=ImageDraw.Draw(canvas)
            for col,name in enumerate(['foundation','teacher']):
                im=image.copy();di=ImageDraw.Draw(im);gx,gy=p['gt_xy'];x,y=(r['baseline']['xy'][p['query'],p['point']-1]*r['pixels']).tolist() if name=='foundation' else p['teacher_xy'];di.line((x,y,gx,gy),fill='yellow',width=2);di.ellipse((x-8,y-8,x+8,y+8),outline='red',width=4);di.line((gx-10,gy,gx+10,gy),fill='#00ff66',width=4);di.line((gx,gy-10,gx,gy+10),fill='#00ff66',width=4);canvas.paste(im,(col*720,100));d.text((col*720+10,10),f'{name} error {p[name+"_error"]:.2f}px',font=font(25),fill='black');d.text((col*720+10,50),f'{p["session"]} f{p["frame"]} cam{p["camera"]} point{p["point"]}',font=font(20),fill='black')
            file=f'{label}_{selection}.jpg';canvas.save(OUT/file,quality=95);chosen.append(dict(file=file,**p))
    write_json(OUT/'visual_examples.json',chosen)
    replay_note='已在全部48张当前基础模型使用的Bicubic图像上重新执行冻结HRNet，使用原生归一化、单图推理；不修改权重。' if args.fresh else '本页暂用旧Lanczos输入的缓存交集预测，部分当前图原来被排除，没有旧缓存，尚未计入本页指标；下一步在全部当前Bicubic输入上重放。'
    page=['<!doctype html><meta charset="utf-8"><title>相同当前标注下的定位教师对照</title><style>body{font:17px system-ui;max-width:1450px;margin:30px auto}img{max-width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>同源图、同当前标注、同评价点：专门定位教师对照</h1><p>48张当前开发图，核验原图SHA和letterbox仿射相同；原流程教师用Lanczos、基础模型用Bicubic，因此旧输入像素并非完全一致。'+replay_note+'</p><p>每张均未进入教师训练。教师是已有四足版SuperAnimal-HRNet，训练150轮后旧开发集选best110；不是TopView，也不是本轮新训练。这是不同预训练、架构、训练量和数据配方的系统级诊断，不是公平架构消融。仅单鼠，不证明双鼠可用；不向GT写入预测。双方评价当前模型合格匹配的同一鼻尖/四脚点；不是检测AP，开发集已反复使用。</p><table><tr><th>范围</th><th>方法</th><th>点数</th><th>输入像素均值</th><th>P95</th><th>PCK32</th></tr>']
    for label,metrics in result.items():
        for name,z in metrics.items():page.append(f'<tr><td>{label}</td><td>{name}</td><td>{z["n"]}</td><td>{z["mean"]:.3f}</td><td>{z["p95"]:.3f}</td><td>{z["pck32"]:.3f}</td></tr>')
    page.append('</table><h2>极值案例，不代表频率</h2><p>绿十字：当前评价GT；红圈：预测。旧教师的背部6/7点不在本表，不把其解释为准确内部骨骼位置。</p>')
    for e in chosen:page.append(f'<h3>{e["file"]}</h3><img src="{e["file"]}">')
    page.append('<p><a href="results.json">结果与限制</a> | <a href="points.json">逐点数据</a> | <a href="image_parity.json">图像与划分核验</a></p>');(OUT/'index.html').write_text('\n'.join(page),encoding='utf-8')
    for path in OUT.iterdir():
        if path.is_file():shutil.copy2(path,DEST/path.name)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
