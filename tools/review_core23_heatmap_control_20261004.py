"""Fixed-frame paired6400-step control versus reliable-point heatmap loss."""
import html,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from run_depth_pose_ablation import write_json,fingerprint
CYCLE=ROOT/'output/core23_spatial_cycle_20261004';OUT=CYCLE/'review_reliable_heatmap_v1'


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    dirs=[CYCLE/'readout_expanded_all_soft_box_extended4800_v1',CYCLE/'readout_post4800_control_v1',CYCLE/'readout_post4800_reliable_heatmap_v1']
    names=['4800coordinate','6400coordinate','6400reliable_heatmap'];pred=[]
    for p in dirs:
        with np.load(p/'predictions.npz') as a:pred.append(a['spatial_seed17'].copy())
    controls=json.loads((dirs[1]/'results.json').read_text())['results'];heat=json.loads((dirs[2]/'results.json').read_text())['results']
    checks=[]
    for seed in [17,29,43]:
        assert json.loads((dirs[1]/f'schedule_seed{seed}.json').read_text())==json.loads((dirs[2]/f'schedule_seed{seed}.json').read_text())
        for mode in ['global','spatial']:
            arm=f'{mode}_seed{seed}';a=json.loads((dirs[1]/f'{arm}_resume.json').read_text());b=json.loads((dirs[2]/f'{arm}_resume.json').read_text());assert a==b
            c=next(r for r in controls if r['arm']==arm);h=next(r for r in heat if r['arm']==arm)
            checks.append(dict(arm=arm,resume_checkpoint_sha256=a['checkpoint_sha256'],real_reliable_delta_heatmap_minus_control=h['metrics']['real_validation']['reliable']['mean']-c['metrics']['real_validation']['reliable']['mean'],dual_delta=h['metrics']['synthetic_validation']['multi_instance']['mean']-c['metrics']['synthetic_validation']['multi_instance']['mean']))
    rows=[]
    for r in json.loads((CYCLE/'features_v1/manifest.json').read_text())['records']:
        rows.extend(torch.load(ROOT/r['path'],map_location='cpu',weights_only=False)['observations'])
    page=['<!doctype html><meta charset="utf-8"><title>Reliable heatmap control</title><style>body{background:#20232b;color:white;font:16px sans-serif}img{max-width:100%}</style><h1>Reliable-point heatmap control: fixed prior cases</h1><p>Matched checkpoint/optimizer/minibatch/LR, three seeds; yellow=existing observation, cyan=prediction.6400coordinate versus6400heatmap differs only by selected nose/feet KL loss. Back06/07 receive no heatmap supervision. No GT-selected inference crop. Query association is assisted, not detector AP.</p>'];visuals=[]
    for i in [61,74,188,205,303,313]:
        o=rows[i];m=o['metadata'];r=m['record'];cache=torch.load(ROOT/f'output/surface_curve_cycle_20261003/training_v1/pilot/cache/{r["key"]}.pt',map_location='cpu',weights_only=False)
        pixels=(cache['batch']['images'][0,m['view_index']].permute(1,2,0).numpy()*255).clip(0,255).astype('uint8');h,w=pixels.shape[:2];pw=420;ph=round(h*pw/w);im=Image.fromarray(pixels).resize((pw,ph));canvas=Image.new('RGB',(pw*3,ph+40),(22,24,30));draw=ImageDraw.Draw(canvas)
        for col,name in enumerate(names):
            canvas.paste(im,(col*pw,40));draw.text((col*pw+5,8),name,fill='white')
            for k in [0,14,15,16,17]:
                if not o['valid'][k]:continue
                x,y=pred[col][i,k]*[pw,ph]+[col*pw,40];gx,gy=o['gt_xy'][k].numpy()*[pw,ph]+[col*pw,40]
                draw.line((gx,gy,x,y),fill='#888888');draw.ellipse((x-3,y-3,x+3,y+3),outline='#44ddff',width=2);draw.text((x+4,y+1),str(k+1),fill='#44ddff');draw.line((gx-4,gy,gx+4,gy),fill='#ffe45c',width=2);draw.line((gx,gy-4,gx,gy+4),fill='#ffe45c',width=2)
        name=f'{i:03d}.jpg';canvas.save(OUT/name,quality=95);visuals.append(dict(observation=i,file=name,metadata=m));page.append('<h2>'+html.escape(str(r))+'</h2><img src="'+name+'">')
    write_json(OUT/'paired_checks.json',dict(checks=checks,visuals=visuals,decision='Reject selected heatmap loss for current readout: all3spatial seeds worsen real reliable and syntheticdual. No sigma/weight validation gridsearch. Keep coordinate-only6400 as diagnostic branch, not production.'))
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-Core23-Spatial-20261004')/OUT.name;assert not dest.exists();shutil.copytree(OUT,dest)


if __name__=='__main__':main()
