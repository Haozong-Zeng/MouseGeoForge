"""Compact scientific contact sheets for every CAL bottom prediction, both arms."""
import argparse, json, shutil
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='grouping_pilot_v1');parser.add_argument('--center',action='store_true');parser.add_argument('--scales',action='store_true');parser.add_argument('--arms',nargs='+');parser.add_argument('--fixed',action='store_true');parser.add_argument('--development',action='store_true');args=parser.parse_args()
    out=ROOT/'output/local_grouping_cycle_20261005'/args.run
    assert out.parent==ROOT/'output/local_grouping_cycle_20261005'
    inputs='mixed_local_development_inputs_v1' if args.development else 'mixed_local_fulltrain_rgb_inputs_v1'
    rows=torch.load(ROOT/'output/singleview_fusion_cycle_20261005'/inputs/'evaluation_only.pt',weights_only=False,map_location='cpu')
    selected=[r['index'] for r in rows if r['metadata']['stage']==('validation' if args.development else 'calibration') and r['metadata']['kind']=='real' and r['metadata']['physical_bottom']]
    assert len(selected)==(8 if args.development else 16)
    if args.fixed:
        selected=json.loads((out/'protocol.json').read_text())['visual_indices'] if args.development else [next(r['index'] for r in rows if r['metadata']['stage']==stage and r['metadata']['kind']==kind) for stage in ['fit','calibration'] for kind in ['real','single','dual']]
    dest=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005')/args.run
    dest.mkdir(parents=True,exist_ok=True)
    arms=args.arms or (['native','half','quarter'] if args.scales else (['center'] if args.center else ['independent','within_instance']))
    for arm in arms:
        for start in range(0,len(selected),4):
            canvas=Image.new('RGB',(1600,980),'white');draw=ImageDraw.Draw(canvas)
            for slot,i in enumerate(selected[start:start+4]):
                im=Image.open(out/'examples'/f'{arm}_{i}.jpg');im.thumbnail((790,465))
                y,x=divmod(slot,2);canvas.paste(im,(x*800,y*490))
            category='fixed' if args.fixed else 'bottom'
            name=f'{arm}_{category}_contact_{start//4+1}.jpg';canvas.save(out/name,quality=94);shutil.copy2(out/name,dest/name)
    print(json.dumps(dict(indices=selected,contacts=((len(selected)+3)//4)*len(arms),run=args.run)))

if __name__=='__main__':main()
