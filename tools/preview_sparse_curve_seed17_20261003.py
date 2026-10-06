"""CPU-only fixed completed-seed preview while remaining seeds run."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from probe_diverse_curve_features_20261003 import FeatureProbe
from review_sparse_real_curve_adaptation_20261003 import crossings
CYCLE=ROOT/'output/surface_curve_cycle_20261003'
BASE=CYCLE/'sparse_real_curve_adaptation_v1'
OUT=BASE/'preview_seed17'


def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(2)
    saved=torch.load(CYCLE/'real_curve_adaptation_inputs_v1/frozen_inputs.pt',map_location='cpu',weights_only=False);data=saved['tensors'];meta=saved['metadata'];pred={}
    for arm,path in [('initial',CYCLE/'curve_data_scaling_v1/expanded_seed17.pt'),('replay',BASE/'replay_seed17_latest.pt'),('weak_real',BASE/'weak_real_seed17_latest.pt')]:
        checkpoint=torch.load(path,map_location='cpu',weights_only=False);assert checkpoint['step']==2400
        model=FeatureProbe(False).eval();model.load_state_dict(checkpoint['state'])
        with torch.no_grad():pred[arm]=model(data['pose'],data['boxes'],data['base_logits']).numpy()
    selected={('AZL3',0,5),('FR2',0,5),('FR2',584,1)};visuals=[]
    for i,m in enumerate(meta):
        if (m['session'],m['frame'],m['view']) not in selected:continue
        with Image.open(ROOT/m['scene']/'blended_images'/f'{m["view"]:08d}.png') as source:im=source.convert('RGB').resize((352,384))
        canvas=Image.new('RGB',(1056,412),(20,20,26));draw=ImageDraw.Draw(canvas)
        for col,(arm,curves) in enumerate(pred.items()):
            canvas.paste(im,(352*col,28));draw.text((352*col+4,5),arm,fill='white');p=curves[i]*[352,384]+[352*col,28];draw.line([tuple(x) for x in p],fill='#39dfff',width=2)
            for k in range(2):
                if data['observed_valid'][i,k]:
                    x,y=data['observed_xy'][i,k].numpy()*[352,384]+[352*col,28];draw.ellipse((x-3,y-3,x+3,y+3),fill='#ffe454');draw.text((x+4,y-8),str(k+6),fill='#ffe454')
        name=f'{m["session"]}_{m["frame"]:06d}_{m["view"]}.jpg';canvas.save(OUT/name,quality=94);visuals.append(dict(file=name,role=m['role'],session=m['session'],frame=m['frame'],view=m['view']))
    shapes=[]
    for arm,curves in pred.items():
        for role in ['train','validation']:
            ids=[i for i,m in enumerate(meta) if m['role']==role];n=[crossings(curves[i]) for i in ids]
            shapes.append(dict(arm=arm,role=role,n=len(ids),self_crossing_curves=sum(v>0 for v in n)))
    (OUT/'report.json').write_text(json.dumps(dict(shapes=shapes,visuals=visuals,seed17_only=True,not_final_multiseed_conclusion=True),indent=2));print(json.dumps(shapes,indent=2))


if __name__=='__main__':main()
