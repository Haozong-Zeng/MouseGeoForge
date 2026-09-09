"""Isolated one-image smoke test of the original style-transfer recipe."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'img2img-turbo/scripts'))
import transfer_blenderpymouse_pseudosynthetic as transfer
name='E_L2_2024-11-18_1_Trial_1_PseudoSynthetic'
root=BASE/'dataset/BlenderPyMouse'
relative='frame_000001/blended_images/00000000.jpg'
source=root/name/relative
expected=root/(name+'Transfered')/relative
out=BASE/'output/temporal_completion/style_reference'
out.mkdir(parents=True,exist_ok=True)
model=transfer.CycleGAN_Turbo(pretrained_path=str(transfer.DEFAULT_CHECKPOINT))
model.eval().half()
embedding=transfer.make_caption_embedding(model,transfer.DEFAULT_PROMPT,torch.float16)
transform=transfer.build_transform('resize_pad_576x384')
original=Image.open(source).convert('RGB')
tensor=transfer.transforms.ToTensor()(transform(original))
tensor=transfer.transforms.Normalize([0.5],[0.5])(tensor).unsqueeze(0).cuda().half()
with torch.inference_mode():
    result=model(tensor,direction='a2b',caption_emb=embedding)[0]
result=transfer.transforms.ToPILImage()(result.float().cpu()*0.5+0.5)
result=result.resize(original.size,Image.Resampling.LANCZOS)
result.save(out/'reference.jpg',quality=95,subsampling=0)
a=np.asarray(Image.open(expected)).astype(float)
b=np.asarray(Image.open(out/'reference.jpg')).astype(float)
report={'checkpoint':str(transfer.DEFAULT_CHECKPOINT),'image_size':list(original.size),
    'rgb_mae_255':float(np.abs(a-b).mean()),'source':str(source),'old_style':str(expected),
    'test_style':str(out/'reference.jpg'),'inference_succeeded':True}
(out/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
