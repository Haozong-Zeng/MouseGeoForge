"""CPU review of fixed first and minimum-delta examples; no model/selection changes."""
from pathlib import Path
import datetime as dt,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
import cv2,numpy as np
from PIL import Image,ImageDraw
OUT=ROOT/'output/calibrated_mouse_resume8h_20261025';BASE=OUT/'pixel_center_pair_v2';DEST=OUT/'pixel_center_visual_review_v3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not DEST.exists();DEST.mkdir();paths={policy:OUT/f'pixel_center_{policy}_native_probability_audit_v2.json' for policy in ['legacy','center_exact']};audits={p:json.loads(f.read_text()) for p,f in paths.items()};a,b=[{x['id']:x for x in audits[p]['rows']} for p in ['legacy','center_exact']];assert set(a)==set(b) and len(a)==879
 manifest=ROOT/'output/calibrated_mouse_10h_20261025/semseg_items.json';items=json.loads(manifest.read_text())['items'];items={x['id']:x for x in items if x['split']=='dev'};pins={str(p.relative_to(ROOT)):sha(p) for p in [manifest,*paths.values()]};selections=[]
 for domain,bottom in [('real',True),('real',False),('synthetic',True),('synthetic',False)]:
  ids=sorted(k for k,v in a.items() if (v['domain'],v['bottom'])==(domain,bottom));delta=lambda k:b[k]['native_original']['iou']-a[k]['native_original']['iou']
  for choice,id in [('first_id',ids[0]),('minimum_delta',min(ids,key=lambda k:(delta(k),k)))]:selections.append(dict(choice=choice,id=id,domain=domain,bottom=bottom,legacy_iou=a[id]['native_original']['iou'],center_iou=b[id]['native_original']['iou'],delta_iou=delta(id)))
 canvas=Image.new('RGB',(1400,len(selections)*275),'white');draw=ImageDraw.Draw(canvas)
 for index,row in enumerate(selections):
  item=items[row['id']];ip=ROOT/item['image'];lp=ROOT/item['label'];assert sha(ip)==item['image_sha256'] and sha(lp)==item['label_sha256'];pins[item['image']]=sha(ip);pins[item['label']]=sha(lp);rgb=np.asarray(Image.open(ip).convert('RGB'))
  if lp.suffix=='.npz':
   with np.load(lp) as z:gt=z['body'].astype(np.uint8)
  else:
   raw=np.asarray(Image.open(lp).convert('L'));gt=raw if item.get('label_encoding')=='class_index_0_1_255' else (raw>0).astype(np.uint8)
  masks={}
  for policy in ['legacy','center_exact']:
   path=BASE/(policy+'_dev')/'masks_ema'/(row['id']+'.png');assert sha(path)==a[row['id']]['prediction_sha256'] if policy=='legacy' else sha(path)==b[row['id']]['prediction_sha256'];pins[str(path.relative_to(ROOT))]=sha(path);masks[policy]=cv2.imread(str(path),0)>0
  union=(gt==1)|masks['legacy']|masks['center_exact'];y,x=np.where(union);x0,x1=max(0,int(x.min())-12),min(rgb.shape[1],int(x.max())+13);y0,y1=max(0,int(y.min())-12),min(rgb.shape[0],int(y.max())+13)
  truth=rgb.copy();truth[gt==1]=(.4*truth[gt==1]+.6*np.array([0,230,0])).astype(np.uint8);images=[rgb,truth[y0:y1,x0:x1]]
  for policy in ['legacy','center_exact']:
   mask=masks[policy];im=(rgb*.35).astype(np.uint8);im[mask&(gt==1)]=[40,210,40];im[mask&(gt==0)]=[255,110,0];im[(~mask)&(gt==1)]=[40,100,255];images.append(im[y0:y1,x0:x1])
  change=(rgb*.35).astype(np.uint8);change[masks['center_exact']&~masks['legacy']]=[0,220,80];change[masks['legacy']&~masks['center_exact']]=[240,40,80];images.append(change[y0:y1,x0:x1])
  label=f'{row["domain"]} bottom={row["bottom"]} {row["choice"]} | IoU {row["legacy_iou"]:.4f} -> {row["center_iou"]:.4f} delta{row["delta_iou"]:+.4f}';draw.text((4,index*275+2),label,fill='black')
  for col,(im,title) in enumerate(zip(images,['native RGB','GT overlay (display)','legacy TPgreen FPorange FNblue','center TPgreen FPorange FNblue','added green / removed red'])):
   image=Image.fromarray(im);image.thumbnail((274,234));canvas.paste(image,(col*280+(280-image.width)//2,index*275+36+(234-image.height)//2));draw.text((col*280+3,index*275+18),title,fill='black')
 canvas.save(DEST/'comparison.png');result=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_sha256=sha(__file__),input_sha256=pins,image_sha256=sha(DEST/'comparison.png'),selections=selections,GT_crops='DISPLAY ONLY; no inference, mask, metric, threshold or checkpoint changes',limits='Descriptive first and minimum-delta examples across each domain/view stratum. Diagnostic ranking only, not a fresh validation or selection set.')
 (DEST/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(selections))
if __name__=='__main__':main()
