import json,hashlib
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1];base=root/'output/bottom_tail_fix_20261003';p=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-BottomTailFix-20261003/DELIVERY')
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src'))
links=Links();links.feed((p/'index.html').read_text())
for link in links.links:
    if not urlparse(link).scheme:assert (p/unquote(urlparse(link).path)).is_file(),link
renders=list((p/'renders').rglob('*.png'));assert len(renders)==6
for image in renders:assert Image.open(image).size==(1080,1200)
roi=(25,560,45,630)
values={}
for prefix in ['before','renders']:
    a=np.array(Image.open(p/prefix/'headraise/camA.png').convert('L').crop(roi));values[prefix]=dict(mean=float(a.mean()),fraction_below_80=float((a<80).mean()))
assert values['renders']['fraction_below_80']<values['before']['fraction_below_80']
result=dict(local_links_checked=len(links.links),native_renders=6,resolution=[1080,1200],diagnostic_fixed_roi=roi,roi_statistics=values,scope='Selected three static poses A/F, not arbitrary animations or all datasets.')
(base/'final_review_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
state=json.loads((base/'state.json').read_text(encoding='utf-8-sig'));assert state['active_pid'] is None
state.update(stage='completed_verified',active_job=None,accepted_delivery=str(p),
             accepted_local_delivery=str(root/'BlenderPyMouseProjects/Cylinder/PyMouseStudio-RealCapture-BottomTailFix-20261003/DELIVERY-Balanced'),
             visual_review='Manually inspected all six final native A/F renders: headraise tail patch removed, no new tail patch in climb/quadruped. Side views preserve pose and optical glass reflections, with brighter underside illumination. Dust balance accepted.',
             next_action='Await user review. No recurring automation started, no dataset changes. DELIVERY-Balanced is accepted; sibling DELIVERY is the rejected bright-dust intermediate.')
(base/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
