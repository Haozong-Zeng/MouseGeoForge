"""CPU unit/stress tests on analytic and actual newly rendered surface curves."""
import importlib.util,json,sys,datetime as dt
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import BSpline
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mousegeoforge.surface_curve_supervision import curve_loss,point_loss
OUT=ROOT/'output/surface_curve_cycle_20261003/stress_v2'
DS=ROOT/'dataset/BlenderPyMouse/PyMouseStudio-RealCapture-SurfaceCurve-Multiview'
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('surface_tests',ROOT/'tests/test_surface_curve_supervision.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tests=[]
    for name in sorted(n for n in vars(module) if n.startswith('test_')):
        getattr(module,name)();tests.append(name);print('PASS',name,flush=True)
    rows=[];fits=[];t=np.linspace(0,1,65);basis4=np.stack([(1-t)**3,3*(1-t)**2*t,3*(1-t)*t*t,t**3],-1)
    knots=np.r_[np.zeros(4),1/3,2/3,np.ones(4)];basis6=BSpline.design_matrix(t,knots,3).toarray()
    for f in sorted(DS.glob('frame_*')):
        with np.load(f/'surface_curve.npz') as a:curves=a['uv'][:,0];visibility=a['visible'][:,0]
        for vi,pixels in enumerate(curves):
            p=pixels*np.array([691/1080,768/1200])+[6,0];gt=torch.tensor(p,dtype=torch.float32);valid=torch.ones(65,dtype=torch.bool)
            remap=np.linspace(0,64,65)**1.15/(64**.15);rp=np.stack([np.interp(remap,np.arange(65),p[:,i]) for i in range(2)],-1)
            derivative=np.gradient(p,axis=0);normal=np.stack([-derivative[:,1],derivative[:,0]],-1);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9)
            candidates=dict(exact=p,reparameterized=rp,normal8px=p+8*normal,reverse=p[::-1].copy(),collapsed=np.repeat(p[32:33],65,axis=0))
            losses={};gradients={}
            for tag,c in candidates.items():
                pred=torch.tensor(c,dtype=torch.float32,requires_grad=True);loss=curve_loss(pred,gt,valid,full_curve=True);loss.backward()
                assert torch.isfinite(loss) and torch.isfinite(pred.grad).all();losses[tag]=float(loss.detach());gradients[tag]=float(pred.grad.norm())
            assert losses['exact']<1e-8 and losses['reparameterized']<losses['normal8px']*.1
            assert min(losses['reverse'],losses['collapsed'])>losses['reparameterized']+1e-5
            rows.append(dict(frame=f.name,camera=chr(65+vi),losses=losses,gradient_norm=gradients,visible_samples=int(visibility[vi].sum())))
            # Measure representation floor without training: cubic4 vs cubic B-spline6.
            fit={};curves_fit={}
            for name,basis in [('bezier4',basis4),('bspline6',basis6)]:
                control=np.linalg.lstsq(basis,p,rcond=None)[0];curve=basis@control;err=np.linalg.norm(curve-p,axis=-1);curves_fit[name]=curve
                fit[name]=dict(mean_px=float(err.mean()),p95_px=float(np.quantile(err,.95)),max_px=float(err.max()))
            fits.append(dict(frame=f.name,camera=chr(65+vi),fits=fit))
            if vi in [2,3,5]:
                im=Image.open(f/'blended_images'/f'{vi:08d}.jpg').convert('RGB').resize((691,768));canvas=Image.new('RGB',(704,808),'#172335');canvas.paste(im,(6,40));draw=ImageDraw.Draw(canvas)
                for name,c,color in [('truth',p,'#ffad22'),('bezier4',curves_fit['bezier4'],'#12ffff'),('bspline6',curves_fit['bspline6'],'#ff30e0')]:draw.line([tuple(x+[0,40]) for x in c],fill=color,width=2)
                draw.text((6,6),f'{f.name} cam{chr(65+vi)} orange GT / cyan cubic4 / pink spline6',fill='white');canvas.save(OUT/f'{f.name}_cam{chr(65+vi)}_fits.png')
    report=dict(passed=True,unit_tests=tests,stress_views=rows,representation_fits=fits,
      scope='Loss/gradient and representation tests only, not image-model training or evidence of real3D accuracy',
      supervision='Complete synthetic material curves include known occluded coordinates; visibility is separately recorded. Real missing coordinates do not become full curves.',
      input_wh=[704,768],labels_modified=False)
    save(OUT/'report.json',report)
    print(json.dumps(dict(passed=True,unit_tests=len(tests),rendered_curve_views=len(rows),max_bezier4_p95=max(r['fits']['bezier4']['p95_px'] for r in fits),max_bspline6_p95=max(r['fits']['bspline6']['p95_px'] for r in fits)),indent=2))
    state=json.loads((OUT.parent/'state.json').read_text());state.update(stage='surface_curve_stress_passed_training_protocol_next',updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),next='Read stress_v2/report.json; choose curve representation based on representation floor. Prepare frozen source-separated768 mixed synthetic/real training control with actual Core23 loss replacement, no protected-label or heldout changes.');save(OUT.parent/'state.json',state)
if __name__=='__main__':main()
