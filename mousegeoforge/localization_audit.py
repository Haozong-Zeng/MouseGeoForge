"""Read-only diagnostics for coarse coordinates and local soft-argmax."""
import numpy as np


def refinement_statistics(coarse, refined, target, logits, image_hw, box_wh,
                          radius=.1, scales=(1.,4.,8.,16.)):
    coarse,refined,target,logits=map(lambda x:np.asarray(x,dtype=np.float64),(coarse,refined,target,logits))
    if coarse.shape!=(2,) or refined.shape!=(2,) or target.shape!=(2,):
        raise ValueError('Expected individual xy pairs')
    if logits.ndim!=2 or logits.shape[0]!=logits.shape[1] or logits.shape[0]<2 or radius<=0:
        raise ValueError('Invalid local grid')
    if not all(np.isfinite(x).all() for x in (coarse,refined,target,logits)):
        raise ValueError('Nonfinite coordinates/logits')
    pixel=np.asarray(image_hw[::-1]);diag=max(np.linalg.norm(np.asarray(box_wh)*pixel),1.)
    axis=np.linspace(-radius,radius,logits.shape[0]);yy,xx=np.meshgrid(axis,axis,indexing='ij')
    offsets=np.stack((xx,yy),axis=-1)
    def decode(scale):
        z=logits*scale;prob=np.exp(z-z.max());prob/=prob.sum()
        xy=np.clip(coarse+(prob[...,None]*offsets).sum((0,1)),0,1)
        return xy,prob
    xy,prob=decode(1.)
    if np.max(np.abs(xy-refined))>2e-6: raise ValueError('Soft-argmax reconstruction mismatch')
    delta=target-coarse
    inside=bool((np.abs(delta)<=radius).all())
    nearest=np.clip(target,np.maximum(coarse-radius,0),np.minimum(coarse+radius,1))
    error=lambda xy:float(np.linalg.norm((xy-target)*pixel))
    coarse_px,refined_px=error(coarse),error(refined)
    result=dict(coarse_px=coarse_px,refined_px=refined_px,coarse_nme=coarse_px/diag,
        refined_nme=refined_px/diag,refinement_delta_px=float(np.linalg.norm((refined-coarse)*pixel)),
        improvement_px=coarse_px-refined_px,inside_window=inside,
        window_oracle_min_px=error(nearest),entropy_normalized=float(-(prob*np.log(prob)).sum()/np.log(prob.size)),
        max_probability=float(prob.max()),logit_range=float(np.ptp(logits)),
        clipped_window=bool(((coarse-radius)<0).any() or ((coarse+radius)>1).any()))
    for scale in scales:
        point,_=decode(scale);result[f'scale_{scale:g}_nme']=error(point)/diag
    return result


def summarize(rows):
    if not rows:return dict(points=0)
    keys=('coarse_px','refined_px','coarse_nme','refined_nme','refinement_delta_px','improvement_px',
          'inside_window','window_oracle_min_px','entropy_normalized','max_probability','logit_range',
          'clipped_window','scale_1_nme','scale_4_nme','scale_8_nme','scale_16_nme')
    return dict(points=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in keys},
        refinement_improved_fraction=float(np.mean([r['improvement_px']>0 for r in rows])),
        refinement_delta_p95_px=float(np.percentile([r['refinement_delta_px'] for r in rows],95)))
