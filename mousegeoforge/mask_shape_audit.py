"""GT-free binary-mask diagnostics. No denoising or deployment decision."""
import numpy as np
from scipy import ndimage


def mask_shape_statistics(mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or min(mask.shape)<=0:raise ValueError('Expected nonempty H,W mask')
    h,w=mask.shape;labels,n=ndimage.label(mask,structure=np.ones((3,3),dtype=bool))
    areas=np.bincount(labels.ravel())[1:];area=int(mask.sum())
    def box(binary):
        yy,xx=np.nonzero(binary)
        if not len(xx):return [0.,0.,0.,0.]
        lo=np.array([xx.min()/w,yy.min()/h]);hi=np.array([(xx.max()+1)/w,(yy.max()+1)/h])
        return np.r_[(lo+hi)/2,hi-lo].tolist()
    full=box(mask);largest=labels==(int(areas.argmax())+1) if area else np.zeros_like(mask)
    big=box(largest)
    return dict(area=area,components=int(n),largest_component_area=int(areas.max()) if area else 0,
        largest_component_fraction=float(areas.max()/area) if area else 0.,
        fill_fraction=float(area/max(1,full[2]*w*full[3]*h)),
        touches_boundary=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
        full_box_cxcywh=full,largest_component_box_cxcywh=big,
        largest_to_full_box_area=float(big[2]*big[3]/max(1e-12,full[2]*full[3])))
