"""Human surface-line review contract, separate from Core23 and pseudo-GT."""
import numpy as np

LINE_LABELS={'dorsal_surface','dorsal_surface_uncertain','silhouette_only'}
ENDPOINT_LABELS={'neck_surface_endpoint','tail_root_surface_endpoint'}


def parse_annotation(a, *, width, height, image_name):
    errors=[];lines={k:[] for k in LINE_LABELS};endpoints={}
    if not isinstance(a,dict):return dict(reviewed=False,valid=False,errors=['not_annotation_object'],lines=lines,endpoints=endpoints,no_visible=False)
    flags=a.get('flags',{})
    if not isinstance(flags,dict):flags={};errors.append('flags_not_object')
    for name in ('reviewed','no_visible_dorsal_surface','only_silhouette_identifiable'):
        if name in flags and not isinstance(flags[name],bool):errors.append(name+':not_boolean')
    shapes=a.get('shapes',[])
    if not isinstance(shapes,list):shapes=[];errors.append('shapes_not_list')
    if a.get('imageWidth')!=width or a.get('imageHeight')!=height:errors.append('image_size_mismatch')
    if str(a.get('imagePath','')).replace('\\','/').split('/')[-1]!=image_name:errors.append('image_name_mismatch')
    for index,s in enumerate(shapes):
        if not isinstance(s,dict):errors.append(f'shape{index}:not_object');continue
        label=s.get('label');kind=s.get('shape_type');prefix=f'shape{index}:'
        if s.get('group_id') not in (None,0):errors.append(prefix+'unexpected_instance');continue
        if not isinstance(label,str) or label not in LINE_LABELS|ENDPOINT_LABELS:errors.append(prefix+'unknown_label');continue
        try:p=np.asarray(s.get('points',[]),dtype=float)
        except (TypeError,ValueError):errors.append(prefix+'bad_coordinates');continue
        if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():errors.append(prefix+'nonfinite_or_bad_shape');continue
        if ((p<0).any() or (p[:,0]>=width).any() or (p[:,1]>=height).any()):errors.append(prefix+'out_of_bounds');continue
        if label in LINE_LABELS:
            if kind!='linestrip' or len(p)<2 or np.linalg.norm(np.diff(p,axis=0),axis=1).sum()<1:
                errors.append(prefix+'not_open_nonzero_polyline');continue
            if np.linalg.norm(p[0]-p[-1])<1e-6:errors.append(prefix+'closed_curve');continue
            lines[label].append(p)
        else:
            if kind!='point' or len(p)!=1:errors.append(prefix+'not_point');continue
            if label in endpoints:errors.append(prefix+'duplicate_endpoint');continue
            # Explicit uncertain endpoints are never geometric anchors.
            shape_flags=s.get('flags',{})
            if not isinstance(shape_flags,dict):errors.append(prefix+'flags_not_object');continue
            if not shape_flags.get('uncertain',False):endpoints[label]=p[0]
    reviewed=flags.get('reviewed') is True
    if reviewed:
        if not any(lines.values()) and not flags.get('no_visible_dorsal_surface',False):errors.append('empty_without_explicit_visibility_status')
        if lines['dorsal_surface'] and flags.get('no_visible_dorsal_surface',False):errors.append('visible_line_contradicts_no_visible_flag')
        if flags.get('only_silhouette_identifiable',False) and (lines['dorsal_surface'] or not lines['silhouette_only']):errors.append('silhouette_status_contradiction')
        if endpoints and not lines['dorsal_surface']:errors.append('reliable_endpoint_requires_visible_surface_fragment')
        for label,p in endpoints.items():
            if lines['dorsal_surface']:
                distance,_=distance_to_fragments(p[None],lines['dorsal_surface'])
                if float(distance[0])>5:errors.append(label+':endpoint_off_visible_curve')
    return dict(reviewed=reviewed,valid=not errors,errors=errors,lines=lines,endpoints=endpoints,
                no_visible=flags.get('no_visible_dorsal_surface',False))


def sample_fragments(fragments, spacing=2.):
    """Resample within each fragment only; never bridge an occlusion gap."""
    if spacing<=0:raise ValueError('spacing must be positive')
    samples=[]
    for line in fragments:
        line=np.asarray(line,dtype=float)
        lengths=np.linalg.norm(np.diff(line,axis=0),axis=1);arc=np.r_[0,np.cumsum(lengths)]
        keep=np.r_[True,np.diff(arc)>1e-10];arc=arc[keep];line=line[keep]
        if len(line)<2:continue
        locations=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/spacing))+1))
        samples.append(np.stack([np.interp(locations,arc,line[:,k]) for k in range(2)],-1))
    return np.concatenate(samples) if samples else np.empty((0,2))


def distance_to_fragments(points,fragments):
    """Closest true segment, without connecting separate line fragments."""
    if not fragments:return np.full(len(points),np.nan),np.full((len(points),2),np.nan)
    starts=np.concatenate([p[:-1] for p in fragments]);ends=np.concatenate([p[1:] for p in fragments])
    segment=ends-starts;delta=np.asarray(points)[:,None]-starts[None]
    t=np.clip((delta*segment).sum(-1)/np.maximum((segment**2).sum(-1),1e-12),0,1)
    closest=starts[None]+t[...,None]*segment[None]
    distance=np.linalg.norm(np.asarray(points)[:,None]-closest,axis=-1);index=distance.argmin(-1)
    return distance[np.arange(len(points)),index],closest[np.arange(len(points)),index]


def compare_fragments(a,b):
    if not a or not b:return dict(comparable=False,reason='one_or_both_no_confirmed_visible_surface')
    pa,pb=sample_fragments(a),sample_fragments(b)
    ab,_=distance_to_fragments(pa,b);ba,_=distance_to_fragments(pb,a)
    def stats(x):return dict(n=len(x),mean_px=float(x.mean()),median_px=float(np.median(x)),p95_px=float(np.quantile(x,.95)))
    length=lambda fragments:float(sum(np.linalg.norm(np.diff(p,axis=0),axis=1).sum() for p in fragments))
    return dict(comparable=True,a_to_b=stats(ab),b_to_a=stats(ba),
                equal_direction_mean_px=float((ab.mean()+ba.mean())/2),
                length_a_px=length(a),length_b_px=length(b),
                within_5px_fraction_a=float((ab<=5).mean()),within_5px_fraction_b=float((ba<=5).mean()),
                caveat='2D annotator agreement only. Different visible supports inflate endpoint distances. Five pixels is descriptive, not an acceptance threshold. No 2D equal-arclength triangulation.')
