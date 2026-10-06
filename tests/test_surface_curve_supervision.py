import torch
from mousegeoforge.surface_curve_supervision import point_loss,back_tangents,curve_loss,points_to_polyline,bezier,cubic_bspline

def points():
    gt=torch.zeros(23,2);gt[4:8]=torch.tensor([[100.,100.],[200.,100.],[300.,100.],[400.,100.]])
    valid=torch.zeros(23,dtype=torch.bool);valid[4:8]=True
    weak=torch.zeros_like(valid);weak[5:7]=True
    return gt,valid,weak

def arc():
    t=torch.linspace(0,1,65);return torch.stack([100+300*t,250-90*torch.sin(torch.pi*t)],-1)

def test_along_vs_normal_and_strong_semantics():
    gt,v,w=points();along=gt.clone();along[5,0]+=20;normal=gt.clone();normal[5,1]+=20
    assert point_loss(along,gt,v,weak_back=w)<point_loss(normal,gt,v,weak_back=w)*.3
    assert point_loss(along,gt,v)==point_loss(normal,gt,v)
    nose=gt.clone();v[0]=True;nose[0,0]+=20
    assert point_loss(nose,gt,v,weak_back=w)==point_loss(nose,gt,v)

def test_missing_neighbor_does_not_invent_tangent():
    gt,v,w=points();gt[4]=float('nan');v[4]=False
    t,known=back_tangents(gt,v);assert not known[5] and known[6]
    pred=torch.nan_to_num(gt).requires_grad_();loss=point_loss(pred,gt,v,weak_back=w);loss.backward();assert torch.isfinite(pred.grad).all()

def test_all_missing_differentiable_zero():
    pred=torch.full((2,23,2),float('nan'),requires_grad=True);gt=pred.detach().clone();v=torch.zeros(2,23,dtype=torch.bool)
    loss=point_loss(pred,gt,v);assert loss==0;loss.backward();assert torch.isfinite(pred.grad).all() and pred.grad.eq(0).all()

def test_arbitrary_instance_axes_and_no_cross_instance_average():
    gt,v,w=points();gt=gt.expand(2,3,2,23,2).clone();v=v.expand(2,3,2,23);w=w.expand_as(v);p=gt.clone();p[:,:,1,:,1]+=20
    combined=point_loss(p,gt,v,weak_back=w)
    expected=(point_loss(p[:,:,0],gt[:,:,0],v[:,:,0],weak_back=w[:,:,0])+point_loss(p[:,:,1],gt[:,:,1],v[:,:,1],weak_back=w[:,:,1]))/2
    torch.testing.assert_close(combined,expected)

def test_wrong_id_cannot_be_weakened():
    gt,v,w=points();w[22]=True
    try:point_loss(gt,gt,v,weak_back=w)
    except ValueError:return
    raise AssertionError('Abdomen accidentally weakened')

def test_curve_reparameterization_order_collapse_and_normal_shift():
    gt=arc();valid=torch.ones(len(gt),dtype=torch.bool);t=torch.linspace(0,1,65).square();reparam=torch.stack([100+300*t,250-90*torch.sin(torch.pi*t)],-1)
    good=curve_loss(reparam,gt,valid,full_curve=True)
    normal=curve_loss(gt+torch.tensor([0.,30.]),gt,valid,full_curve=True)
    reverse=curve_loss(gt.flip(0),gt,valid,full_curve=True)
    collapse=curve_loss(gt[32].expand_as(gt),gt,valid,full_curve=True)
    assert good<1e-6 and min(normal,reverse,collapse)>good+1e-4
    scrambled=gt.clone();scrambled[10:25]=scrambled[10:25].flip(0)
    assert curve_loss(scrambled,gt,valid,full_curve=True)>1e-5

def test_partial_missing_endpoints_not_fabricated():
    gt=arc();valid=torch.zeros(len(gt),dtype=torch.bool);valid[15:28]=True;valid[40:50]=True
    missing=gt.clone();missing[~valid]=float('nan');pred=gt.clone();pred[:10,0]-=200;pred[55:,0]+=200;pred.requires_grad_()
    loss=curve_loss(pred,missing,valid,full_curve=False);assert loss<1e-8;loss.backward();assert torch.isfinite(pred.grad).all()
    assert pred.grad[[0,-1]].abs().sum()==0

def test_polyline_never_bridges_missing_segment():
    line=torch.tensor([[0.,0.],[1.,0.],[float('nan'),float('nan')],[3.,0.],[4.,0.]])
    result=points_to_polyline(torch.tensor([[2.,0.]]),line,torch.tensor([True,True,False,True,True]))
    torch.testing.assert_close(result[0],torch.tensor([1.]))

def test_two_points_do_not_become_curve_truth():
    pred=arc().requires_grad_();valid=torch.zeros(65,dtype=torch.bool);valid[[10,50]]=True
    loss=curve_loss(pred,arc(),valid,full_curve=False);assert loss==0;loss.backward();assert pred.grad.eq(0).all()

def test_bezier_finite_gradients_and_endpoint_definition():
    c=torch.tensor([[100.,300.],[200.,100.],[350.,100.],[400.,300.]],requires_grad=True);p=bezier(c)
    torch.testing.assert_close(p[[0,-1]],c[[0,-1]])
    loss=curve_loss(p,arc(),torch.ones(65,dtype=torch.bool),full_curve=True);loss.backward();assert torch.isfinite(c.grad).all() and c.grad.abs().sum()>0

def test_cubic_bspline_matches_reference_and_endpoints():
    import numpy as np
    from scipy.interpolate import BSpline
    controls=torch.tensor([[0.,1.],[1.,3.],[2.,4.],[3.,2.],[4.,1.],[5.,0.]],dtype=torch.float64,requires_grad=True)
    curve=cubic_bspline(controls,65);knots=np.r_[np.zeros(4),1/3,2/3,np.ones(4)]
    reference=BSpline(knots,controls.detach().numpy(),3)(np.linspace(0,1,65))
    torch.testing.assert_close(curve,torch.tensor(reference),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(curve[[0,-1]],controls[[0,-1]])
    curve.square().sum().backward();assert torch.isfinite(controls.grad).all()
