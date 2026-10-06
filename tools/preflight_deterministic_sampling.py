"""GPU guard: deterministic local sampling, forward parity and repeat gradient."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from run_depth_pose_ablation import ROOT,write_json
from train_spatial_anchor_diverse32 import local_refine
from mousegeoforge.deterministic_local_sampling import frozen_bilinear_sample,deterministic_local_refine


def main():
    out=ROOT/'output/deterministic_local_sampling_preflight_v1_retry1';out.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    for n in (1,3):
        p=torch.randn(n,64,108,192,device='cuda')
        g=(torch.rand(n,184,8,2,device='cuda')*2.4-1.2).requires_grad_(True)
        actual=frozen_bilinear_sample(p,g);reference=torch.nn.functional.grid_sample(p,g,align_corners=False)
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=1e-5)
        g1=torch.autograd.grad(actual.square().mean(),g)[0]
        g2=torch.autograd.grad(frozen_bilinear_sample(p,g).square().mean(),g)[0]
        torch.testing.assert_close(g1,g2,atol=0,rtol=0)
        coarse=torch.rand(n,2,23,2,device='cuda',requires_grad=True)
        s=dict(local_pixel=p,local_query=torch.nn.functional.normalize(torch.randn(n,2,23,64,device='cuda'),dim=-1),
               scale=torch.ones(n,2,23,device='cuda')*4,size=8,radius=.1)
        legacy=local_refine(s,coarse);new=deterministic_local_refine(s,coarse)
        torch.testing.assert_close(new,legacy,atol=1e-6,rtol=1e-6)
        grad1=torch.autograd.grad(new.square().mean(),coarse)[0]
        grad2=torch.autograd.grad(deterministic_local_refine(s,coarse).square().mean(),coarse)[0]
        torch.testing.assert_close(grad1,grad2,atol=0,rtol=0)
        assert torch.isfinite(grad1).all()
        rows.append(dict(views=n,pixel_forward_max_difference=float((actual-reference).abs().max()),
                         final_xy_max_difference=float((new-legacy).abs().max()),gradient_repeat_exact=True))
    write_json(out/'verification.json',dict(passed=True,rows=rows,scope='random frozen-feature CPU/GPU primitive tests, not complete checkpoint training replay'))
    print(rows)


if __name__=='__main__':main()
