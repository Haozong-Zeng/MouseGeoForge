"""Read-only isolate requires_grad and hook effects after strict preflight failure."""
import json
import fcntl
from run_depth_pose_ablation import ROOT,torch,VGGTOmega,write_json,fingerprint,tensor_state,state_hash
from train_cylinder_adaptive_temperature import TemperatureModel
from train_cylinder_view_balance import oriented_batch
from mousegeoforge.adaptive_temperature import restore_adaptive


def run():
    out=ROOT/'output/mask_box_freeze_equivalence_v1';out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Already completed')
    with (ROOT/'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent=ROOT/'output/cylinder_adaptive_temperature_v1';spec=json.loads((parent/'protocol.json').read_text())
        torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        spatial=VGGTOmega(enable_camera=False,enable_depth=True,enable_multi_instance_pose=True,num_keypoints=23)
        weights=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',weights_only=True,mmap=True)
        spatial.load_state_dict({k:v for k,v in weights.items() if k.startswith(('aggregator.','dense_head.'))},strict=False);del weights
        model=TemperatureModel.from_spatial_model(spatial).cuda().eval();del spatial
        original_flags={n:p.requires_grad for n,p in model.named_parameters()}
        state=torch.load(parent/'adaptive_temperature/final_heads.pt',weights_only=True,map_location='cpu')['model'];restore_adaptive(model,state)
        saved=torch.load(parent/'adaptive_temperature/prediction_000.pt',weights_only=True,map_location='cpu');r=saved['record'];images=oriented_batch(r,num_views=len(r['views']))['images'].cuda();rows=[]
        for frozen,hooked in ((False,False),(False,True),(True,False),(True,True),(False,False)):
            for n,p in model.named_parameters():p.requires_grad_(False if frozen else original_flags[n])
            hook=model.instance_pose_head.instance_decoder.register_forward_hook(lambda module,args,output:None) if hooked else None
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):cached,start=model.aggregator(images)
            with torch.no_grad():pred=model.forward_cached(images,cached,start,coupling='off')
            rows.append(dict(all_requires_grad_false=frozen,decoder_hook=hooked,differences={k:float((pred[k].cpu()-v).abs().max()) for k,v in saved['predictions'].items()}))
            if hook:hook.remove()
        assert state_hash(tensor_state(model))==state_hash(state)
        write_json(out/'report.json',dict(rows=rows,optimizer_steps=0,weights_unchanged=True,code_sha256=fingerprint(ROOT/'tools/probe_box_freeze_equivalence.py')))
        print(json.dumps(rows,indent=2))


if __name__=='__main__':run()
