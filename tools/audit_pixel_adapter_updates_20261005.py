"""Post-pilot parameter and prediction-change audit; no further training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
from run_depth_pose_ablation import write_json
P=ROOT/'output/singleview_fusion_cycle_20261005/pixel_adapter_matched_pilot_v1'
def read(p):return torch.load(p,map_location='cpu',weights_only=False)
def main():
    assert json.loads((P/'status.json').read_text())['stage']=='complete'
    torch.set_num_threads(2);initial=read(P/'initial.pt');raw=torch.load(ROOT/'vggt-omega/vggt_omega_1b_512.pt',map_location='cpu',weights_only=True,mmap=True)
    base=raw['aggregator.patch_embed.blocks.23.attn.qkv.weight'];result={}
    checkpoints={arm:read(P/f'{arm}_step0400.pt') for arm in ['head_only','head_plus_lora']}
    for arm,c in checkpoints.items():
        a,b=c['adapter']['a'],c['adapter']['b'];delta=b@a
        headnorm=sum((v-initial['pose'][k]).float().square().sum().item() for k,v in c['pose'].items())**.5
        result[arm]=dict(head_delta_norm=headnorm,lora_a_delta=float((a-initial['adapter']['a']).norm()),lora_b_norm=float(b.norm()),effective_qkv_delta_norm=float(delta.norm()),relative_qkv_delta=float(delta.norm()/base.float().norm()),effective_qkv_max_abs=float(delta.abs().max()))
    data=read(P/'predictions.pt');ids=data['evaluated_ids'];pixels=data['pixels'][ids,None,None,:]
    confidence=read(P.parent/'point_quality_inputs_v1/inputs.pt')['confident'][ids]
    for arm in checkpoints:
        pred=data['predictions'][arm][ids];baseline=data['predictions']['baseline'][ids];drift=((pred-baseline)*pixels).norm(dim=-1)
        result[arm]['allquery_prediction_drift']={label:dict(mean=float(drift[:,:,kp].mean()),max=float(drift[:,:,kp].max())) for label,kp in [('supervised_five',[0,14,15,16,17]),('tail_unsupervised',list(range(7,14))),('abdomen_unsupervised',[22]),('all23',list(range(23)))]}
        result[arm]['confident_query_prediction_drift']={label:dict(mean=float(drift[:,:,kp][confidence].mean()),max=float(drift[:,:,kp][confidence].max())) for label,kp in [('supervised_five',[0,14,15,16,17]),('tail_unsupervised',list(range(7,14))),('abdomen_unsupervised',[22]),('all23',list(range(23)))]}
    difference=((data['predictions']['head_plus_lora'][ids]-data['predictions']['head_only'][ids])*pixels).norm(dim=-1)
    result['between_arms_allquery_drift']=dict(mean=float(difference.mean()),max=float(difference.max()))
    result['scope']='All6query drift measures coordinate change, not accuracy; includes background queries and unsupervisedpoints. Original checkpoints/labels unchanged, experimental retraining can change non-supervised outputs. No promotion.'
    write_json(P/'weight_update_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
