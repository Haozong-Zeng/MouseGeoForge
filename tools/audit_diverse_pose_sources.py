"""Probe complete-label training sources without touching held-out images."""
import json
from run_depth_pose_ablation import ROOT, torch, load_scene, load_batch, depth_alignment_audit, write_json


def main():
    torch.set_num_threads(2)
    groups=json.loads((ROOT/'dataset/motion_source_groups.json').read_text())['groups']
    groups['moseq_sample10']=dict(split='train',members=[
        'BlenderPyMouse/Homecage-BlackFur-KeypointMoseq-Core23-Sample10-Frames200-399',
        'BlenderPyMouse/PyMouseStudio-BrownFur-KeypointMoseq-Core23-Sample10-First200'])
    rows=[]
    for group,spec in groups.items():
        if spec['split']!='train': continue
        for member in spec['members']:
            if not member.startswith('BlenderPyMouse/') or member.endswith('Transfered'): continue
            paths=sorted((ROOT/'dataset'/member).glob('frame_*'))
            for path in [paths[0],paths[len(paths)//2],paths[-1]]:
                scene=load_scene(path)
                row=dict(group=group,member=member,scene=path.name,count=len(paths),views=list(scene.frames))
                for flip in (False,True):
                    record=dict(scene=str(path.relative_to(ROOT)),views=[0,2],image_hw=[512,512],
                                expected_instances=1,pfm_flip_y=flip)
                    try:
                        batch=load_batch(record)
                        row[str(flip)]=depth_alignment_audit(batch)
                    except Exception as error:
                        row[str(flip)]=dict(error=str(error))
                rows.append(row); print(json.dumps(row),flush=True)
    write_json(ROOT/'output/diverse_pose_source_audit.json',rows)


if __name__=='__main__': main()
