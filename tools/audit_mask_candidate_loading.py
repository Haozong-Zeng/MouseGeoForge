"""CPU-only loading guard proof; reuse unchanged runtime's176 strict GPU replays."""
import json
from run_depth_pose_ablation import ROOT,torch,write_json,fingerprint
from mousegeoforge.mask_box_candidate import MaskBoxCandidate as Original,tensor_mapping_sha256
from mousegeoforge.mask_box_candidate_safe import MaskBoxCandidate
from export_audit_mask_box_candidates import OUT,BASES


def main():
    path=OUT/'loading_guard_verification.json'
    if path.exists():raise RuntimeError('Existing verification; do not overwrite')
    assert json.loads((OUT/'verification.json').read_text())['passed']
    assert MaskBoxCandidate.__call__ is Original.__call__ and MaskBoxCandidate.capture is Original.capture
    encoder=ROOT/'vggt-omega/vggt_omega_1b_512.pt';rows=[]
    for seed,base in BASES.items():
        torch.manual_seed(901);before=torch.random.get_rng_state().clone()
        c=MaskBoxCandidate.load(OUT/seed,base_checkpoint=base,encoder_checkpoint=encoder,repository_root=ROOT)
        assert torch.equal(before,torch.random.get_rng_state())
        manifest=json.loads((OUT/seed/'manifest.json').read_text())
        assert tensor_mapping_sha256(c._refiner.state_dict())==manifest['refiner_state_sha256']
        wrong=BASES['seed29' if seed=='seed17' else 'seed17']
        try:MaskBoxCandidate.load(OUT/seed,base_checkpoint=wrong,encoder_checkpoint=encoder,repository_root=ROOT)
        except ValueError:pass
        else:raise AssertionError('Bad pair accepted')
        assert torch.equal(before,torch.random.get_rng_state())
        with c.capture():
            p={'sentinel':object()};assert c(p) is p
        rows.append(dict(seed=seed,successful_and_failed_load_cpu_rng_unchanged=True,refiner_state_exact=True,default_identity=True))
    write_json(path,dict(passed=True,rows=rows,optimizer_steps=0,encoder_forwards=0,
        original_runtime_verification_sha256=fingerprint(OUT/'verification.json'),
        source_sha256={p:fingerprint(ROOT/p) for p in ('mousegeoforge/mask_box_candidate_safe.py','tools/audit_mask_candidate_loading.py')},
        proof='Guard only wraps load in CPU RNG preservation. Loaded state identical; capture and forward are the exact same inherited function objects validated by prior176 GPU replays. No new GPU replays in this guard audit.',production_promoted=False))
    print(path.read_text())


if __name__=='__main__':main()
