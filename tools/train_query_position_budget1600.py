"""Independent paired1600 budget sensitivity. Never alters locked800 artifacts."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import time
import train_query_position_pilot as pilot
from run_depth_pose_ablation import ROOT, torch, write_json, fingerprint, tensor_state, state_hash

OUT = ROOT / 'output/query_position_seed17_budget1600_v1'
PRIOR = ROOT / 'output/query_position_seed17_v1'
AUDIT = ROOT / 'output/query_position_budget_audit_v1/report.json'
ORIGINAL_CONFIG = pilot.config
ORIGINAL_OPTIMIZER = pilot.optimizer


def config():
    cfg = ORIGINAL_CONFIG()
    cfg.update(steps=1600, max_run_hours=1.0,
               checkpoint_policy='Fresh original warm start; two identical800-slot cycles, uninterrupted AdamW1600. Fixed800 snapshot and final1600; no best-dev selection or automatic doubling.',
               scope='Budget sensitivity after explicit user authorization; same architecture/data/head-only scope as locked800 pilot')
    return cfg


def prepare():
    if (OUT / 'protocol.json').exists():
        raise RuntimeError('Already prepared, preserve')
    assert AUDIT.exists()
    assert json.loads((PRIOR / 'verification.json').read_text())['passed']
    spec = copy.deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
    for p, h in spec['code_sha256'].items():
        assert fingerprint(ROOT / p) == h
    assert spec['schedule'] == list(range(800))
    assert all(len(v) == 800 and all(len(x) == 1 for x in v) for v in spec['arms'].values())
    assert spec['arms']['query_position_off'] == spec['arms']['query_position_on']
    # Reuse exactly the same800 scene-index choices twice; no new train records.
    spec['schedule'] *= 2
    spec['slots'] *= 2
    assert len(spec['schedule']) == len(spec['slots']) == 1600
    assert spec['schedule'][:800] == spec['schedule'][800:]
    spec['config'] = config()
    spec['scope'] = config()['scope']
    spec['budget_policy'] = dict(steps_per_arm=1600, images_per_arm=3600,
                                patch_tokens_per_arm=7080768, new_training_poses=0,
                                prior_optimizer_state_available=False,
                                rerun_from_original_for_uninterrupted_optimizer=True)
    spec['acceptance_policy'] = 'Original17 gates; ON-OFF dual PCK>=.01, NME<=1.02xOFF, dualTP not lower, all-viewFP and duplicate mask views not higher. Compare800 and1600 at fixed endpoints; no best checkpoint. If no benefit or retention failure, do not extend this mechanism again on training loss alone; pivot to separately preregistered spatial-candidate architecture. Independent seed29 only with practical retention gain.'
    spec['prior800_protocol_sha256'] = fingerprint(PRIOR / 'protocol.json')
    spec['budget_audit_sha256'] = fingerprint(AUDIT)
    spec['code_sha256'].update({p: fingerprint(ROOT / p) for p in (
        'tools/train_query_position_budget1600.py', 'tools/audit_query_position_training_budget.py')})
    write_json(OUT / 'protocol.json', spec)
    write_json(OUT / 'status.json', dict(state='prepared', total_steps=3200))


def build_protocol(cfg):
    spec = json.loads((OUT / 'protocol.json').read_text())
    assert cfg == config() == spec['config']
    assert fingerprint(PRIOR / 'protocol.json') == spec['prior800_protocol_sha256']
    assert fingerprint(AUDIT) == spec['budget_audit_sha256']
    assert fingerprint(pilot.CHECKPOINT) == spec['checkpoint_sha256']
    assert fingerprint(pilot.PREFLIGHT / 'report.json') == spec['preflight_sha256']
    assert fingerprint(ROOT / 'vggt-omega/vggt_omega_1b_512.pt') == spec['encoder_sha256']
    assert fingerprint(ROOT / cfg['manifest']) == spec['manifest_sha256']
    for p, h in spec['code_sha256'].items():
        assert fingerprint(ROOT / p) == h
    for p, h in spec['source_sha256'].items():
        assert pilot.base.scene_digest(ROOT / p) == h
    return spec


def optimizer(*args, **kwargs):
    opt = ORIGINAL_OPTIMIZER(*args, **kwargs)
    original_step = opt.step
    arm = pilot.ARMS[pilot.OPTIMIZERS - 1]
    steps = 0

    def step(*a, **kw):
        nonlocal steps
        value = original_step(*a, **kw)
        steps += 1
        if steps in (800, 1600):
            # Saving states alone causes no forward, optimizer reset or RNG draw.
            state = tensor_state(pilot.MODEL)
            torch.save(dict(model=state, optimizer=opt.state_dict(), steps=steps, arm=arm,
                            model_sha256=state_hash(state), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all(),
                            protocol_sha256=fingerprint(OUT / 'protocol.json')),
                       OUT / arm / f'training_state_{steps:04d}.pt')
        return value

    opt.step = step
    return opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        return
    if args.launch:
        assert json.loads((OUT / 'status.json').read_text())['state'] == 'prepared'
        with (OUT / 'run.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, '-u', __file__], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, output=str(OUT))))
        return
    if time.time() + 3600 >= pilot.diagnostics.DEADLINE:
        raise RuntimeError('Insufficient time before authorized deadline')
    with (ROOT / 'output/cylinder_training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pilot.OUT = OUT
        pilot.diagnostics.OUT = OUT
        pilot.diagnostics.ARMS = pilot.ARMS
        engine = pilot.base.engine
        engine.build_protocol = build_protocol
        engine.load_batch = pilot.base.oriented_batch
        engine.DepthPoseModel = pilot.QueryPositionModel
        engine.restore = pilot.restore
        engine.torch.optim.AdamW = optimizer
        engine.torch.nn.utils.clip_grad_norm_ = pilot.selected_clip
        engine.evaluate_scene = pilot.checked_metrics
        engine.instance_pose_loss = pilot.diagnostics.checked_loss
        engine.run(OUT, config(), False)
        build_protocol(config())
        assert pilot.EVAL_CALLS == 264 and pilot.OPTIMIZERS == 2
        assert all(len(layer.multihead_attn._forward_pre_hooks) == 0 for layer in pilot.MODEL.instance_pose_head.instance_decoder.layers)
        for arm in pilot.ARMS:
            history = json.loads((OUT / arm / 'history.json').read_text())
            assert len(history) == 1600
            assert history[-1]['images_seen'] == 3600
            assert history[-1]['patch_tokens_seen'] == 7080768
            for n in (800, 1600):
                assert (OUT / arm / f'training_state_{n:04d}.pt').exists()
        write_json(OUT / 'verification.json', dict(passed=True, evaluations=264,
                   source_weights_code_unchanged=True, initial_prediction_max_difference=0,
                   depth_output_max_difference=0, non_head_state_unchanged=True,
                   optimizer_steps_per_arm=1600, encoder_gradient_free=True, hooks_cleaned=True,
                   midpoint_and_final_optimizer_states_saved=True,
                   protocol_sha256=fingerprint(OUT / 'protocol.json'), production_promoted=False))


if __name__ == '__main__':
    main()
