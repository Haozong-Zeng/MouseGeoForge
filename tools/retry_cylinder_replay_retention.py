"""Versioned retry: capture stopped teacher through the actual train-mode path."""
import json
from unittest.mock import patch
import preflight_cylinder_replay_retention as original

OUT = original.ROOT / 'output/cylinder_retention_preflight_v1_retry'
FAILED = original.OUT
forward = original.DepthPoseModel.forward_cached
write_json = original.write_json
capture_count = 0


def training_path(self, *args, **kwargs):
    global capture_count
    self.train()  # Frozen aggregator remains eval; student heads follow engine.train().
    detached = not original.torch.is_grad_enabled()
    with original.torch.enable_grad():
        result = forward(self, *args, **kwargs)
    if detached:
        capture_count += 1
        result = {k: v.detach().clone() for k, v in result.items()}
    return result


def provenance_write(path, value):
    if path.name == 'protocol.json':
        value['code_sha256']['tools/retry_cylinder_replay_retention.py'] = original.fingerprint(original.ROOT / 'tools/retry_cylinder_replay_retention.py')
        value['retry'] = dict(failed_protocol_sha256=original.fingerprint(FAILED / 'protocol.json'),
            mode_audit_sha256=original.fingerprint(original.ROOT / 'output/cylinder_teacher_capture_mode_v1/report.json'),
            change='All decoder forwards use train mode and enabled autograd; teacher outputs immediately detached. No optimizer updates, no gate relaxation.')
    if path.name == 'report.json':
        assert capture_count == 8
        value['train_mode_detached_teacher_captures'] = capture_count
        value['retry_code_unchanged'] = original.fingerprint(original.ROOT / 'tools/retry_cylinder_replay_retention.py') == json.loads((OUT / 'protocol.json').read_text())['code_sha256']['tools/retry_cylinder_replay_retention.py']
        assert value['retry_code_unchanged']
    write_json(path, value)


if __name__ == '__main__':
    original.OUT = OUT
    original.write_json = provenance_write
    with patch.object(original.DepthPoseModel, 'forward_cached', training_path):
        original.main()
