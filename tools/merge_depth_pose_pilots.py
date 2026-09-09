"""Collect disjoint seed workers; never select the better runs or overwrite output."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def read(p): return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--worker',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); cfg=read(a.config)
    if a.output.exists(): raise FileExistsError('Merged output must be new')
    bundles=[read(w/'results.json') for w in a.worker]
    manifests=[read(w/'manifest.json') for w in a.worker]
    runs=[r for b in bundles for r in b['runs']]
    assert sorted((r['seed'],r['mode']) for r in runs)==sorted((s,m) for s in cfg['seeds'] for m in cfg['modes'])
    for b,m in zip(bundles,manifests):
        assert b['passed']
        assert {k:v for k,v in b['config'].items() if k!='seeds'}=={k:v for k,v in cfg.items() if k!='seeds'}
        assert m['records']==manifests[0]['records']
        assert b['provenance']['checkpoint_sha256']==bundles[0]['provenance']['checkpoint_sha256']
    a.output.mkdir(parents=True)
    for worker,b in zip(a.worker,bundles):
        for r in b['runs']:
            name=f"seed{r['seed']}_{r['mode']}"
            shutil.copytree(worker/name,a.output/name)  # own generated artifacts; input workers preserved
    merged=dict(config=cfg,provenance=bundles[0]['provenance'],runs=runs,passed=True,
                workers=[str(w) for w in a.worker],
                elapsed_seconds=max(b['elapsed_seconds'] for b in bundles),
                peak_gpu_bytes_per_worker=[b['peak_gpu_bytes'] for b in bundles])
    manifest=manifests[0]; manifest['config']=cfg
    for name,data in (('results.json',merged),('manifest.json',manifest)):
        (a.output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(a.worker[0]/'data_audit.json',a.output/'data_audit.json')
    print(f'Merged exactly {len(runs)} runs, input workers preserved: {a.output}')


if __name__=='__main__': main()
