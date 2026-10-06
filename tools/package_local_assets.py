"""Create resumable, verified local asset archives; never delete source assets.

Linux/WSL, Python standard library only. Each tar.gz part is independently
extractable at the workspace root. No Git mutation or network operation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import threading
import time
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.pytest_cache', 'node_modules'}
WEIGHTS = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx'}


def dump(path, data):
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def entries(base):
    if base.is_file() or base.is_symlink():
        yield base
        return
    for folder, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.venv'))
        yield Path(folder)
        for d in list(dirs):
            f = Path(folder) / d
            if f.is_symlink():
                yield f
                dirs.remove(d)
        for name in sorted(files):
            if not name.endswith(('.pyc', ':Zone.Identifier', '.part')):
                yield Path(folder) / name


def make_plan(out, part_gib):
    groups = []
    # Keep model inputs and all experimental outputs, not only selected winners.
    for parent in ('dataset', 'BlenderPyMouseProjects', 'output'):
        for p in sorted((ROOT / parent).iterdir()):
            groups.append((parent + '--' + p.name, [p]))
    dlc = ROOT / 'DeepLabCut'
    for p in sorted(dlc.iterdir()):
        if p.is_dir() and (p.name.startswith('wr_cam') or p.name == 'camc_with_weights'):
            groups.append(('DeepLabCut--' + p.name, [p]))
    for rel in ('img2img-turbo/output', 'water-reaching-dataset'):
        if (ROOT / rel).exists():
            groups.append((rel.replace('/', '--'), [ROOT / rel]))
    # Excludes environments and data projects covered above; includes external
    # pretrained weights outside the main experiment tree.
    for parent in ('vggt-omega', 'vggt', 'gui_based_vggt_o', 'third_party', 'DeepLabCut'):
        roots = []
        for p in entries(ROOT / parent):
            rel = p.relative_to(ROOT / parent)
            if parent == 'DeepLabCut' and rel.parts and (rel.parts[0].startswith('wr_cam') or rel.parts[0] == 'camc_with_weights'):
                continue
            if p.is_file() and not p.is_symlink() and p.suffix.lower() in WEIGHTS:
                roots.append(p)
        if roots:
            groups.append(('weights--' + parent, roots))
    jobs = []
    seen = set()
    ceiling = int(part_gib * 1024**3)
    (out / 'lists').mkdir(parents=True, exist_ok=True)
    for group, roots in groups:
        batch, total, part = [], 0, 1
        def flush():
            nonlocal batch, total, part
            if not batch:
                return
            name = f'{group}--part{part:03d}'
            manifest = out / 'lists' / (name + '.json')
            dump(manifest, batch)
            jobs.append({'name': name, 'manifest': str(manifest.relative_to(out)), 'bytes': total, 'members': len(batch)})
            batch, total, part = [], 0, part + 1
        for base in roots:
            for p in entries(base):
                rel = p.relative_to(ROOT).as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                st = p.lstat()
                size = st.st_size if p.is_file() and not p.is_symlink() else 0
                if batch and total + size > ceiling:
                    flush()
                batch.append({'path': rel, 'bytes': size, 'mtime_ns': st.st_mtime_ns,
                              'type': 'symlink' if p.is_symlink() else 'directory' if p.is_dir() else 'file'})
                total += size
        flush()
    jobs.sort(key=lambda j: j['bytes'])  # Early verified small artifacts, then large datasets.
    originals = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size}
                 for p in sorted(dlc.glob('*.zip'))]
    plan = {'root': str(ROOT), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'jobs': jobs, 'source_bytes': sum(j['bytes'] for j in jobs),
            'existing_download_archives_not_recompressed': originals,
            'excluded': ['backup (previous archives)', 'virtual environments', 'third-party source trees except selected weights', 'cache/partial-download files'],
            'part_target_source_gib': part_gib, 'single_files_larger_than_target_kept_intact': True,
            'source_deletion': False, 'upload': False}
    dump(out / 'plan.json', plan)
    print(json.dumps({'jobs': len(jobs), 'source_gib': plan['source_bytes'] / 1024**3,
                      'free_gib': shutil.disk_usage(out).free / 1024**3}, indent=2), flush=True)


class HashReader:
    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()
    def read(self, n=-1):
        data = self.f.read(n)
        self.h.update(data)
        return data


def run(out, workers):
    with (out / 'packaging.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(out, workers)


def run_locked(out, workers):
    plan = json.loads((out / 'plan.json').read_text())
    if Path(plan['root']) != ROOT:
        raise RuntimeError('Plan root does not match checkout')
    packages, states = out / 'packages', out / 'jobs'
    packages.mkdir(exist_ok=True)
    states.mkdir(exist_ok=True)
    dump(out / 'status.json', {'state': 'running', 'pid': os.getpid(), 'workers': workers, 'jobs': len(plan['jobs']), 'started_at': time.time()})
    print(f'Packaging PID {os.getpid()}; {len(plan["jobs"])} jobs', flush=True)
    def work(job):
        name = job['name']
        statefile = states / (name + '.json')
        archive = packages / (name + '.tar.gz')
        if statefile.exists():
            state = json.loads(statefile.read_text())
            if state.get('state') == 'verified' and archive.exists():
                if archive.stat().st_size == state['archive_bytes'] and digest(archive) == state['sha256']:
                    return state
        if archive.exists():
            raise RuntimeError(f'Refuse to overwrite unverified archive: {archive}')
        if shutil.disk_usage(out).free < max(64 * 1024**3, job['bytes'] * (workers + 1)):
            raise RuntimeError('Insufficient free space; retain sources and partial artifacts')
        rows = json.loads((out / job['manifest']).read_text())
        temp = archive.with_suffix(archive.suffix + '.partial')
        if temp.exists():
            temp.rename(temp.with_name(temp.name + f'.interrupted-{time.time_ns()}'))
        checks = []
        state = {**job, 'state': 'compressing', 'pid': os.getpid(), 'processed_bytes': 0, 'started_at': time.time()}
        dump(statefile, state)
        last = 0
        with tarfile.open(temp, 'w:gz', compresslevel=1, dereference=False) as tar:
            for row in rows:
                p = ROOT / row['path']
                if not p.absolute().is_relative_to(ROOT) or '..' in Path(row['path']).parts:
                    raise RuntimeError('Unsafe manifest path')
                st = p.lstat()
                if st.st_mtime_ns != row['mtime_ns'] or (row['type'] == 'file' and st.st_size != row['bytes']):
                    raise RuntimeError(f'Source changed since plan: {p}')
                # Store hard-linked regular files independently so every file
                # has its own content hash and each part restores independently.
                tar.inodes.clear()
                info = tar.gettarinfo(str(p), arcname=row['path'])
                record = {'path': row['path'], 'type': row['type'], 'bytes': row['bytes']}
                if info.isfile():
                    with p.open('rb') as f:
                        reader = HashReader(f)
                        tar.addfile(info, reader)
                    record['sha256'] = reader.h.hexdigest()
                    after = p.stat()
                    if (after.st_size, after.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
                        raise RuntimeError(f'Source changed while reading: {p}')
                else:
                    tar.addfile(info)
                    if info.issym():
                        record['target'] = info.linkname
                checks.append(record)
                state['processed_bytes'] += row['bytes']
                if time.monotonic() - last > 10:
                    dump(statefile, state)
                    last = time.monotonic()
        state['state'] = 'verifying'
        dump(statefile, state)
        with tarfile.open(temp, 'r|gz') as tar:
            count = 0
            for member, expected in zip(tar, checks, strict=True):
                if member.name != expected['path']:
                    raise RuntimeError('Archive member path mismatch')
                if member.isfile():
                    f = tar.extractfile(member)
                    h = hashlib.sha256()
                    for b in iter(lambda: f.read(8 * 1024**2), b''):
                        h.update(b)
                    if h.hexdigest() != expected['sha256'] or member.size != expected['bytes']:
                        raise RuntimeError('Archive content verification failed')
                elif member.issym() and member.linkname != expected['target']:
                    raise RuntimeError('Symlink target mismatch')
                count += 1
        if count != len(checks):
            raise RuntimeError('Archive count mismatch')
        # Validate gzip trailer/CRC as well, including bytes after tar EOF.
        import gzip
        with gzip.open(temp, 'rb') as stream:
            while stream.read(8 * 1024**2):
                pass
        checksum = digest(temp)
        dump(packages / (name + '.manifest.json'), {'members': checks, 'archive_sha256': checksum})
        temp.rename(archive)
        state.update(state='verified', archive_bytes=archive.stat().st_size, sha256=checksum, completed_at=time.time())
        dump(statefile, state)
        print(f'VERIFIED {name} {state["archive_bytes"]} bytes', flush=True)
        return state
    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_jobs = {pool.submit(work, j): j for j in plan['jobs']}
        for future in as_completed(future_jobs):
            try:
                completed.append(future.result())
            except Exception as exc:
                job = future_jobs[future]
                failures.append({'job': job['name'], 'error': repr(exc)})
                print(f'FAILED {job["name"]}: {exc}', flush=True)
                for other in future_jobs:
                    other.cancel()
                break
            dump(out / 'status.json', {'state': 'running', 'pid': os.getpid(), 'verified': len(completed), 'jobs': len(plan['jobs']), 'archive_bytes': sum(s['archive_bytes'] for s in completed), 'updated_at': time.time()})
    dump(out / 'status.json', {'state': 'failed' if failures else 'complete', 'pid': os.getpid(), 'verified': len(completed), 'jobs': len(plan['jobs']), 'failures': failures, 'updated_at': time.time()})
    if failures:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['plan', 'run', 'launch'])
    p.add_argument('--output', required=True)
    p.add_argument('--part-gib', type=float, default=16)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if a.workers not in (1, 2) or a.part_gib <= 0:
        p.error('Use 1 or 2 workers and positive part size')
    out = Path(a.output).resolve()
    if not out.is_relative_to(ROOT / 'backup'):
        p.error('Local archives must be under this workspace backup directory')
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'plan':
        if (out / 'plan.json').exists():
            p.error('Plan exists; reuse it or choose a new output directory')
        make_plan(out, a.part_gib)
    elif a.mode == 'launch':
        if not (out / 'plan.json').is_file():
            p.error('Create the plan first')
        with (out / 'packaging.log').open('ab') as log:
            proc = subprocess.Popen(['nice', '-n', '10', 'ionice', '-c', '2', '-n', '7',
                                     sys.executable, '-u', str(Path(__file__).resolve()),
                                     'run', '--output', str(out), '--workers', str(a.workers)],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'launched_pid': proc.pid, 'log': str(out / 'packaging.log')}))
    else:
        run(out, a.workers)


if __name__ == '__main__':
    main()
