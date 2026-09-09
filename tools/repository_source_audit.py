"""Audit source-only publication; optionally stage nested source without gitlinks.

No datasets are traversed. Secret findings print filenames/reasons, never values.
This is a heuristic release check, not a comprehensive security scanner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TREES = {'.github', 'BlenderScripts', 'configs', 'docs', 'scripts', 'tools', 'mousegeoforge', 'tests',
         'vggt-omega', 'gui_based_vggt_o', 'literature-and-research-plan'}
ROOT_FILES = {'.gitignore', '.gitattributes', 'README.md', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml'}
PRUNE = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'checkpoints',
         'gui_data', 'output', 'logs', 'wandb', 'collaborator_blenderpymouse_10frame_package'}
SECRETS = {
    'github token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
    'private key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'AWS key': rb'\bAKIA[A-Z0-9]{16}\b',
    'Hugging Face token': rb'\bhf_[A-Za-z0-9]{30,}\b',
    'OpenAI-style key': rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b',
    'URL credentials': rb'https?://[^\s/"\x27:]+:[^\s/"\x27@]+@',
}


def git(*args, data=None):
    return subprocess.run(['git', '-C', str(ROOT), *args], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout


def ignored(paths):
    if not paths:
        return set()
    proc = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index',
                           '-z', '--stdin'], input=b'\0'.join(p.encode() for p in paths)+b'\0',
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.decode())
    return {p.decode() for p in proc.stdout.split(b'\0') if p}


def candidates():
    result = [p for p in ROOT_FILES if (ROOT/p).is_file()]
    for tree in sorted(TREES):
        for base, dirs, files in os.walk(ROOT/tree, followlinks=False):
            dirs[:] = [d for d in dirs if d not in PRUNE and not d.startswith('.venv')
                       and not d.endswith('.egg-info') and not (Path(base)/d).is_symlink()]
            result.extend((Path(base)/f).relative_to(ROOT).as_posix() for f in files)
    result = sorted(result)
    excluded = ignored(result)
    return [p for p in result if p not in excluded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--cached', action='store_true')
    modes.add_argument('--stage', action='store_true')
    args = parser.parse_args()
    if git('rev-parse', '--show-toplevel').decode().strip() != str(ROOT):
        raise SystemExit('Run only with MouseGeoForge as the Git root')
    entries = {}
    if args.cached:
        for entry in git('ls-files', '--stage', '-z').split(b'\0'):
            if entry:
                info, name = entry.split(b'\t', 1)
                mode, sha, stage = info.decode().split()
                if stage != '0':
                    raise SystemExit('Resolve staged conflicts first')
                entries[name.decode()] = (mode, sha)
        paths = sorted(entries)
    else:
        paths = candidates()
    errors, sizes = [], []
    excluded = ignored(paths)
    for name in paths:
        path = ROOT/name
        if name in excluded or (name.split('/')[0] not in TREES and name not in ROOT_FILES):
            errors.append([name, 'outside publication scope'])
            continue
        if args.cached:
            mode, sha = entries[name]
            if mode not in ('100644', '100755'):
                errors.append([name, 'symlink/gitlink forbidden'])
                continue
            size = int(git('cat-file', '-s', sha))
        else:
            if path.is_symlink() or not path.is_file():
                errors.append([name, 'symlink/non-file forbidden'])
                continue
            size = path.stat().st_size
        if size > 5_000_000:
            errors.append([name, 'file exceeds 5 MB source limit'])
            continue
        data = git('cat-file', 'blob', entries[name][1]) if args.cached else path.read_bytes()
        # Raster assets are allowed only if small; archives/datasets are ignored.
        if b'\0' in data and path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.ico', '.webp', '.gif'):
            errors.append([name, 'unexpected binary'])
        for label, pattern in SECRETS.items():
            if re.search(pattern, data):
                errors.append([name, label])
        sizes.append((size, name))
    print(json.dumps({'mode': 'cached' if args.cached else 'source', 'files': len(paths),
                      'bytes': sum(s for s, _ in sizes), 'errors': errors,
                      'largest': sorted(sizes, reverse=True)[:10]}, indent=2))
    if errors:
        raise SystemExit(1)
    if args.stage:
        # Audit again on the indexed blobs below to detect changes during import.
        hashes = git('hash-object', '-w', '--stdin-paths',
                     data=('\n'.join(paths)+'\n').encode()).decode().splitlines()
        assert len(hashes) == len(paths)
        index = b''.join(f'{"100755" if os.access(ROOT/p, os.X_OK) else "100644"} {sha}\t{p}'.encode()+b'\0'
                         for p, sha in zip(paths, hashes))
        git('update-index', '-z', '--index-info', data=index)
        print(f'Staged {len(paths)} audited source files; nested Git metadata unchanged.')
        subprocess.run([os.sys.executable, str(Path(__file__)), '--cached'], check=True)


if __name__ == '__main__':
    main()
