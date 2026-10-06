"""Read-only release inventory; does not stage, commit, push, or delete files."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import hashlib
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--snapshot', action='store_true', help='Write and verify a source-only ZIP; does not stage or commit')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    def git(*a):
        return subprocess.check_output(['git', '-C', str(root), *a])
    tracked = set(git('ls-files', '-z').decode().split('\0')) - {''}
    names = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')) - {''})
    text = {}
    files, missing, large, secrets = [], [], [], []
    patterns = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----', r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}', r'AKIA[A-Z0-9]{16}', r'sk-[A-Za-z0-9]{32,}']
    for name in names:
        f = root / name
        if not f.is_file():
            missing.append(name)
            continue
        size = f.stat().st_size
        files.append({'path': name, 'bytes': size, 'tracked': name in tracked})
        if size >= 10 * 1024**2:
            large.append({'path': name, 'bytes': size})
        if size < 5 * 1024**2:
            try:
                body = f.read_text(encoding='utf-8')
            except (UnicodeError, OSError):
                continue
            text[name] = body
            for pattern in patterns:
                if re.search(pattern, body):
                    secrets.append({'path': name, 'pattern': pattern})
    candidates = []
    for name in text:
        if name in tracked:
            continue
        base = Path(name).name
        selected = (name.startswith('BlenderScripts/') and re.match(r'(probe_|inspect_realism_|diagnose_realism_|build_realism_v\d|render_realism_v\d)', base)) or (name.startswith('tools/review_realism_'))
        if selected:
            refs = [n for n, body in text.items() if n != name and (base in body or Path(base).stem in body)]
            candidates.append({'path': name, 'references': refs, 'unreferenced': not refs})
    report = {'git_head': git('rev-parse', 'HEAD').decode().strip(), 'files': files,
              'file_count': len(files), 'bytes': sum(x['bytes'] for x in files),
              'tracked_count': len(tracked), 'missing': missing, 'large_files': large,
              'possible_secrets': secrets, 'oneoff_candidates': candidates,
              'extensions': dict(collections.Counter(Path(x['path']).suffix for x in files)),
              'note': 'Heuristic secret/reference checks are not a security or dependency proof. No source edits or Git staging performed.'}
    (out / 'source_inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (out / 'git_status_before.txt').write_bytes(git('status', '--short'))
    (out / 'tracked_changes.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    if args.snapshot:
        if missing or large or secrets:
            raise RuntimeError('Resolve inventory blockers before producing a source snapshot')
        archive = out / 'MouseGeoForge-source.zip'
        if archive.exists():
            raise FileExistsError(archive)
        manifest = []
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for row in files:
                data = (root / row['path']).read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                z.writestr('MouseGeoForge/' + row['path'], data)
                manifest.append({'path': row['path'], 'bytes': len(data), 'sha256': checksum})
        with zipfile.ZipFile(archive) as z:
            for row in manifest:
                if hashlib.sha256(z.read('MouseGeoForge/' + row['path'])).hexdigest() != row['sha256']:
                    raise RuntimeError('Source ZIP verification failed')
        (out / 'source_manifest.json').write_text(json.dumps({'files': manifest, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'oneoff_candidates')}, indent=2))
    print('UNREFERENCED ONE-OFFS:', json.dumps([x['path'] for x in candidates if x['unreferenced']], indent=2))


if __name__ == '__main__':
    main()
