"""Inventory legacy Orbbec exports; optionally remove only Zone.Identifier files."""
import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remove-zone-identifiers', action='store_true')
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    if root.name != 'Orbbec OpenField' or root.parent != Path('/home/haozong/projects'):
        raise ValueError(f'Unexpected source root: {root}')
    folders = {}
    removed = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(directory, d).is_symlink()]
        counts = Counter()
        examples = {}
        for name in files:
            p = Path(directory, name)
            if name.endswith(':Zone.Identifier') or name == 'Zone.Identifier':
                if args.remove_zone_identifiers:
                    if p.is_symlink() or not p.resolve().is_relative_to(root):
                        raise ValueError(f'Unsafe target: {p}')
                    size = p.stat().st_size
                    p.unlink()
                    removed.append({'path': str(p.relative_to(root)), 'bytes': size})
                continue
            ext = p.suffix.lower()
            counts[ext] += 1
            examples.setdefault(ext, [])
            if len(examples[ext]) < 3:
                examples[ext].append(name)
        folders[str(Path(directory).relative_to(root))] = {'counts': dict(counts), 'examples': examples}
    report = {'source': str(root), 'removed_zone_identifiers': removed,
              'removed_count': len(removed), 'removed_bytes': sum(x['bytes'] for x in removed),
              'folders': folders}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == '.gz' else open
    with opener(args.output, 'wt', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: report[k] for k in ('source', 'removed_count', 'removed_bytes')}))
    for path, info in folders.items():
        if '-labeled' in path or path.count('/') <= 1:
            print(path, info)


if __name__ == '__main__':
    main()
