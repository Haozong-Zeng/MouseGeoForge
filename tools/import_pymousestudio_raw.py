"""Import the explicitly scoped September12 source archives without overwrites."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

SESSIONS=('AZL3','AZR2','AZR2L2','CR2','EL2','EL2L2','ER3L2','FR2','FR3','GR2','UR2','UR3')


def safe_member(name,session):
    p=PurePosixPath(name)
    if '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=session:
        raise ValueError(f'Unsafe or unexpected archive member: {name}')
    return p


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def run(source,destination):
    destination=destination.resolve()
    if destination.exists():raise FileExistsError('Preserve existing destination; no overwrite import')
    archives=[];total=0
    for session in SESSIONS:
        found=list(source.glob(f'{session}-20260912T*-1-001.zip'))
        if len(found)!=1:raise ValueError(f'Expected exactly one archive for {session}: {found}')
        archive=found[0]
        with zipfile.ZipFile(archive) as z:
            seen=set()
            for item in z.infolist():
                safe_member(item.filename,session)
                if item.filename in seen or stat.S_ISLNK(item.external_attr>>16):raise ValueError('Duplicate or symlink archive member')
                seen.add(item.filename);total+=item.file_size
            if f'{session}/calibration.toml' not in seen:raise ValueError('Missing calibration')
        archives.append((session,archive))
    if shutil.disk_usage(destination.parent).free<total+(1<<30):raise OSError('Insufficient free space')
    destination.mkdir();metadata=destination/'_metadata';metadata.mkdir()
    report=dict(state='extracting',source=str(source),destination=str(destination),uncompressed_bytes=total,archives=[],pickle_deserialized=False)
    def save(): (metadata/'import.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for session,archive in archives:
        result=dict(session=session,archive=str(archive),archive_sha256=sha256(archive),files=[])
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                rel=safe_member(item.filename,session);target=destination.joinpath(*rel.parts)
                if not target.resolve().is_relative_to(destination):raise ValueError('Target escaped destination')
                if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
                with z.open(item) as src,target.open('xb') as dst:
                    for chunk in iter(lambda:src.read(8<<20),b''):dst.write(chunk);h.update(chunk);count+=len(chunk)
                assert count==item.file_size  # ZipExtFile also verifies CRC at EOF.
                result['files'].append(dict(path=str(rel),bytes=count,sha256=h.hexdigest(),crc32=item.CRC))
        report['archives'].append(result);save();print('IMPORTED',session,len(result['files']),flush=True)
    report['state']='completed';save();print('COMPLETE',total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();run(a.source,a.destination)
