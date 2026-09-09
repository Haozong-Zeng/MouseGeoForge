"""Detach converter-created session hardlinks without touching source inodes."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from core23_temporal_format import write_json_atomic, write_jsonl_atomic

ROOT=Path('/home/haozong/projects/MouseGeoForge/dataset/Orbbec-Openfield')
EXPECTED=ROOT/'sessions'


def main():
    if EXPECTED.resolve()!=EXPECTED or not EXPECTED.is_dir():
        raise ValueError('Unexpected sessions target')
    session_records=[]
    total=0
    total_bytes=0
    for session in sorted(EXPECTED.iterdir()):
        if not session.is_dir():
            continue
        files=0
        byte_count=0
        digest_samples=[]
        for directory,dirs,names in os.walk(session,followlinks=False):
            for name in names:
                p=Path(directory)/name
                if p.is_symlink():
                    raise ValueError(f'Unexpected symlink: {p}')
                stat=p.stat()
                if stat.st_nlink<=1:
                    continue
                if not p.resolve().is_relative_to(EXPECTED):
                    raise ValueError(f'Unsafe copy destination: {p}')
                tmp=p.with_suffix(p.suffix+'.independent-copy.tmp')
                shutil.copy2(p,tmp)
                if tmp.stat().st_size!=stat.st_size:
                    raise IOError(f'Copy size mismatch: {p}')
                if files%5000==0:
                    a=hashlib.sha256(p.read_bytes()).hexdigest()
                    b=hashlib.sha256(tmp.read_bytes()).hexdigest()
                    if a!=b:
                        raise IOError(f'Copy hash mismatch: {p}')
                    digest_samples.append({'path':str(p.relative_to(ROOT)),'sha256':a})
                # Atomic replacement detaches this pathname only; source data stay unchanged.
                os.replace(tmp,p)
                if p.stat().st_nlink!=1:
                    raise IOError(f'Copy is not independent: {p}')
                files+=1
                byte_count+=stat.st_size
                if files%10000==0:
                    print(f'{session.name}: {files} independent files, {byte_count/2**30:.2f} GiB',flush=True)
        rec=json.loads((session/'conversion_complete.json').read_text())
        rec['storage']={'copy':rec['frame_count']*2}
        rec['independent_files']=True
        write_json_atomic(session/'conversion_complete.json',rec)
        session_records.append({k:v for k,v in rec.items() if k!='labeled_scenes'})
        write_json_atomic(session/'independent_copy_audit.json',{'converted_files':files,'converted_bytes':byte_count,
            'sample_hash_checks':digest_samples,'all_replaced_files_size_checked':True,'source_unchanged':True})
        total+=files
        total_bytes+=byte_count
        print(f'Completed {session.name}: {files} files, {byte_count/2**30:.2f} GiB',flush=True)
    write_jsonl_atomic(ROOT/'sequences.jsonl',session_records)
    metadata=json.loads((ROOT/'metadata.json').read_text())
    metadata['complete_sequence_storage']='independent local file copies'
    metadata['independent_files']=True
    write_json_atomic(ROOT/'metadata.json',metadata)
    print(json.dumps({'converted_files':total,'converted_bytes':total_bytes}),flush=True)


if __name__=='__main__':
    main()
