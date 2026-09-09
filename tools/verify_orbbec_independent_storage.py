"""Verify every archived RGB-D path/size and that all session files are independent."""
import json
import os
from pathlib import Path
from convert_orbbec_openfield_sessions import copy_file
from core23_temporal_format import write_json_atomic

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'dataset/Orbbec-Openfield'
report={'sessions':[],'passed':False}
for line in (ROOT/'sequences.jsonl').read_text().splitlines():
    rec=json.loads(line)
    session=ROOT/rec['session_path']
    source=Path(rec['source_path'])
    preserved=0
    for jp in source.with_name(source.name+'-labeled').glob('*.json'):
        dst=session/'source_annotations'/jp.name
        if not dst.exists():
            copy_file(jp,dst)
            preserved+=1
    expected_ids=[]
    for line in (session/'frames.jsonl').read_text().splitlines():
        frame=json.loads(line)
        expected_ids.append(frame['frame_id'])
        for field in ('rgb','depth_encoded'):
            dst=session/frame[field]
            original=source/dst.name
            assert dst.stat().st_size==original.stat().st_size,(dst,'size')
            assert not os.path.samefile(dst,original),(dst,'still shares source inode')
    files=0
    byte_count=0
    for directory,dirs,names in os.walk(session):
        for name in names:
            p=Path(directory)/name
            assert not p.is_symlink(),p
            assert not p.name.endswith('.tmp'),p
            st=p.stat()
            assert st.st_nlink==1,(p,st.st_nlink)
            files+=1
            byte_count+=st.st_size
    assert len(expected_ids)==rec['frame_count']
    report['sessions'].append({'session_id':rec['session_id'],'paired_frames_verified':len(expected_ids),
        'independent_files':files,'logical_bytes':byte_count,'extra_original_json_preserved':preserved})
    print(f'{rec["session_id"]}: {files} independent files verified',flush=True)
report['passed']=True
report['total_paired_frames']=sum(s['paired_frames_verified'] for s in report['sessions'])
report['total_logical_bytes']=sum(s['logical_bytes'] for s in report['sessions'])
write_json_atomic(BASE/'output/orbbec_openfield_audit/independent_storage_validation.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='sessions'}),flush=True)
