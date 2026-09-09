"""Discover only completed full-length DANNCE pseudo-label chunks."""
import json


def full_annotations(session):
    root=session/'pseudo_depth_full_v1'
    rows={}
    chunks=[]
    invalid=0
    warnings=0
    for marker in sorted(root.glob('chunks/*/chunk_complete.json')):
        meta=json.loads(marker.read_text())
        assert meta['validated']
        chunk_rows=[json.loads(s) for s in (marker.parent/'frames.jsonl').read_text().splitlines()]
        assert len(chunk_rows)==meta['frame_count']
        assert [r['source_frame_id'] for r in chunk_rows]==list(range(meta['frame_range'][0],meta['frame_range'][1]+1,4))
        for row in chunk_rows:
            fid=row['source_frame_id']
            assert fid not in rows,'Overlapping complete pseudo-depth chunks'
            rows[fid]={k:v for k,v in row.items() if k!='source_frame_id'}
        invalid+=meta['invalid_depth_frames']
        warnings+=meta['warning_frames']
        chunks.append({'path':str(marker.parent.relative_to(session)),**meta})
    complete=set(rows)==set(range(0,18000,4))
    return rows,{'path':'pseudo_depth_full_v1','frame_count':len(rows),'expected_frame_count':4500,
        'source_frame_step':4,'fps':25.,'source_fps':100.,
        'complete':complete,'invalid_depth_frames':invalid,'warning_frames':warnings,'chunks':chunks,
        'depth_unit':'mm','depth_semantics':'camera_z','pfm_flip_y':True,'foreground_only':True,
        'depth_is_pseudo_label':True,'mask_is_pseudo_label':True,'confidence_min_votes_recommended':5,
        'training_default':False}
