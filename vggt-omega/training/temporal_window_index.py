"""Gap-safe windows and source-group split enforcement, independently of RGB decoding."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def valid_window_starts(frame_ids,segment_ids,transition_valid,*,source_step=1,
                        window_size=16,frame_step=1,window_stride=1):
    """frame_step counts stored samples (a stride-3 source remains stride 3, not 1).

    Check every intervening edge, including those skipped by frame_step. Thus a
    dropped frame/action cut may never be bridged by a strided window.
    """
    if min(source_step,window_size,frame_step,window_stride)<=0:
        raise ValueError('Window and source strides must be positive')
    ids=np.asarray(frame_ids,dtype=np.int64)
    seg=np.asarray(segment_ids)
    transition=np.asarray(transition_valid,dtype=bool)
    if len(seg)!=len(ids) or len(transition)!=len(ids):
        raise ValueError('Index arrays must have equal length')
    if np.any(np.diff(ids)<=0):
        raise ValueError('Source frame IDs must be strictly increasing within a sequence')
    span=(window_size-1)*frame_step
    bad=(np.diff(ids)!=source_step)|(seg[1:]!=seg[:-1])|~transition[1:]
    prefix=np.r_[0,np.cumsum(bad)]
    starts=np.arange(0,max(0,len(ids)-span),window_stride,dtype=np.int64)
    return starts[prefix[starts+span]==prefix[starts]]


class TemporalWindowIndex:
    """Returns frame records/paths; never flattens time into camera views.

    Existing specialized RGB/pose loaders can consume these window references.
    Unknown/pose-set/sparse exports and incomplete full exports are excluded.
    """
    def __init__(self,dataset_root,*,split=None,window_size=16,frame_step=1,
                 window_stride=16,dataset_ids=None,require_rgb=False):
        self.root=Path(dataset_root)
        self.window_size=window_size
        self.frame_step=frame_step
        catalog=json.loads((self.root/'temporal_catalog.json').read_text())['datasets']
        groups={}
        for rec in catalog:
            group=rec.get('motion_source_id')
            if group:
                assigned=rec.get('recommended_split')
                if group in groups and groups[group]!=assigned:
                    raise ValueError(f'Source-group split leakage: {group}')
                groups[group]=assigned
        self.sequences=[]
        self.windows=[]
        for rec in catalog:
            if not rec.get('temporal_eligible') or rec.get('export_complete') is False:
                continue
            if split is not None and rec.get('recommended_split')!=split:
                continue
            if dataset_ids is not None and rec['dataset_id'] not in dataset_ids:
                continue
            if require_rgb and rec.get('rgb_available') is False:
                continue
            path=self.root/rec['path']
            if rec.get('index_format')=='core23_temporal_npz':
                with np.load(path/rec['index'],allow_pickle=False) as f:
                    positions=f['rgb_pose_indices'] if require_rgb else np.arange(len(f['frame_indices']))
                    ids=f['frame_indices'][positions].astype(np.int64)
                rows=None
                seg=np.zeros(len(ids),np.int32)
                valid=np.r_[False,np.diff(ids)==rec.get('source_frame_step',1)]
            else:
                rows=[json.loads(line) for line in (path/rec['index']).read_text().splitlines() if line.strip()]
                ids=np.array([r.get('source_frame_id',r.get('frame_id')) for r in rows],np.int64)
                seg=np.array([r.get('segment_id',rec['dataset_id']) for r in rows])
                valid=np.array([r.get('transition_valid',r.get('consecutive_with_previous',False)) for r in rows])
                positions=np.arange(len(ids))
            starts=valid_window_starts(ids,seg,valid,source_step=rec.get('source_frame_step',1),
                window_size=window_size,frame_step=frame_step,window_stride=window_stride)
            si=len(self.sequences)
            self.sequences.append((rec,path,ids,rows,positions))
            self.windows.extend((si,int(start)) for start in starts)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        si,start=self.windows[index]
        rec,path,ids,rows,source_positions=self.sequences[si]
        positions=start+np.arange(self.window_size)*self.frame_step
        return {'dataset_id':rec['dataset_id'],'root':path,'metadata':rec,
            'motion_source_id':rec['motion_source_id'],'split':rec['recommended_split'],
            'frame_ids':ids[positions].copy(),'frame_delta':np.diff(ids[positions]),
            'source_positions':source_positions[positions].copy(),
            'rows':[rows[int(p)] for p in positions] if rows is not None else None}
