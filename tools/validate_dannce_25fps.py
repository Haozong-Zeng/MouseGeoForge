"""Read-only validation of storage sampling, source clocks, and available pseudo labels."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'vggt-omega'))
from training.dannce_temporal_dataset import DANNCECompleteTemporalDataset,read_depth_pfm
from core23_temporal_format import write_json_atomic


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-full-depth',action='store_true')
    args=parser.parse_args()
    results=[]
    for mouse in (1,2):
        session=BASE/f'dataset/DANNCE-MarkerlessMouse{mouse}-Core23-RealMultiview/sessions/markerless_mouse_{mouse}'
        metadata=json.loads((session/'metadata.json').read_text())
        rows=[json.loads(s) for s in (session/'frames.jsonl').read_text().splitlines()]
        ids=np.arange(0,18000,4)
        assert metadata['frame_count']==len(rows)==4500 and metadata['fps']==25 and metadata['source_frame_step']==4
        assert [r['source_frame_id'] for r in rows]==ids.tolist()
        assert [r['local_index'] for r in rows]==list(range(4500))
        assert np.allclose([r['time_seconds'] for r in rows],ids/100.)
        assert all(r['source_frame_delta']==4 and r['transition_valid'] for r in rows[1:])
        with np.load(session/'sync_com.npz') as com:
            assert np.array_equal(com['source_frame_id'],ids)
            assert com['com_world_xyz_mm'].shape==(4500,3)
        checked=0
        available=0
        rgb_bytes=0
        for view in range(6):
            files=sorted((session/f'rgb/{view:08d}').glob('*.jpg'))
            assert [int(p.stem) for p in files]==ids.tolist()
            rgb_bytes+=sum(p.stat().st_size for p in files)
        for row in rows:
            if row.get('depth'):
                available+=1
                for view in range(6):
                    path=session/row['depth'][view]
                    with path.open('rb') as handle:
                        assert handle.readline().strip()==b'Pf'
                        assert list(map(int,handle.readline().split()))==[1152,1024]
                        assert float(handle.readline())<0
                        assert path.stat().st_size-handle.tell()==1152*1024*4
                    for key in ('mask','confidence'):
                        assert (session/row[key][view]).is_file()
                    checked+=1
            elif args.require_full_depth:
                raise AssertionError(f'Missing full-depth frame {mouse}/{row["source_frame_id"]}')
        dataset=DANNCECompleteTemporalDataset(BASE/'dataset',split='train' if mouse==1 else 'test',
            window_size=2,window_stride=1,image_size=(288,256),view_ids=(0,2),load_pseudo_depth=True)
        for index in (0,len(dataset)//2,len(dataset)-1):
            sample=dataset[index]
            assert sample['frame_delta'].tolist()==[4]
            assert np.isclose(float(sample['time_seconds'][1]-sample['time_seconds'][0]),.04)
            assert sample['images'].shape==(2,2,3,256,288)
            if args.require_full_depth:
                assert sample['depth_available'].all() and sample['mask_available'].all()
        results.append({'mouse':mouse,'frames':4500,'fps':25,'source_frame_step':4,
            'rgb_images':27000,'rgb_bytes':rgb_bytes,'pseudo_depth_frames_available':available,
            'pfm_headers_checked':checked,'loader_first_middle_last_passed':True,
            'all_sparse_manual_scenes_preserved':len(list(session.parent.parent.glob('frame_*')))==(81 if mouse==1 else 91)})
    report={'passed':True,'require_full_depth':args.require_full_depth,'sessions':results}
    out=BASE/'output/dannce_full_pseudo_depth'/('full_delivery_validation.json' if args.require_full_depth else '25fps_storage_validation.json')
    write_json_atomic(out,report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
