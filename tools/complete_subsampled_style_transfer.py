"""Translate only newly rendered PseudoSynthetic frames, then atomically publish.

Uses the same checkpoint, prompt, fp16 inference and resize-pad recipe as the
original exports. Geometry is copied byte-for-byte, not inferred from styled RGB.
The production style-transfer status is updated only after a frame is published.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch
from complete_subsampled_blender_datasets import (
    BASE, DATA, OUT, read, validate_scene, verify_original, refresh_metadata, digest,
)
from core23_temporal_format import write_json_atomic
sys.path.insert(0, str(BASE/'img2img-turbo/scripts'))
import transfer_blenderpymouse_pseudosynthetic as transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='')
    args = parser.parse_args()
    jobs = [j for j in read(OUT/'plan.json')['jobs'] if j['transferred']]
    if args.dataset:
        jobs = [j for j in jobs if j['dataset'] == args.dataset]
    checkpoint = transfer.DEFAULT_CHECKPOINT.resolve()
    model = None
    for job in jobs:
        source = DATA/job['dataset']
        destination = DATA/job['transferred']
        original_status = read(OUT/'original'/job['transferred']/'style_transfer_status.json')
        production_status = read(destination/'style_transfer_status.json')
        recipe = read(OUT/'original'/job['transferred']/'metadata.json')['style_transfer']
        assert Path(recipe['checkpoint']).resolve() == checkpoint
        assert recipe['prompt'] == transfer.DEFAULT_PROMPT
        assert recipe['direction'] == 'a2b' and recipe['image_prep'] == 'resize_pad_576x384'
        assert Path(production_status['source']).resolve() == source.resolve()
        # Existing source frames acquire new masks during completion. Copy only absent masks.
        for fid in job['original_frames']:
            from_dir = source/f'frame_{fid:06d}/body_masks'
            to_dir = destination/f'frame_{fid:06d}/body_masks'
            to_dir.mkdir(exist_ok=True)
            for file in from_dir.glob('*.png'):
                target = to_dir/file.name
                if target.exists():
                    assert digest(target) == digest(file), target
                else:
                    shutil.copy2(file, target)
        pending = [fid for fid in job['missing_frames'] if not (destination/f'frame_{fid:06d}').exists()]
        stage = OUT/'style_staging'/job['transferred']
        stage.mkdir(parents=True, exist_ok=True)
        # Recover the narrow crash window between an atomic frame move and status write.
        if (stage/'style_transfer_status.json').exists():
            staged_completed = set(read(stage/'style_transfer_status.json')['completed'])
            for fid in job['missing_frames']:
                scene = destination/f'frame_{fid:06d}'
                if scene.exists():
                    validate_scene(scene, job)
                    for vid in range(len(job['cameras'])):
                        relative = f'{scene.name}/blended_images/{vid:08d}.jpg'
                        if relative not in production_status['completed']:
                            assert relative in staged_completed, relative
                            assert digest(destination/relative) != digest(source/relative)
                            production_status['completed'].append(relative)
            production_status['completed'] = sorted(set(production_status['completed']))
            write_json_atomic(destination/'style_transfer_status.json', production_status)
        if not (stage/'style_transfer_status.json').exists():
            state = dict(production_status, destination=str(stage),
                completed=list(production_status['completed']))
            write_json_atomic(stage/'style_transfer_status.json', state)
            write_json_atomic(stage/'metadata.json', read(destination/'metadata.json'))
        for fid in pending:
            scene = source/f'frame_{fid:06d}'
            target = stage/scene.name
            if not target.exists():
                # Copy to a hidden path first; an interrupted copy is never a frame scene.
                temporary = stage/f'.copy_{scene.name}'
                if temporary.exists():
                    assert temporary.resolve().is_relative_to((OUT/'style_staging').resolve())
                    shutil.rmtree(temporary)  # owned, incomplete copy; source remains intact
                shutil.copytree(scene, temporary, copy_function=shutil.copy2)
                temporary.rename(target)
        if pending:
            if model is None:
                model = transfer.CycleGAN_Turbo(pretrained_path=str(checkpoint))
                model.eval().half()
                transform = transfer.build_transform('resize_pad_576x384')
                embedding = transfer.make_caption_embedding(model, transfer.DEFAULT_PROMPT, torch.float16)
            options = argparse.Namespace(max_images=0, batch_size=1, use_fp16=True,
                jpeg_quality=95, direction='a2b', prompt=transfer.DEFAULT_PROMPT,
                image_prep='resize_pad_576x384')
            transfer.transfer_dataset(source, stage, checkpoint, model, transform, embedding, options)
            stage_status = read(stage/'style_transfer_status.json')
            completed = set(stage_status['completed'])
            for fid in pending:
                scene = stage/f'frame_{fid:06d}'
                validate_scene(scene, job)
                for file in (source/scene.name).rglob('*'):
                    if not file.is_file():
                        continue
                    relative = file.relative_to(source/scene.name)
                    if relative.parts[0] == 'blended_images':
                        if not file.stem.endswith('_masked'):
                            assert (Path(scene.name)/relative).as_posix() in completed
                            assert digest(scene/relative) != digest(file), scene/relative
                    else:
                        assert digest(scene/relative) == digest(file), scene/relative
            for fid in pending:
                scene = stage/f'frame_{fid:06d}'
                target = destination/scene.name
                assert not target.exists(), target
                assert scene.resolve().is_relative_to((OUT/'style_staging').resolve())
                assert target.resolve().parent == destination.resolve()
                scene.rename(target)
                for vid in range(len(job['cameras'])):
                    relative = f'{scene.name}/blended_images/{vid:08d}.jpg'
                    if relative not in production_status['completed']:
                        production_status['completed'].append(relative)
                production_status['completed'].sort()
                production_status['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(destination/'style_transfer_status.json', production_status)
        assert len(set(production_status['completed'])) == job['end_frame']*len(job['cameras'])
        for fid in range(1, job['end_frame']+1):
            validate_scene(destination/f'frame_{fid:06d}', job)
        verified = verify_original(job['transferred'])
        refresh_metadata(job, job['transferred'], complete=True)
        metadata = read(destination/'metadata.json')
        metadata['style_transfer']['rgb_image_count'] = job['end_frame']*len(job['cameras'])
        write_json_atomic(destination/'metadata.json', metadata)
        report = transfer.validate_dataset(source, destination)
        assert report['valid'], report
        report['original_files_hash_verified'] = verified
        write_json_atomic(OUT/f'{job["transferred"]}_validation.json', report)
        print('STYLE_COMPLETION_VALIDATED', job['transferred'], flush=True)


if __name__ == '__main__':
    main()
