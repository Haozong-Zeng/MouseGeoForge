"""Deterministic variable-view selection, before encoder feature computation.

Camera IDs are metadata, not fixed learned identities. No fake/repeated views.
Training sources/splits must be chosen by the caller before invoking this module.
"""
import hashlib
import json
import math
import random


def sample_view_ids(available_ids, *, sample_index, seed=17, max_views=None):
    available = sorted(available_ids)
    if not available or len(set(available)) != len(available):
        raise ValueError('Available cameras must be nonempty and distinct')
    if not isinstance(sample_index, int) or sample_index < 0:
        raise ValueError('sample_index must be a nonnegative integer')
    limit = len(available) if max_views is None else max_views
    if not isinstance(limit,int) or limit < 1:
        raise ValueError('max_views must be a positive integer')
    limit = min(limit,len(available))
    # Balanced counts over a cycle; RNG independently samples combination/order.
    count = 1 + sample_index % limit
    rng = random.Random(f'{seed}:{sample_index}')
    return rng.sample(available,count)


def subset_record(record, available_ids, *, sample_index, scene_scale, seed=17, max_views=None):
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError('A fixed positive scene scale is required for every view count')
    selected = sample_view_ids(available_ids,sample_index=sample_index,seed=seed,max_views=max_views)
    return dict(record, views=selected, diagnostic_view_order=selected.copy(),
                normalization_scale=float(scene_scale))


def view_cache_key(*, scene, view_ids, rgb_revision, encoder_sha256, preprocessing):
    """Order/subset-sensitive identity; caller supplies verified RGB revision.

    Never slice a four-view contextual cache to simulate one-/two-view encoding.
    Equal tensor shapes do not imply equal view identity or identical RGB.
    """
    if not view_ids or len(set(view_ids)) != len(view_ids):
        raise ValueError('Cache identity needs nonempty distinct camera IDs')
    if not rgb_revision or not encoder_sha256:
        raise ValueError('Cache identity requires RGB and encoder provenance')
    payload = dict(scene=str(scene),view_ids=list(view_ids),rgb_revision=rgb_revision,
                   encoder_sha256=encoder_sha256,preprocessing=preprocessing)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
