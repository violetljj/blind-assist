"""Invocation-scoped reuse for the serial UE X73 replay call.

Use ``with cached_replay_inputs(): replay.predict_episode(...)``. The cache
lives for one complete causal prefix only; it never carries a tracker result
into a longer prefix. Original functions and mutable-result isolation are
restored/preserved. This context patches module functions and therefore belongs
only in the existing single-request, serial replay worker.
"""
from contextlib import contextmanager
from copy import deepcopy

import ue_dtr_replay as replay


@contextmanager
def cached_replay_inputs():
    metric_original = replay.x24.predict_episode
    depth_original = replay.load_linear_depth
    receipt_original = replay.x24.load_receipt
    metric_cache, depth_cache, receipt_cache = {}, {}, {}
    counts = {"metric_misses": 0, "metric_hits": 0,
              "depth_misses": 0, "depth_hits": 0,
              "receipt_misses": 0, "receipt_hits": 0}

    def metric(episode, candidates, calibration):
        key = (id(episode), id(candidates), id(calibration))
        if key not in metric_cache:
            counts["metric_misses"] += 1
            metric_cache[key] = metric_original(episode, candidates, calibration)
        else:
            counts["metric_hits"] += 1
        return deepcopy(metric_cache[key])

    def depth(observation, calibration):
        key = (observation.depth.path, observation.depth.sha256, id(calibration))
        if key not in depth_cache:
            counts["depth_misses"] += 1
            depth_cache[key] = depth_original(observation, calibration)
        else:
            counts["depth_hits"] += 1
        return depth_cache[key].copy()

    def receipt(observation, original_cache):
        key = (observation.issued_plan["path"],
               observation.issued_plan["receipt_sha256"],
               observation.episode_id, observation.navigation_session_id,
               observation.issued_plan["authority"])
        if key not in receipt_cache:
            counts["receipt_misses"] += 1
            receipt_cache[key] = receipt_original(observation, original_cache)
        else:
            counts["receipt_hits"] += 1
        return deepcopy(receipt_cache[key])

    replay.x24.predict_episode = metric
    replay.load_linear_depth = depth
    replay.x24.load_receipt = receipt
    try:
        yield counts
    finally:
        replay.x24.predict_episode = metric_original
        replay.load_linear_depth = depth_original
        replay.x24.load_receipt = receipt_original
