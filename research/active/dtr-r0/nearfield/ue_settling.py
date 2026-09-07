"""Reuse converged rendering history only within an unchanged settled-pose clip."""


def settling_count(case, previous, index, initial=32, default=8, policy='full'):
    if policy not in ('full', 'reuse'):
        raise ValueError('Unknown settling policy: ' + policy)
    if index == 0:
        return initial
    if policy == 'reuse' and previous is not None:
        keys = ('clip_id', 'camera', 'objects')
        if all(key in case and key in previous and case[key] == previous[key] for key in keys):
            return 0
    return case.get('settling_frames', default)
