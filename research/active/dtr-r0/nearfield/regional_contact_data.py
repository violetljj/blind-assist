"""Public-feature-only regional pairing and deterministic within-frame control."""
import hashlib
import numpy as np


def derangements(raw_features):
    """One reproducible perframe permutation; no ids, splits or labels enter."""
    raw=np.asarray(raw_features,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=4864 or not np.isfinite(raw).all():
        raise ValueError('Expected finite public Bx4864 features')
    output=[]
    for row in raw:
        digest=hashlib.sha256(b'regional-contact-20260923-v1'+row.tobytes()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
        for _ in range(128):
            perm=rng.permutation(64)
            if np.all(perm!=np.arange(64)):
                output.append(perm);break
        else:raise RuntimeError('Derangement construction failed; do not alter seed')
    return np.asarray(output,np.int64)


def paired_inputs(regional_rgb,sensor,train_indices,permutations):
    rgb=np.asarray(regional_rgb,np.float32);sensor=np.asarray(sensor,np.float32)
    if rgb.shape[1:]!=(64,40) or sensor.shape!=(len(rgb),64,6):
        raise ValueError('Expected Bx64x40 regional RGB and Bx64x6 sensor')
    perm=np.asarray(permutations)
    if perm.shape!=(len(rgb),64) or not np.all(np.sort(perm,axis=1)==np.arange(64)):
        raise ValueError('Expected perframe permutation')
    mean=rgb[train_indices].mean(axis=(0,1));std=np.maximum(rgb[train_indices].std(axis=(0,1)),.01)
    normalized=(rgb-mean)/std
    aligned=np.concatenate((normalized,sensor),axis=-1).astype(np.float32)
    shuffled=normalized[np.arange(len(rgb))[:,None],perm]
    misaligned=np.concatenate((shuffled,sensor),axis=-1).astype(np.float32)
    return aligned,misaligned,dict(mean=mean,std=std)
