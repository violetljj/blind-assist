"""Fixed Development-only CNH baseline; layout holdout is not a site-held-out test."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from cnh_street_e2e_partitions import guard_rows, planned_split, declared_partition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(roots, partition_plan=None):
    features, labels, rows, sources = [], [], [], []
    query_names = None
    assumptions = None
    for root in map(Path, roots):
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
        guard_rows(manifest['frames'], partition_plan)  # Before NPZ/hash/target access.
        receipt = json.loads((root/'receipt.json').read_text(encoding='utf-8-sig'))
        if receipt.get('status') != 'PASS_DEVELOPMENT_MATERIALIZATION_ONLY':
            raise ValueError('Completed materialization required')
        for name in ('manifest.json', 'observations.npz', 'targets.npz'):
            if receipt['outputs'][name] != sha(root/name):
                raise ValueError('Materialized input hash mismatch: '+name)
        if assumptions is not None and assumptions != manifest['assumptions']:
            raise ValueError('Host sensor assumptions mismatch')
        assumptions = manifest['assumptions']
        with np.load(root/'observations.npz', allow_pickle=False) as obs, np.load(root/'targets.npz', allow_pickle=False) as target:
            keys = obs['frame_key'].astype(str)
            if not np.array_equal(keys, target['frame_key'].astype(str)):
                raise ValueError('Observation/target identity mismatch')
            h, ambient, y = obs['histogram'], obs['ambient'], target['labels']
            if h.shape != (len(keys), 8, 8, 16) or ambient.shape != (len(keys), 8, 8) or y.shape != (len(keys), 6):
                raise ValueError('CNH/target schema mismatch')
            if not np.isfinite(h).all() or not np.isfinite(ambient).all() or not np.isin(y, [-1, 0, 1]).all():
                raise ValueError('Invalid values or UNKNOWN encoding')
            # Histogram subtraction may legitimately produce negative counts.
            x = np.concatenate([np.sign(h)*np.log1p(np.abs(h)), np.log1p(np.maximum(ambient, 0))[..., None]], -1)
            features.append(x.reshape(len(keys), -1).astype(np.float32))
            labels.append(y.astype(np.int8))
        part = manifest['frames']
        if len(part) != len(keys) or [r['frame_key'] for r in part] != list(keys):
            raise ValueError('Manifest identity mismatch')
        if any(r.get('data_role') != 'Development' for r in part):
            raise ValueError('Only explicit Development frames allowed')
        if query_names is not None and query_names != manifest['query_names']:
            raise ValueError('Query order mismatch')
        query_names = manifest['query_names']
        rows.extend(part)
        sources.append(dict(path=str(root.resolve()), hashes={name: sha(root/name) for name in ('manifest.json', 'observations.npz', 'targets.npz', 'receipt.json')}))
    if len({r['frame_key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate frame identities')
    return np.concatenate(features), np.concatenate(labels), rows, query_names, sources


def layout_split(rows):
    """One first lexical layout per host/category trains, second validates.

    This predeclared split uses metadata only. Clips and poses never cross.
    Physical site overlaps deliberately: no independent benchmark authority.
    """
    groups = {}
    identities = {}
    for row in rows:
        key = (row['machine_id'], row['environment_category'])
        layout = row['layout_id']
        if layout in identities and identities[layout] != key:
            raise ValueError('Layout metadata changes within layout')
        identities[layout] = key
        groups.setdefault(key, set()).add(layout)
    if len(groups) != 6 or any(len(v) != 2 for v in groups.values()):
        raise ValueError('Expected two layouts for each of two hosts and three categories')
    train_layouts = sorted(sorted(v)[0] for v in groups.values())
    validation_layouts = sorted(sorted(v)[1] for v in groups.values())
    train = np.array([r['layout_id'] in train_layouts for r in rows])
    return train, ~train, dict(train_layouts=train_layouts, debug_validation_layouts=validation_layouts,
        scope='DEVELOPMENT_LAYOUT_HOLDOUT_WITH_SHARED_PHYSICAL_SITE_NOT_INDEPENDENT_TEST',
        physical_sites=sorted({r['physical_site_id'] for r in rows}),
        rule='First lexical layout per host/category trains; second is debug validation; no selection by labels')


def metrics(y, logits):
    known = y >= 0
    pred = logits >= 0
    truth = y == 1
    counts = dict(tp=int((known & pred & truth).sum()), fp=int((known & pred & ~truth).sum()),
                  fn=int((known & ~pred & truth).sum()), tn=int((known & ~pred & ~truth).sum()))
    return dict(**counts, known_queries=int(known.sum()), unknown_queries=int((~known).sum()),
                accuracy=float(((pred == truth) & known).sum()/known.sum()) if known.any() else None)


def run(roots, output, epochs=30, seed=20260924, partition_plan=None, expected_frames=1920):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    started = time.monotonic()
    if epochs != 30:
        raise ValueError('This interface run fixes 30 epochs; no sweep or outcome-based extension')
    plan=json.loads(Path(partition_plan).read_text(encoding='utf-8-sig')) if partition_plan else None
    x, y, rows, query_names, sources = load_inputs(roots,plan)
    if plan is None and any(declared_partition(row) is not None for row in rows):
        raise ValueError('Authored partitions require an explicit frozen partition plan')
    if len(rows) != expected_frames or expected_frames < 1 or (plan is None and expected_frames != 1920):
        raise ValueError('Frame count differs from explicitly bounded run')
    train, val, split = planned_split(rows,plan) if plan is not None else layout_split(rows)
    if partition_plan:split['partition_plan_sha256']=sha(partition_plan)
    known = y >= 0
    if not known[train].any() or not known[val].any():
        raise ValueError('No evaluable targets for training or debug validation')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU baseline requested, CUDA unavailable')
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fresh or runtime-precreated empty output directory required')
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    # Statistics use only training observations, never labels or validation.
    mean = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 1e-3)
    tx = torch.as_tensor((x-mean)/scale, device=device)
    ty = torch.as_tensor(np.maximum(y, 0).astype(np.float32), device=device)
    mask = torch.as_tensor(known, device=device)
    ti = torch.as_tensor(np.flatnonzero(train), device=device)
    model = torch.nn.Linear(x.shape[1], 6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tx[ti])
        loss = loss_fn(logits, ty[ti])[mask[ti]].mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        logits = model(tx).cpu().numpy()
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite inference output')
    torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    mean=torch.from_numpy(mean), scale=torch.from_numpy(scale), input_dim=x.shape[1],
                    query_names=query_names, seed=seed), output/'model.pt')
    # Actual save/load inference check, not merely successful serialization.
    saved = torch.load(output/'model.pt', weights_only=True)
    reloaded = torch.nn.Linear(x.shape[1], 6).to(device)
    reloaded.load_state_dict(saved['state_dict'])
    with torch.no_grad():
        check = reloaded(tx).cpu().numpy()
    if not np.array_equal(check, logits):
        raise ValueError('Reloaded model differs')
    np.savez_compressed(output/'predictions.npz', frame_key=np.array([r['frame_key'] for r in rows]), logits=logits,
                        split=np.where(train, 'train', 'debug_validation'))
    report = dict(status='PASS_DEVELOPMENT_E2E_INTERFACE', benchmark_eligible=False,
        frame_count=len(rows), train_frames=int(train.sum()), debug_validation_frames=int(val.sum()),
        input_shape=list(tx.shape), input_features='signed_log1p_CNH_histogram_and_log1p_ambient',
        model='Linear 1088->6; Adam lr0.001; 30 full-batch epochs; logits threshold0 fixed',
        seed=seed, query_names=query_names, split=split, source_inputs=sources,
        assumptions='SceneDepth synthetic H3; uniform reflectance0.5/incidence1; no normals, RGB, IDs, geometry, host or layout identifiers in model input',
        claim_limit='Pipeline engineering only; geometry-derived labels not independently precision-admitted; shared Street block and obstacle assets; no real-device/generalization claim',
        losses=losses, train=metrics(y[train], logits[train]), debug_validation=metrics(y[val], logits[val]),
        cohorts={}, device=torch.cuda.get_device_name(), backend='torch_cuda', model_reload='BIT_EXACT',
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), wall_s=time.monotonic()-started,
        code_sha256=sha(__file__), model_sha256=sha(output/'model.pt'), predictions_sha256=sha(output/'predictions.npz'))
    for host in sorted({r['machine_id'] for r in rows}):
        subset = val & np.array([r['machine_id']==host for r in rows])
        report['cohorts'][host] = metrics(y[subset], logits[subset])
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','frame_count','train_frames','debug_validation_frames','train','debug_validation','wall_s')}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partition-plan',type=Path)
    p.add_argument('--expected-frames',type=int,default=1920)
    a = p.parse_args()
    run(a.inputs, a.output,partition_plan=a.partition_plan,expected_frames=a.expected_frames)
