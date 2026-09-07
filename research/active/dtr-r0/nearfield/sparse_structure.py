"""Sparse RGB plane sweep under supplied metric ego poses; no evaluator inputs."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def pose_matrix(pose):
    if pose.get("roll", 0) != 0:
        raise ValueError("This probe supports roll-zero poses only")
    p, y = math.radians(pose["pitch"]), math.radians(pose["yaw"])
    cp, sp, cy, sy = math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    # Columns: camera right, down, forward in UE world XYZ.
    rotation = np.array([[-sy, sp*cy, cp*cy], [cy, sp*sy, cp*sy], [0, -cp, sp]])
    return rotation, np.array([pose[k] for k in ("x", "y", "z")])


def project(points, inverse_depth, source_pose, target_pose, width, height, hfov, device):
    """NxH hypotheses, returning source pixels and optical forward depth."""
    f = width / (2 * math.tan(math.radians(hfov/2)))
    rs, ts = pose_matrix(source_pose)
    rt, tt = pose_matrix(target_pose)
    relative = torch.tensor(rs.T @ rt, device=device, dtype=torch.float32)
    offset = torch.tensor(rs.T @ (tt-ts), device=device, dtype=torch.float32)
    rays = torch.stack(((points[:, 0]-(width-1)/2)/f,
                        (points[:, 1]-(height-1)/2)/f, torch.ones_like(points[:, 0])), -1)
    xyz = (rays @ relative.T)[:, None, :] / inverse_depth[None, :, None] + offset
    uv = xyz[:, :, :2] / xyz[:, :, 2:] * f
    uv += torch.tensor([(width-1)/2, (height-1)/2], device=device)
    return uv, xyz[:, :, 2]


def patches(gray, uv):
    """Bilinear nine-pixel patches at arbitrary NxH pixel centres."""
    h, w = gray.shape
    offset_y, offset_x = torch.meshgrid(torch.arange(-4,5,device=gray.device),
        torch.arange(-4,5,device=gray.device), indexing="ij")
    offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), -1)
    grid = uv[..., None, :] + offsets
    grid = grid / torch.tensor([w-1, h-1], device=gray.device) * 2 - 1
    result = F.grid_sample(gray[None,None], grid.reshape(1, -1, 81, 2), align_corners=True)
    return result.reshape(*uv.shape[:-1], 81)


def ncc(reference, candidate):
    a = reference - reference.mean(-1, keepdim=True)
    b = candidate - candidate.mean(-1, keepdim=True)
    return (a*b).sum(-1) / (a.square().sum(-1)*b.square().sum(-1)).sqrt().clamp_min(1e-6)


@torch.inference_mode()
def match(images, poses, hfov=100., device="cuda:0"):
    if len(images) != 3 or len(poses) != 3:
        raise ValueError("Require exactly three chronological images and poses")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    # Input BGR uint8, no native depth or object labels.
    gray = [torch.tensor(i, device=device, dtype=torch.float32).mean(-1)/255 for i in images]
    h, w = gray[-1].shape
    gx = F.pad((gray[-1][:,2:]-gray[-1][:,:-2])/2, (1,1))
    gy = F.pad((gray[-1][2:]-gray[-1][:-2])/2, (0,0,1,1))
    grad = (gx.square()+gy.square()).sqrt()
    grad[:5] = grad[-5:] = 0
    grad[:,:5] = grad[:,-5:] = 0
    values, indices = F.max_pool2d(grad[None,None],16,16,ceil_mode=True,return_indices=True)
    indices = indices.flatten()[values.flatten() >= .04]
    points = torch.stack((indices % w, indices // w), -1).float()
    assert len(points) <= 920
    if not len(points):
        return dict(points=[], depth_m=[], accepted=[], correlation=[], rotation_correlation=[],
            parallax_px=[], margin=[], interval_m=[], gradient=[], inverse_depth_grid=[])
    inv = torch.linspace(.05, 2., 96, device=device)
    reference = patches(gray[-1], points[:,None,:])
    scores, rotation_scores, displacement = [], [], []
    for image, pose in zip(gray[:2], poses[:2]):
        uv, z = project(points, inv, pose, poses[-1], w, h, hfov, device)
        rotated, _ = project(points, torch.tensor([1e-8],device=device), pose, poses[-1],w,h,hfov,device)
        valid = (z>0) & (uv[:,:,0]>=4) & (uv[:,:,0]<w-4) & (uv[:,:,1]>=4) & (uv[:,:,1]<h-4)
        score = ncc(reference, patches(image, uv)).masked_fill(~valid, -1.)
        scores.append(score)
        rotation_scores.append(ncc(reference,patches(image,rotated)).squeeze(1))
        displacement.append(torch.linalg.vector_norm(uv-rotated,dim=-1))
    score = torch.stack(scores).amin(0)  # Both past views must agree.
    best, index = score.max(-1)
    best_inv = inv[index]
    outside = (inv[None,:]-best_inv[:,None]).abs() > .2*best_inv[:,None]
    alternative = score.masked_fill(~outside, -2).amax(-1)
    margin = best-alternative
    plausible = score >= best[:,None]-.03
    lo = inv[None,:].expand_as(score).masked_fill(~plausible,float("inf")).amin(-1)
    hi = inv[None,:].expand_as(score).masked_fill(~plausible,0).amax(-1)
    parallax = displacement[0].gather(1,index[:,None]).squeeze(1)
    translation = np.linalg.norm(pose_matrix(poses[-1])[1]-pose_matrix(poses[0])[1])
    accepted = (best>=.75) & (margin>=.05) & ((hi-lo)<=.5*best_inv) & (parallax>=1.)
    if translation <= 1e-6:
        accepted.zero_()
    def cpu(value):
        return value.detach().cpu().tolist()
    return dict(points=cpu(points),depth_m=cpu(1/best_inv),accepted=cpu(accepted),correlation=cpu(best),
        rotation_correlation=cpu(torch.stack(rotation_scores).amin(0)),parallax_px=cpu(parallax),
        margin=cpu(margin),interval_m=cpu(torch.stack((1/hi,1/lo),-1)),
        gradient=cpu(grad.flatten()[indices]),inverse_depth_grid=cpu(inv))
