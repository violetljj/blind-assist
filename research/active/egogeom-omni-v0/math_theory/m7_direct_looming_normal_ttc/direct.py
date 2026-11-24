"""Small-patch photometric affine warp relative to a rigid tangent plane."""
import cv2
import numpy as np


def level_to_cam(normals, R):
    down = -R[2].astype(float)
    down /= np.linalg.norm(down)
    z = np.array([0., 0., 1.])
    fwd = z - down * (down @ z)
    if np.linalg.norm(fwd) <= .001:
        fwd = np.array([0., -1., 0.]) + down * down[1]
    fwd /= np.linalg.norm(fwd)
    return normals @ np.stack([np.cross(down, fwd), down, fwd])


def static_warp(centers, depth, normal, K0, K1, R0, R1, t0, t1):
    """Projected center and J of each plane homography (old pixel to new)."""
    Ki = np.linalg.inv(K0)
    rays = np.column_stack([centers, np.ones(len(centers))]) @ Ki.T
    dot = np.sum(rays * normal, axis=1)
    ok = np.abs(dot) > .10
    denom = depth * np.where(ok, dot, 1.)
    rel_R = R1.T @ R0
    rel_t = R1.T @ (t0 - t1)
    H = K1 @ (rel_R[None] + rel_t[None, :, None] * normal[:, None, :] / denom[:, None, None]) @ Ki
    mapped = (H @ np.column_stack([centers, np.ones(len(centers))])[:, :, None])[:, :, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        center = mapped[:, :2] / mapped[:, 2:3]
        jac = (H[:, :2, :2] - center[:, :, None] * H[:, 2:3, :2]) / mapped[:, 2:, None]
    return center, jac, ok & np.isfinite(jac).all(axis=(1, 2)) & (mapped[:, 2] > .05)


def gather(im, xy):
    return cv2.remap(im, xy.astype(np.float32), None, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def gradients(image):
    return [cv2.Sobel(image, cv2.CV_32F, int(x), int(not x), ksize=3, scale=1/8) for x in (True, False)]


def affine_ic(old, new, grad, x0, center, J, radius, locked=False, steps=4):
    """Batched IC-GN with fixed template. Locked means no center update."""
    side = np.linspace(-radius, radius, 7)
    offsets = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
    template_xy = x0[:, None, :] + offsets
    template = gather(old, template_xy).astype(float)
    dx, dy = (gather(ch, template_xy).astype(float) for ch in grad)
    unit = offsets / radius
    phi = unit if locked else np.column_stack([unit, np.ones(len(unit))])
    design = np.concatenate([dx[:, :, None]*phi, dy[:, :, None]*phi], axis=2)
    gram = design.transpose(0, 2, 1) @ design
    eigen = np.linalg.eigvalsh(gram)
    ok = (eigen[:, 0] > 1.) & (eigen[:, 0] > .0001 * eigen[:, -1])
    row, col = old.shape
    ok &= ((template_xy >= 1) & (template_xy <= [col-2, row-2])).all(axis=(1, 2))
    inverse = np.linalg.inv(gram + np.eye(gram.shape[-1])[None] * 1.e-6)
    pinv = inverse @ design.transpose(0, 2, 1)
    warp, pos = J.copy(), center.copy()
    for iteration in range(steps+1):
        mapped = pos[:, None, :] + offsets @ warp.transpose(0, 2, 1)
        ok &= ((mapped >= 1) & (mapped <= [col-2, row-2])).all(axis=(1, 2))
        residual = gather(new, mapped).astype(float) - template
        if iteration == steps:
            break
        delta = (pinv @ residual[:, :, None]).reshape(len(pos), 2, -1)
        inc = np.eye(2)[None] + delta[:, :, :2] / radius
        valid = (np.linalg.det(inc) > .4) & (np.max(np.abs(delta), axis=(1, 2)) < radius)
        ok &= valid
        inc[~valid] = np.eye(2)
        delta[~valid] = 0.
        warp = warp @ np.linalg.inv(inc)
        if not locked:
            pos -= (warp @ delta[:, :, 2:3])[:, :, 0]
    # log area expansion / 2, measured relative to static tangent-plane Jacobian.
    det = np.linalg.det(warp)
    base = np.linalg.det(J)
    ok &= (det > .25) & (base > .25) & (det < 4.) & (base < 4.)
    log_scale = .5*np.log(np.maximum(det, 1.e-8) / np.maximum(base, 1.e-8))
    c = np.zeros(gram.shape[-1])
    c[0] = c[len(c)//2+1] = .5/radius
    variance = (c @ inverse @ c) * np.maximum(4., (residual**2).sum(axis=1)/(len(offsets)-len(c)))
    return log_scale, np.sqrt(variance), ok
