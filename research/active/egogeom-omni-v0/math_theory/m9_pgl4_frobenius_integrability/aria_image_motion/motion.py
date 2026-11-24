"""Measure translation-induced image displacement after camera rotation compensation."""
import cv2
import numpy as np

H, W = 288, 384
Y, X = np.indices((H, W), dtype=np.float32)
U, V = X[::4, ::4], Y[::4, ::4]


def gray_frame(blob, hw, intr, K, bridge):
    # Aria uses square native rectification. This is its exact frozen crop.
    assert tuple(hw) == (512, 512)
    bgr = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    mode = cv2.INTER_AREA if W < bgr.shape[1] else cv2.INTER_LINEAR
    bgr = cv2.resize(bgr, (W, W), interpolation=mode)[48:336]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    assert np.max(np.abs(gray[::4, ::4].astype(int)-bridge)) <= 1
    np.testing.assert_allclose([K[0,0], K[1,1], K[0,2], K[1,2]],
                               intr*.75-[0,0,0,48], atol=2.e-5, rtol=0)
    return gray


def projection(x, y, K):
    return np.stack(((x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], np.ones_like(x)), axis=-1)


def pixels(points, K):
    return (points[...,:2]/np.maximum(points[...,2:],1e-6) *
            [K[0,0],K[1,1]] + [K[0,2],K[1,2]]).astype(np.float32)


def sample(image, x, y):
    return cv2.remap(image, x.astype(np.float32), y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def image_evidence(current, previous, K, old_K, R, old_R, flow):
    g = current.astype(np.float32)
    mean = cv2.blur(g,(7,7))
    texture = np.sqrt(np.maximum(0.,cv2.blur(g*g,(7,7))-mean*mean))[::4,::4]
    brightness = g[::4,::4]/255.
    shape = U.shape
    if previous is None:
        return brightness, texture, np.zeros((*shape,2),np.float32), np.zeros(shape,bool), np.zeros(shape)
    directions = projection(X,Y,K) @ (R.T @ old_R)
    coord = pixels(directions,old_K)
    x,y = coord[...,0],coord[...,1]
    good = (directions[...,2] > .01) & (x >= 4) & (x < W-4) & (y >= 4) & (y < H-4)
    aligned = sample(previous,x,y)
    aligned[~good] = current[~good]
    direct = flow.calc(current,aligned,None)[::4,::4].copy()
    reverse = flow.calc(aligned,current,None)
    xx, yy = U+direct[...,0], V+direct[...,1]
    fb = np.linalg.norm(direct+sample(reverse, xx, yy),axis=2)
    valid = (good[::4,::4] & (sample(good.astype(np.float32),xx,yy) > .999) &
             (xx >= 4) & (xx < W-4) & (yy >= 4) & (yy < H-4))
    return brightness, texture, direct, valid, fb


def motion_channels(gray_info, z, K, R, delta):
    brightness, texture, flow, valid, fb = gray_info
    cam_points = projection(U,V,K)*z[...,None]
    expected = pixels(cam_points + delta @ R,K)-np.stack((U,V),axis=-1)
    magnitude = np.linalg.norm(expected,axis=-1)
    measured = np.linalg.norm(flow,axis=-1)
    error = np.linalg.norm(expected-flow,axis=-1)
    # With no parallax the near-depth and infinity hypotheses are indistinguishable.
    reliable = valid & (fb<=1.5) & (texture>=4.) & (magnitude>=1.) & ((cam_points+delta@R)[...,2]>.10)
    affinity = np.clip((measured-error)/(magnitude+1.),-1,1)
    visual = np.stack((brightness,np.minimum(texture/40.,2.),valid,reliable,
                       reliable*np.exp(-np.minimum(error,40.)/2.),
                       reliable*affinity, reliable*np.minimum(measured/20.,2.)),axis=-1)
    return np.minimum(magnitude,40.)/20., np.nan_to_num(visual)
