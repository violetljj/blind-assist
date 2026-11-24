"""Metric/sign and causality checks for the correspondence measurement."""
from pathlib import Path
import cv2
import numpy as np
from build import OUT, dump
from motion import H,W,U,V,image_evidence,motion_channels,pixels,projection

def main():
    cv2.setNumThreads(1)
    rng=np.random.default_rng(6151)
    cur=cv2.GaussianBlur(rng.integers(0,256,(H,W),dtype=np.uint8),(3,3),.7)
    K=np.array([[240.,0,192],[0,240.,144],[0,0,1]])
    R=np.eye(3)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    pp=projection(U,V,K)
    np.testing.assert_allclose(pixels(pp,K),np.stack((U,V),-1),atol=2.e-5)
    # For a plane at 2 m and baseline 5 cm the previous view moves 6 px right.
    prev=cv2.warpAffine(cur,np.float32([[1,0,6],[0,1,0]]),(W,H),borderMode=cv2.BORDER_REFLECT)
    info=image_evidence(cur,prev,K,K,R,R,flow)
    _, close=motion_channels(info,np.full(U.shape,2.),K,R,np.array([.05,0,0]))
    _, wrong=motion_channels(info,np.full(U.shape,8.),K,R,np.array([.05,0,0]))
    middle=np.s_[5:-5,5:-5]
    med=float(np.median(info[2][middle][...,0]))
    c=float(close[middle][...,4].mean())
    w=float(wrong[middle][...,4].mean())
    assert abs(med-6)<.35 and c>w+.4
    first=image_evidence(cur,None,K,K,R,R,flow)
    null=motion_channels(first,np.full(U.shape,2.),K,R,np.zeros(3))[1]
    assert not np.any(null[...,2:])
    # A point on the translation axis has no parallax at any depth.
    for d in (1.,4.):
        np.testing.assert_allclose(pixels(np.array([0.,0.,d+.2]),K),[192,144])
    dump(OUT/"motion_verification.json",dict(median_flow_x_px=med,correct_affinity=c,wrong_affinity=w,
                                            missing_first_frame=True))
    print(med,c,w)

if __name__=="__main__":
    main()
