"""Check depth direction, abstention and strict cutoff with a known textured plane."""
from pathlib import Path
import unittest

import cv2
import numpy as np

from holoset_rgb_contradiction_20261119 import RGBVerifier, View
from holoset_rgb_exclude_near_20261119 import RGBNearExcluder


class RGBVerifierTest(unittest.TestCase):
    def test_bounded_near_hypothesis_with_unresolved_far_surface(self):
        cv2.setNumThreads(1)
        shape=(192,320)
        texture=cv2.GaussianBlur(np.random.default_rng(19).integers(0,255,shape,np.uint8),(5,5),1)
        front=np.full(shape,255,np.uint8)
        front[50:140,120:230]=0
        K=np.array([[320.,0,159.5],[0,320.,95.5],[0,0,1.]])
        views,images=[],{}
        source,cutoff=2_000_000,4_000_000
        for ts in (1_000_000,source,3_000_000):
            pose=np.eye(4)
            pose[0,3]=(ts-source)/1e7
            views.append(View(ts,Path("unused"),K,pose))
            def shift(im,z):
                M=np.array([[1.,0,-320*pose[0,3]/z],[0.,1.,0.]])
                return cv2.warpAffine(im,M,(320,192))
            im=np.where(shift(front,2)>127,shift(texture,2),shift(texture,8))
            images[ts]=cv2.resize(im,(160,96),interpolation=cv2.INTER_AREA).astype(np.float32)
        v,u=np.indices((96,160),np.float32)
        mono=np.full(u.shape,2.,np.float32)
        mono[25:70,60:115]=.70
        rates=[]
        for cls in (RGBVerifier,RGBNearExcluder):
            verifier=cls(views)
            verifier.image=images.__getitem__
            mask,_=verifier.classify(source,cutoff,(2*u+.5,2*v+.5),mono)
            rates.append(mask[33:60,73:102].mean())
            self.assertFalse(mask[10:25,25:47].any())
        self.assertLess(rates[0],.1)
        self.assertGreater(rates[1],.8)

    def test_metric_depth_and_causal_witnesses(self):
        cv2.setNumThreads(1)
        h,w = 192,320
        plane = np.random.default_rng(7).integers(0,255,(h,w),np.uint8)
        plane = cv2.GaussianBlur(plane,(5,5),1.05)  # Resolvable texture after optical sampling.
        plane[14:72,20:82] = 110
        K = np.array([[300.,0,(w-1)/2],[0,300.,(h-1)/2],[0,0,1.]])
        source,cutoff = 2_000_000,4_000_000
        views,images = [],{}
        for ts in range(0,6_000_000,1_000_000):
            pose = np.eye(4)
            pose[0,3]=(ts-source)/1e7
            views.append(View(ts,Path("unused"),K,pose))
            if ts < cutoff:
                shift=-K[0,0]*pose[0,3]/2
                image=cv2.warpAffine(plane,np.array([[1.,0,shift],[0,1.,0]]),(w,h))
                images[ts]=cv2.resize(image,(w//2,h//2),interpolation=cv2.INTER_AREA).astype(np.float32)
        verifier=RGBVerifier(views)
        verifier.image=images.__getitem__  # A forbidden target/future read raises immediately.
        v,u=np.indices((h//2,w//2),np.float32)
        mono=np.full(u.shape,2.,np.float32)
        mono[34:60,58:95]=.85
        maps=2*u+.5,2*v+.5
        mask,info=verifier.classify(source,cutoff,maps,mono)
        self.assertTrue(all(t<cutoff for t in info["witnesses"]))
        self.assertGreater(mask[36:58,60:93].mean(),.80)
        self.assertFalse(mask[mono==2].any())
        self.assertFalse(info["valid"][13:29,17:32].any())
        self.assertAlmostEqual(float(np.nanmedian(info["tri_m"])),2.,delta=.10)
        self.assertFalse(verifier.classify(source,3_000_000,maps,mono)[0].any())
        for view in views:
            view.c2w[:3,3]=0  # Identical optical centres cannot give a metric bound.
        self.assertFalse(RGBVerifier(views).classify(source,cutoff,maps,mono)[0].any())


if __name__=="__main__":
    unittest.main()
