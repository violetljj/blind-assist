"""Reproject Oxford TLS, infer paired padding arms, and score the time-locked panel."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from cohort import HERE, ROOT, OUT, OX, SRC, SITES, persist, sha

sys.path.insert(0, str(ROOT / "research/active/hardware-bringup/host"))
sys.path.insert(0, str(HERE.parent))
import spires_insert_dev_20261117 as camera
from atlas import level_basis, pin_rays, project_kb, project_pinhole, pull

camera.F = 280.
camera._MAP = None
cv2.setNumThreads(1)
ARMS = ("black", "nearest", "median")
CAM = pin_rays(288, 384, (224., 126., 192., 144.))
NATIVE = (camera.F, camera.F, camera.CX, camera.CY)


def cohort():
    return json.loads((OUT / "cohort.json").read_text())["rows"]


def name(row):
    return row["seq"] + "_" + Path(row["entry"][0]).stem


def geometries(rows):
    for seq in sorted({r["seq"] for r in rows}):
        group = [r for r in rows if r["seq"] == seq]
        poses = camera.Traj(SRC / "seq" / seq / "gt-tum.txt").T([r["t"] for r in group])
        for r, pose in zip(group, poses):
            yield r, pose


class ZBuffer:
    """Accumulate point centres by splat radius; flat square erosion takes the exact local minimum."""
    def __init__(self, pose, centre, radius):
        self.pose = pose
        self.centre = centre
        self.radius = radius
        self.layers = {}

    def add(self, points):
        inside = ((np.abs(points[:, 0] - self.centre[0]) < self.radius) &
                  (np.abs(points[:, 1] - self.centre[1]) < self.radius))
        if not inside.any():
            return
        xyz = (points[inside] - self.pose[:3, 3]) @ self.pose[:3, :3]
        xyz = xyz[xyz[:, 2] > .1]
        z = xyz[:, 2]
        u = camera.F * xyz[:, 0] / z + camera.CX
        v = camera.F * xyz[:, 1] / z + camera.CY
        keep = (u > -8) & (u < camera.W+8) & (v > -8) & (v < camera.H+8)
        u, v, z = np.rint(u[keep]).astype(int)+4, np.rint(v[keep]).astype(int)+4, z[keep]
        height, width = camera.H+8, camera.W+8
        keep = (u >= 0) & (v >= 0) & (u < width) & (v < height)
        index, z = v[keep]*width+u[keep], z[keep]
        radii = np.clip(np.ceil(.005 * camera.F / z).astype(int), 0, 4)
        for radius in np.unique(radii):
            if radius not in self.layers:
                self.layers[radius] = np.full((height, width), np.inf, np.float32)
            pick = radii == radius
            np.minimum.at(self.layers[radius].ravel(), index[pick], z[pick])

    def finish(self):
        depth = np.full((camera.H, camera.W), np.inf, np.float32)
        for radius, points in self.layers.items():
            kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
            view = cv2.erode(points, kernel)[4:-4, 4:-4]
            np.minimum(depth, view, out=depth)
        depth[~np.isfinite(depth) | ~camera.undistort_map()[2]] = 0
        return np.clip(np.rint(depth * 1000.), 0, 65535).astype(np.uint16)


def reference():
    folder = OUT / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    rows = cohort()
    bridge = json.loads((OX / "control_0_1.json").read_text())
    old_poses = np.asarray([r["T"] for r in bridge])
    historical = cv2.imread(str(OX / bridge[0]["depth"]), cv2.IMREAD_UNCHANGED)
    ledger = {}
    for site in SITES:
        targets = {name(r): ZBuffer(p, p[:2, 3], 22.) for r, p in
                   geometries([r for r in rows if r["site"] == site])}
        if site in bridge[0]["seq"]:
            targets["bridge"] = ZBuffer(old_poses[0], old_poses[:, :2, 3].mean(0), 6.)
        source = SRC / "map" / (site+".pcd")
        receipt = {}
        for chunk in camera.S.read_pcd_xyz(source, chunk=350000):
            for buf in targets.values():
                buf.add(chunk)
        for key, buf in targets.items():
            image = buf.finish()
            if key == "bridge":
                mismatches = int(np.count_nonzero(image != historical))
                if mismatches:
                    raise ValueError(f"Historical projection differs on {mismatches} pixels")
                receipt[key] = dict(mismatches=0, pixels=image.size, historical=sha(OX / bridge[0]["depth"]))
            else:
                path = folder / (key+".png")
                if not cv2.imwrite(str(path), image):
                    raise IOError(path)
                receipt[key] = sha(path)
        ledger[site] = dict(map_sha256=sha(source), reference=receipt)
        print(site, "reference complete", flush=True)
    persist(OUT / "reference_receipt.json", dict(sites=ledger, cohort=sha(OUT/"cohort.json")))


def predict():
    from eval_empirical import MODEL, infer, session
    inputs = json.loads((OUT/"input_receipt.json").read_text())
    expected = {(r["seq"], r["name"]): r["sha256"] for r in inputs["files"]}
    assert inputs["cohort"] == sha(OUT/"cohort.json")
    sess = session()
    folder = OUT / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    p, ok = project_kb(CAM @ camera.R_CV.T, camera.K0, camera.D0, np.deg2rad(80))
    ledger = {}
    for r, pose in geometries(cohort()):
        key = name(r)
        path = OUT / "original" / r["seq"] / r["entry"][0]
        assert sha(path) == expected[(r["seq"],r["entry"][0])]
        image, valid = pull(cv2.imread(str(path)), p, ok)
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        median = image.copy()
        median[~valid] = np.rint(np.median(image[valid], axis=0)).astype(np.uint8)
        frames = dict(black=image, nearest=image[tuple(nearest)], median=median)
        rays = CAM @ level_basis(-pose[2,:3]).T
        outputs = {}
        for arm, frame in frames.items():
            assert np.array_equal(frame[valid], image[valid]) and not image[~valid].any()
            outputs[arm], _ = infer(sess, frame, rays, valid)
            if not np.isfinite(outputs[arm]).all():
                raise ValueError(f"Nonfinite prediction: {key}/{arm}")
        dst = folder / (key+".npz")
        np.savez_compressed(dst, valid=valid, **outputs)
        ledger[key] = sha(dst)
    persist(OUT/"prediction_receipt.json", dict(files=ledger, cohort=sha(OUT/"cohort.json"),
            model=sha(MODEL), input_receipt=sha(OUT/"input_receipt.json")))
    print(len(ledger), "paired predictions", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["score"]:
        from score_transfer import score
        score()
    elif sys.argv[1:] == ["reference"]:
        reference()
    elif sys.argv[1:] == ["predict"]:
        predict()
    else:
        raise SystemExit("Choose reference, predict, or score")
