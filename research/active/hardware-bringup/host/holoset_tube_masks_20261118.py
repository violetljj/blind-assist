"""Create native HoloSet PV masks from manually traced RGB silhouettes.

The annotations describe only the visible cross-corridor tube. The lower contour
is generally more distinct than the translucent, often clipped upper outline.
Coordinates are original 760 x 428 PV pixels, in x,y order. A downward
translation makes an equal-shape, equal-area non-tube control.
"""
from hashlib import sha256
from pathlib import Path
import json

import cv2
import numpy as np

ROOT = Path("artifacts.local/work/holoset-tube-20261118/masks")
DATA = Path("artifacts.local/work/real-wall-search-20261118/head")

# Clockwise around each visible lower boundary, then back along its top.
# The right-hand continuation behind the wall/door jamb is excluded.
S3 = [
 ((309,0),(311,10),(319,19),(380,37),(449,58),(518,76),(553,83),(561,0),(532,0),(523,10),(516,0)),
 ((329,0),(333,14),(346,20),(413,39),(495,59),(561,73),(607,81),(622,0),(577,0),(563,7),(557,0)),
 ((348,0),(351,13),(361,20),(431,37),(507,57),(593,74),(643,86),(650,0),(612,0),(597,7),(590,0)),
 ((344,0),(349,17),(358,25),(430,45),(519,69),(604,86),(664,100),(681,0),(633,0),(622,9),(614,0)),
 ((335,0),(339,14),(350,22),(431,44),(531,70),(620,89),(679,105),(695,0),(644,0),(628,11),(623,0)),
 ((329,0),(334,16),(346,23),(431,46),(526,73),(633,96),(695,112),(713,0),(655,0),(643,10),(638,0)),
 ((326,0),(332,15),(346,22),(431,43),(530,69),(651,94),(722,110),(741,0),(676,0),(665,9),(659,0)),
 ((336,0),(343,11),(353,15),(443,36),(535,60),(662,86),(759,104),(759,0),(716,0),(703,8),(700,0)),
 ((331,0),(336,11),(345,17),(440,40),(543,65),(650,88),(759,109),(759,0)),
 ((315,0),(323,42),(330,55),(343,61),(446,86),(558,117),(672,144),(759,162),(759,0),(749,0),(739,17),(730,0)),
 ((292,0),(298,42),(307,59),(319,65),(440,99),(563,130),(673,157),(759,174),(759,0),(747,0),(741,23),(727,0)),
 ((263,0),(269,35),(278,51),(294,57),(404,87),(520,118),(650,146),(759,171),(759,0),(751,0),(746,19),(731,0)),
 ((256,0),(260,26),(269,42),(283,48),(401,82),(515,111),(644,142),(759,168),(759,0)),
 ((264,0),(267,16),(275,34),(294,41),(407,75),(529,106),(656,135),(759,157),(759,0)),
 ((303,0),(310,17),(325,22),(435,55),(569,94),(671,124),(759,147),(759,0)),
 ((330,0),(343,6),(452,42),(559,77),(663,111),(759,140),(759,0)),
]
S4 = [
 (),
 ((684,0),(678,67),(720,62),(759,57),(759,0)),
 ((593,0),(590,78),(623,75),(685,64),(759,54),(759,0)),
 ((431,0),(429,77),(469,73),(556,57),(647,39),(735,16),(759,11),(759,0)),
 ((249,0),(254,80),(288,76),(363,62),(454,47),(538,25),(596,9),(602,0)),
 ((144,0),(150,90),(198,85),(290,65),(390,45),(474,22),(520,6),(523,0)),
 ((90,0),(106,107),(155,102),(261,81),(384,51),(478,24),(527,12),(531,0)),
 ((68,0),(84,119),(143,113),(260,94),(370,70),(485,40),(527,26),(529,0)),
 ((17,0),(39,125),(87,120),(197,100),(314,82),(424,58),(520,37),(525,21),(532,0)),
 ((0,0),(0,124),(68,119),(174,102),(278,84),(408,56),(498,37),(511,20),(513,0)),
]


def native_masks(polygon):
    outer = np.zeros((428, 760), np.uint8)
    if polygon:
        cv2.fillPoly(outer, [np.array(polygon, np.int32)], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    # Treat the camera border as unobserved, including the truncated top.
    core = cv2.erode(outer, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return outer, core


def control(mask, outer):
    for shift in range(1, mask.shape[0]):
        if mask[-shift:].any():
            break
        shifted = np.roll(mask, shift, axis=0)
        if not np.any(shifted & outer):
            return shifted, shift
    raise ValueError("No in-frame nonoverlapping downward translation")


def history_frames():
    cases = json.loads((DATA / "alert-review/cases.json").read_text())
    rejected = json.loads((DATA.parent / "holoset-d8/verified.json").read_text())["invalid_history"]
    sequences = {
        "research-center-seq3": "7309b7e6ad861d05d2ba198106ccb47d53ee0460d4e5eff051531c855769563f",
        "research-center-seq4": "616f2885dbdf0ff92c66a5a9f14e04eda88ed19d167be309e817eafd829546b2",
    }
    records = {}
    for seq, expected_digest in sequences.items():
        targets = [case["depth_ts"] for case in cases if case["seq"] == seq]
        timeline = (DATA / seq / "geometry.jsonl").read_text().splitlines()
        rows = []
        for frame in map(json.loads, timeline):
            tick = frame["depth_ts"]
            if tick in rejected[seq] or frame["tag"] == "no_rgb_match":
                continue
            if any(0 <= target - tick <= 30_000_000 for target in targets):
                rows.append(dict(depth_ts=tick, rgb_ts=frame["rgb_ts"], index=len(rows)))
        rgb_order = ",".join(str(row["rgb_ts"]) for row in rows)
        if sha256(rgb_order.encode()).hexdigest() != expected_digest:
            raise ValueError(f"RGB timestamps do not match manually traced frames for {seq}")
        records[seq] = rows
    return records


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = history_frames()
    (ROOT / "histories.json").write_text(json.dumps(records, indent=1))
    allrows = []
    for seq, shapes in zip(("research-center-seq3", "research-center-seq4"), (S3, S4)):
        rows = records[seq]
        if len(rows) != len(shapes):
            raise ValueError("Frozen frame count mismatch")
        for row, shape in zip(rows, shapes):
            outer, core = native_masks(shape)
            comparison, shift = control(outer, outer)
            core_comparison = np.roll(core, shift, axis=0)
            masks = dict(outer=outer, core=core, control=comparison, core_control=core_comparison)
            rgb = cv2.imread(str(DATA / seq / "rgb" / f"{row['rgb_ts']}.png"))
            if rgb is None or rgb.shape[:2] != outer.shape:
                raise ValueError("Missing or unexpected PV image")
            for region, data in masks.items():
                dest = ROOT / region / seq / f"{row['rgb_ts']}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(dest), data):
                    raise IOError(dest)
            allrows.append(dict(seq=seq, **row, outline=shape, status="visible" if shape else "not_visible",
                                control_shift=shift, pixels={k:int(np.count_nonzero(m)) for k,m in masks.items()}))
    (ROOT / "manifest.json").write_text(json.dumps(allrows, indent=1))
    print("frames:", len(allrows), "visible:", sum(r["status"]=="visible" for r in allrows))


if __name__ == "__main__":
    main()
