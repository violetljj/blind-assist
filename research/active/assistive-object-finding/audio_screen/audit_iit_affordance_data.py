"""Verify three native IIT-AFF RGB / categorical pixel-label pairs via ZIP ranges."""
from __future__ import annotations
import argparse
import datetime as dt
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import audit_affordance_data as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", default="2026-10-15T06:15:00+00:00")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    common.SIZE = 5448607854
    common.FILE_ID = "15B0dqmqyKaWRWgZ9xP1lm0tfGUpyRVp3"
    common.PUBLIC_URL = f"https://drive.google.com/file/d/{common.FILE_ID}/view"
    common.LIMIT = 140_000_000  # Reserve at least 10 MB for the completed AGD audit.
    remote = common.Ranges(out, dt.datetime.fromisoformat(args.deadline_utc))
    started = time.monotonic()
    root = "IIT_Affordances_2017/"
    try:
        with zipfile.ZipFile(remote) as archive:
            info_map = {entry.filename: entry for entry in archive.infolist()}

            def get_member(name):
                info = info_map[name]
                remote.get(info.header_offset, min(common.SIZE-1, info.header_offset + 30 +
                           len(info.filename.encode()) + len(info.extra) + info.compress_size + 4096))
                return archive.read(info)

            remote.get(0, 50000)
            labels = [entry for entry in info_map.values() if "/object_labels/" in entry.filename and entry.file_size]
            low = min(entry.header_offset for entry in labels)
            high = max(entry.header_offset+30+len(entry.filename.encode())+len(entry.extra)+entry.compress_size+512 for entry in labels)
            remote.get(low, high)
            split_ids = {}
            for name in ("readme.txt", "train_and_val.txt", "test.txt", "val.txt"):
                data = archive.read(root+name)
                (out/name).write_bytes(data)
                if name != "readme.txt":
                    split_ids[name] = [line.strip().removesuffix(".jpg") for line in data.decode().splitlines() if line.strip()]
            all_labels = {}
            for info in labels:
                body = archive.read(info).decode()
                objects = [[int(x) for x in line.split()] for line in body.splitlines() if line.strip()]
                if any(len(row) != 5 for row in objects):
                    raise ValueError("unexpected native object bbox label format")
                all_labels[Path(info.filename).stem] = objects
            selected, records, tiles = [], [], []
            palette = np.array([[0,0,0],[0,80,255],[255,30,30],[150,150,150],[255,150,0],
                                [0,255,60],[180,0,255],[160,90,20],[0,220,220],[255,0,180]], dtype=np.uint8)
            for index, (object_id, name) in enumerate(((5,"cup"),(4,"knife"),(3,"hammer"))):
                choices = sorted(identifier for identifier in split_ids["test.txt"] if any(
                    row[0] == object_id for row in all_labels[identifier]) and identifier not in selected)
                if not choices:
                    raise ValueError(f"no native test sample for {name}")
                identifier = choices[0]
                selected.append(identifier)
                members = {"image": root+f"rgb/{identifier}.jpg",
                           "pixel_labels": root+f"affordances_labels/{identifier}.txt",
                           "object_boxes": root+f"object_labels/{identifier}.txt"}
                payloads = {kind: get_member(path) for kind, path in members.items()}
                image = Image.open(io.BytesIO(payloads["image"])).convert("RGB")
                image.load()
                pixel_labels = np.loadtxt(io.StringIO(payloads["pixel_labels"].decode()), dtype=np.uint8)
                if pixel_labels.ndim != 2 or image.size != pixel_labels.shape[::-1]:
                    raise ValueError("native categorical mask / RGB dimensions mismatch")
                values, counts = np.unique(pixel_labels, return_counts=True)
                if values.min() < 0 or values.max() > 9:
                    raise ValueError("unrecognized native affordance class ID")
                prefix = f"{index:02d}_{name}_{identifier}"
                (out/f"{prefix}_image.jpg").write_bytes(payloads["image"])
                (out/f"{prefix}_labels.txt").write_bytes(payloads["pixel_labels"])
                (out/f"{prefix}_objects.txt").write_bytes(payloads["object_boxes"])
                Image.fromarray(pixel_labels).save(out/f"{prefix}_labels.png")
                boxes = [row[1:] for row in all_labels[identifier] if row[0] == object_id]
                box_counts = []
                for x1,y1,x2,y2 in boxes:
                    crop = pixel_labels[max(0,y1):min(pixel_labels.shape[0],y2+1),
                                        max(0,x1):min(pixel_labels.shape[1],x2+1)]
                    vs, cs = np.unique(crop, return_counts=True)
                    box_counts.append({str(int(v)):int(c) for v,c in zip(vs,cs)})
                records.append({"object": name, "object_id": object_id, "native_test_id": identifier,
                                "image_size": image.size, "pixel_label_shape": list(pixel_labels.shape),
                                "pixel_label_value_counts": {str(int(v)):int(c) for v,c in zip(values,counts)},
                                "target_object_boxes_native": boxes, "target_box_affordance_counts_inclusive_diagnostic_crop": box_counts,
                                "members": members, "source_sha256": {kind:common.digest(data) for kind,data in payloads.items()},
                                "label_kind": "native integer semantic affordance ID per pixel, not thresholded heatmap"})
                colored = Image.fromarray(palette[pixel_labels]).convert("RGBA")
                colored.putalpha(Image.fromarray(np.where(pixel_labels>0,150,0).astype(np.uint8)))
                overlay = Image.alpha_composite(image.convert("RGBA"), colored).convert("RGB")
                draw = ImageDraw.Draw(overlay)
                for box in boxes:
                    draw.rectangle(box,outline="white",width=3)
                tile = Image.new("RGB",(600,300),"white")
                ImageDraw.Draw(tile).text((4,3),f"{name} | {identifier} | native pixel labels",fill="black")
                for x,pic in ((0,image),(300,overlay)):
                    thumb = ImageOps.contain(pic,(296,250));tile.paste(thumb,(x+(300-thumb.width)//2,28+(250-thumb.height)//2))
                tiles.append(tile)
                print(json.dumps({"sample":name,"id":identifier,"size":image.size,"label_ids":values.tolist(),
                                  "bytes_so_far":remote.downloaded}),flush=True)
            montage=Image.new("RGB",(1200,600),"white")
            for index,tile in enumerate(tiles):
                montage.paste(tile,((index%2)*600,(index//2)*300))
            ImageDraw.Draw(montage).text((605,330),"contain=blue cut=red grasp=green",fill="black")
            ImageDraw.Draw(montage).text((605,350),"hit=purple pound=brown support=cyan",fill="black")
            ImageDraw.Draw(montage).text((605,370),"wrap-grasp=magenta; white=target bbox",fill="black")
            montage.save(out/"categorical_label_montage.jpg",quality=92)
            train_val, val, test = map(set,(split_ids["train_and_val.txt"],split_ids["val.txt"],split_ids["test.txt"]))
            summary={"status":"SUCCESS","selection_rule":"first lexicographic native test ID containing cup, then knife, then hammer; no pixel-content selection",
                     "archive_size":common.SIZE,"native_rgb_images":len(all_labels),
                     "split_counts":{"train_and_val":len(train_val),"val":len(val),"test":len(test),"derived_train_excluding_val":len(train_val-val)},
                     "split_checks":{"val_subset_train_and_val":val<=train_val,"test_overlap_train_val":len(test&train_val)},
                     "affordance_class_ids":{"0":"background","1":"contain","2":"cut","3":"display","4":"engine","5":"grasp","6":"hit","7":"pound","8":"support","9":"wrap-grasp"},
                     "samples":records,"downloaded_body_bytes_including_cache":remote.downloaded,
                     "cpu_s":time.process_time()-remote.started_cpu,"wall_s":time.monotonic()-started,
                     "limits":["Three format samples, not model performance or distribution estimates.","Semantic functional parts do not encode hand contact points, reachability, collision, or safe grasp success.","No model/GPU execution."]}
            common.write_json(out/"summary.json",summary)
            print(json.dumps({k:v for k,v in summary.items() if k!='samples'},indent=2),flush=True)
    finally:
        remote.session.close()


if __name__ == "__main__":
    main()
