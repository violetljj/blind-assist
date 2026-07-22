from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from contract import read_json, safe_file, sha256, write_json


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--bundle-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--samples",type=int,default=12); args=parser.parse_args()
    if args.output.exists(): raise SystemExit("refusing to overwrite output")
    args.output.mkdir(parents=True)
    receipts=[]
    for source_dir in sorted(args.bundle_root.iterdir()):
        if not (source_dir/"bundle.json").is_file(): continue
        bundle=read_json(source_dir/"bundle.json"); root=Path(bundle["source_root"]); frames=[json.loads(line) for line in (source_dir/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
        indexes=[round(i*(len(frames)-1)/(args.samples-1)) for i in range(args.samples)]
        thumbs=[]
        for index in indexes:
            image=Image.open(safe_file(root,frames[index]["rgb_path"])).convert("RGB"); image.thumbnail((320,240)); thumbs.append((index,image.copy()))
        sheet=Image.new("RGB",(1280,720),"black"); draw=ImageDraw.Draw(sheet)
        for slot,(index,image) in enumerate(thumbs):
            x=(slot%4)*320; y=(slot//4)*240; sheet.paste(image,(x,y)); draw.rectangle((x,y,x+92,y+20),fill="black"); draw.text((x+4,y+3),f"frame {index}",fill="white")
        path=args.output/f"{bundle['source']['source_id']}.jpg"; sheet.save(path,quality=92)
        receipts.append({"source_id":bundle["source"]["source_id"],"contact_sheet":path.name,"sha256":sha256(path),"frame_indexes":indexes,"review_workflow":"ustrf_event_review_v1","candidate_alerts_visible":False})
    write_json(args.output/"review_inputs.json",{"schema":"blindassist_ustrf_sensor_replay_review_inputs_v1","inputs":receipts,"production_authority":False})
    print(json.dumps({"sources":len(receipts),"output":str(args.output)})); return 0


if __name__=="__main__": raise SystemExit(main())
